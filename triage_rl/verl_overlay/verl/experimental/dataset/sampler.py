# Copyright 2025 Amazon.com Inc and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# This file has been modified from the original verl source by the authors.
from abc import abstractmethod
from collections.abc import Iterable, Sized
from typing import Any

from omegaconf import DictConfig
from torch.utils.data import Sampler

from verl import DataProto


class AbstractSampler(Sampler[int]):
    """Abstract interface for custom samplers."""

    @abstractmethod
    def __init__(
        self,
        data_source: Sized,
        data_config: DictConfig,
    ):
        pass


class AbstractCurriculumSampler(AbstractSampler):
    """Experimental interface for curriculum learning samplers."""

    @abstractmethod
    def update(self, batch: DataProto) -> None:
        pass


# Shared paired-patient helpers.
#
# Both ``RolloutClassRatioSampler`` and ``ClassBalancedMiniBatchSampler`` rely
# on the dataset's paired-patient layout: each patient occupies exactly two
# consecutive rows (``idx=2k`` and ``idx=2k+1``) that share the same
# ``reward_model.ground_truth``. Patient identity is ``extra_info.index // 2``.
# The downstream sliding-batch separation reward (in ``ray_trainer.py``) and the
# validation paired aggregation (in ``metric_utils.py``) all use this key.


def _parse_binary_label(reward_model: dict[str, Any], row_idx: int) -> int:
    if not isinstance(reward_model, dict):
        raise ValueError(f"reward_model at row {row_idx} must be a dict, got {type(reward_model)}")
    ground_truth = reward_model.get("ground_truth", None)
    if ground_truth is None:
        raise ValueError(f'reward_model["ground_truth"] is missing at row {row_idx}')
    label_str = str(ground_truth).strip()
    if label_str not in {"0", "1"}:
        raise ValueError(
            f'Only binary labels "0"/"1" are supported, got reward_model["ground_truth"]={ground_truth} '
            f"at row {row_idx}"
        )
    return int(label_str)


def _iter_reward_model_column(data_source: Sized) -> Iterable[tuple[int, dict[str, Any]]]:
    dataframe = getattr(data_source, "dataframe", None)
    if dataframe is not None and hasattr(dataframe, "column_names") and "reward_model" in dataframe.column_names:
        for idx, reward_model in enumerate(dataframe["reward_model"]):
            yield idx, reward_model
        return
    for idx in range(len(data_source)):
        sample = data_source[idx]
        if not isinstance(sample, dict) or "reward_model" not in sample:
            raise ValueError(
                "Failed to extract reward_model from data_source. "
                "Expected each sample to be a dict containing 'reward_model'."
            )
        yield idx, sample["reward_model"]


def _iter_paired_metadata(data_source: Sized) -> Iterable[tuple[int, Any, Any]]:
    """Yield ``(row_position, extra_info, reward_model)`` for every dataset row.

    Fast-paths through the underlying ``dataframe`` columns when available so
    the check is O(N) row lookups, not O(N) tokenizer-touching ``__getitem__``
    calls.
    """
    dataframe = getattr(data_source, "dataframe", None)
    if (
        dataframe is not None
        and hasattr(dataframe, "column_names")
        and "extra_info" in dataframe.column_names
        and "reward_model" in dataframe.column_names
    ):
        ei_col = dataframe["extra_info"]
        rm_col = dataframe["reward_model"]
        for row_pos in range(len(rm_col)):
            yield row_pos, ei_col[row_pos], rm_col[row_pos]
        return
    for row_pos in range(len(data_source)):
        sample = data_source[row_pos]
        if not isinstance(sample, dict):
            raise ValueError(
                f"Sample at row {row_pos} is not a dict; cannot verify paired-patient layout."
            )
        yield row_pos, sample.get("extra_info"), sample.get("reward_model")


def _collect_patient_pools(
    data_source: Sized,
    sampler_name: str,
) -> tuple[dict[int, list[int]], dict[int, int], list[int], list[int]]:
    """Single pass over the dataset. Returns:

    * ``patient_to_rows``: ``patient_id -> [row_2k, row_2k+1]`` (sorted).
    * ``patient_to_label``: ``patient_id -> 0|1``.
    * ``class0_patient_ids``: list of patient ids with label 0.
    * ``class1_patient_ids``: list of patient ids with label 1.

    Crashes loudly when the paired-patient invariant is violated:
    each patient must have exactly 2 rows at consecutive ``extra_info.index``
    values that share the same ``reward_model.ground_truth``.
    """
    bucket_rows: dict[int, list[int]] = {}
    bucket_gts: dict[int, set] = {}

    for row_pos, ei, rm in _iter_paired_metadata(data_source):
        if not isinstance(ei, dict) or ei.get("index") is None:
            raise ValueError(
                f"[{sampler_name}] Row {row_pos} is missing extra_info.index. "
                "Every row must carry extra_info.index (set by the data preprocess "
                "script via dataset.map(..., with_indices=True))."
            )
        if not isinstance(rm, dict) or rm.get("ground_truth") is None:
            raise ValueError(
                f"[{sampler_name}] Row {row_pos} is missing reward_model.ground_truth."
            )
        patient_id = int(ei["index"]) // 2
        bucket_rows.setdefault(patient_id, []).append(row_pos)
        bucket_gts.setdefault(patient_id, set()).add(str(rm["ground_truth"]).strip())

    n_patients = len(bucket_rows)
    orphans = sum(1 for rows in bucket_rows.values() if len(rows) < 2)
    triples = sum(1 for rows in bucket_rows.values() if len(rows) > 2)
    gt_mismatches = sum(1 for gts in bucket_gts.values() if len(gts) != 1)

    if orphans > 0 or triples > 0 or gt_mismatches > 0:
        offenders: list[str] = []
        for pid, rows in bucket_rows.items():
            if len(rows) != 2:
                offenders.append(f"patient_id={pid} (n_rows={len(rows)})")
                if len(offenders) >= 5:
                    break
        for pid, gts in bucket_gts.items():
            if len(gts) != 1:
                offenders.append(f"patient_id={pid} (gts={sorted(gts)})")
                if len(offenders) >= 10:
                    break
        raise ValueError(
            f"[{sampler_name}] Paired-patient layout violated. "
            f"Stats over {n_patients} patient buckets (key=extra_info.index // 2): "
            f"orphans (size<2)={orphans}, triples (size>2)={triples}, "
            f"gt-mismatches={gt_mismatches}.\n"
            f"Sample offenders: {offenders}.\n"
            "Each patient must occupy exactly 2 consecutive rows (idx=2k, idx=2k+1) "
            "with the same reward_model.ground_truth. The downstream sliding-batch "
            "separation reward and validation paired aggregation rely on this invariant."
        )

    patient_to_rows: dict[int, list[int]] = {pid: sorted(rows) for pid, rows in bucket_rows.items()}
    patient_to_label: dict[int, int] = {}
    class0_patient_ids: list[int] = []
    class1_patient_ids: list[int] = []
    for pid, gts in bucket_gts.items():
        label = int(next(iter(gts)))
        patient_to_label[pid] = label
        if label == 0:
            class0_patient_ids.append(pid)
        else:
            class1_patient_ids.append(pid)

    print(
        f"[{sampler_name}] paired-patient layout verified: "
        f"{n_patients} patients (class0={len(class0_patient_ids)}, "
        f"class1={len(class1_patient_ids)}), 0 orphans, 0 gt-mismatches."
    )
    return patient_to_rows, patient_to_label, class0_patient_ids, class1_patient_ids


def _shuffle_list(items: list[int], generator) -> list[int]:
    import torch

    if not items:
        return []
    perm = torch.randperm(len(items), generator=generator).tolist()
    return [items[i] for i in perm]


_REASONING_MARKER_SURVIVAL_FIRST = (
    "## Question\nWill the patient experience in-hospital death during this ICU stay?\n\n"
    "Reasoning by the following process:\n"
    "1. If the patient indeed survives, which of the patient's given features might be the cause?"
)
_REASONING_MARKER_DEATH_FIRST = (
    "## Question\nWill the patient experience in-hospital death during this ICU stay?\n\n"
    "Reasoning by the following process:\n"
    "1. If the patient indeed experiences in-hospital death, which of the patient's given features might be the cause?"
)

_P19_REASONING_MARKER_SURVIVAL_FIRST = (
    "Reasoning by the following process:\n"
    "1. If the patient indeed does not experience sepsis onset within the next 6 hours, which of the patient's given features might be the cause?"
)


_P19_REASONING_MARKER_DEATH_FIRST = (
    "Reasoning by the following process:\n"
    "1. If the patient indeed experiences sepsis onset within the next 6 hours, which of the patient's given features might be the cause?"
)

def _iter_prompt_first_content(data_source: Sized) -> Iterable[tuple[int, str]]:
    """Yield ``(row_position, prompt[0]["content"])`` for every dataset row.

    Fast-paths through the underlying ``dataframe`` ``prompt`` column when
    available; otherwise falls back to per-row ``__getitem__``.
    """
    dataframe = getattr(data_source, "dataframe", None)
    if (
        dataframe is not None
        and hasattr(dataframe, "column_names")
        and "prompt" in dataframe.column_names
    ):
        prompt_col = dataframe["prompt"]
        for row_pos in range(len(prompt_col)):
            p = prompt_col[row_pos]
            if hasattr(p, "as_py"):
                p = p.as_py()
            content = ""
            if isinstance(p, list) and p and isinstance(p[0], dict):
                content = str(p[0].get("content", "") or "")
            yield row_pos, content
        return
    for row_pos in range(len(data_source)):
        sample = data_source[row_pos]
        content = ""
        if isinstance(sample, dict):
            p = sample.get("prompt", None)
            if isinstance(p, list) and p and isinstance(p[0], dict):
                content = str(p[0].get("content", "") or "")
        yield row_pos, content


def _classify_reasoning_order(content: str) -> str:
    """Return ``"survival_first"`` / ``"death_first"`` / ``"unknown"``."""
    if _REASONING_MARKER_SURVIVAL_FIRST in content or _P19_REASONING_MARKER_SURVIVAL_FIRST in content:
        return "survival_first"
    if _REASONING_MARKER_DEATH_FIRST in content or _P19_REASONING_MARKER_DEATH_FIRST in content:
        return "death_first"
    return "unknown"


class ClassBalancedMiniBatchSampler(AbstractSampler):
    """Patient-level 1:1 sampler with within-rollout-batch patient uniqueness.

    Each prompt-level mini-batch of size ``ppo_mini_batch_size`` contains
    exactly half samples with ``reward_model.ground_truth == "0"`` and half
    with ``reward_model.ground_truth == "1"``. Within a single rollout batch
    (size ``data.train_batch_size``) every ``patient_id = extra_info.index // 2``
    appears at most once: only one of the patient's two rows is drawn.

    Key behaviours:

    * Operates on **patient pools** (one entry per patient_id), not row pools.
    * Per rollout batch: draw ``train_batch_size // 2`` distinct minority
      patients and ``train_batch_size // 2`` distinct majority patients. For
      each drawn patient pick row ``2k`` or ``2k+1`` uniformly at random.
    * If the remaining patients in either pool are fewer than the next
      rollout batch needs, the pool is reshuffled and the pointer reset
      (oversampling is allowed across rollout batches but never within one).
    * Mini-batches inside a rollout batch are 50:50: ``half_mini`` minority
      rows + ``half_mini`` majority rows, then shuffled within the mini-batch.

    This sampler does NOT expose ``M`` / ``minority_label`` / ``majority_label``,
    so the trainer's ``_apply_class_balance_advantage_scaling`` is a no-op when
    this sampler is active (which is the desired behaviour: 1:1 batches do not
    need the M-aware compensation).

    The rollout batch is built so that consecutive prompts share the same
    50:50 mini-batch invariant. ``actor_rollout_ref.actor.shuffle`` MUST be
    ``False`` (asserted in the trainer) so that the actor's inner mini-batch
    construction does not shuffle this layout away.
    """

    def __init__(
        self,
        data_source: Sized,
        data_config: DictConfig,
        ppo_mini_batch_size: int | None = None,
        seed: int | None = None,
    ):
        import torch

        self.data_source = data_source
        self.data_config = data_config

        if ppo_mini_batch_size is None:
            ppo_mini_batch_size = data_config.get("class_balance_mini_batch_size", None)
        if ppo_mini_batch_size is None:
            raise ValueError("ppo_mini_batch_size must be provided for ClassBalancedMiniBatchSampler.")

        self.ppo_mini_batch_size = int(ppo_mini_batch_size)
        if self.ppo_mini_batch_size <= 0:
            raise ValueError(f"ppo_mini_batch_size must be > 0, got {self.ppo_mini_batch_size}")
        if self.ppo_mini_batch_size % 2 != 0:
            raise ValueError(
                "ClassBalancedMiniBatchSampler requires an even ppo_mini_batch_size to keep 50:50 class balance."
            )

        self.train_batch_size = int(data_config.get("train_batch_size", 0))
        if self.train_batch_size <= 0:
            raise ValueError(f"train_batch_size must be > 0, got {self.train_batch_size}")
        if self.train_batch_size % self.ppo_mini_batch_size != 0:
            raise ValueError(
                f"train_batch_size ({self.train_batch_size}) must be divisible by "
                f"ppo_mini_batch_size ({self.ppo_mini_batch_size}) for strict per-mini-batch balance."
            )
        if self.train_batch_size % 2 != 0:
            raise ValueError(
                f"train_batch_size ({self.train_batch_size}) must be even for 1:1 patient-level balance."
            )

        self.seed = data_config.get("seed", None) if seed is None else seed
        self._torch_generator = torch.Generator()
        self._debug_print = bool(data_config.get("class_balance_sampler_debug_print", False))

        # Verify paired-patient layout and collect patient-level pools.
        (
            self._patient_to_rows,
            self._patient_to_label,
            self._class0_patient_ids,
            self._class1_patient_ids,
        ) = _collect_patient_pools(self.data_source, sampler_name="ClassBalancedMiniBatchSampler")

        if not self._class0_patient_ids:
            raise ValueError('No patients found with class "0".')
        if not self._class1_patient_ids:
            raise ValueError('No patients found with class "1".')

        # Per-rollout-batch patient counts (1:1).
        self.n_per_class_per_rollout_batch = self.train_batch_size // 2
        # Per-mini-batch counts.
        self.half_mini_batch = self.ppo_mini_batch_size // 2
        self.mini_batches_per_rollout = self.train_batch_size // self.ppo_mini_batch_size

        # Within-batch patient uniqueness requires each pool to be at least
        # as large as the per-rollout-batch chunk size. Otherwise we'd be
        # forced to wrap the pool inside a single rollout batch, which would
        # introduce duplicate patient_ids in that batch.
        if len(self._class0_patient_ids) < self.n_per_class_per_rollout_batch:
            raise ValueError(
                f"class0 patient pool has {len(self._class0_patient_ids)} patients, but each "
                f"rollout batch needs {self.n_per_class_per_rollout_batch}. Within-batch patient "
                "uniqueness cannot be guaranteed. Reduce train_batch_size or grow the dataset."
            )
        if len(self._class1_patient_ids) < self.n_per_class_per_rollout_batch:
            raise ValueError(
                f"class1 patient pool has {len(self._class1_patient_ids)} patients, but each "
                f"rollout batch needs {self.n_per_class_per_rollout_batch}. Within-batch patient "
                "uniqueness cannot be guaranteed. Reduce train_batch_size or grow the dataset."
            )

        # Match the original epoch length: same number of total row draws as the
        # row-level sampler ((len(dataset) // train_batch_size) * train_batch_size).
        total_rows = len(self.data_source)
        self.num_samples = (total_rows // self.train_batch_size) * self.train_batch_size
        if self.num_samples == 0:
            raise ValueError(
                f"Dataset is too small for one rollout batch: len(dataset)={total_rows}, "
                f"train_batch_size={self.train_batch_size}."
            )
        self.num_rollout_batches = self.num_samples // self.train_batch_size

        # Stateful iterator information for checkpoint-friendly restoration.
        self._epoch = 0
        self._position = 0
        self._current_epoch_indices: list[int] = []

        if self._debug_print:
            print(
                "[ClassBalancedMiniBatchSampler] "
                f"train_batch_size={self.train_batch_size}, "
                f"ppo_mini_batch_size={self.ppo_mini_batch_size}, "
                f"mini_batches_per_rollout={self.mini_batches_per_rollout}, "
                f"n_per_class_per_rollout_batch={self.n_per_class_per_rollout_batch}, "
                f"|class0_patients|={len(self._class0_patient_ids)}, "
                f"|class1_patients|={len(self._class1_patient_ids)}, "
                f"num_rollout_batches_per_epoch={self.num_rollout_batches}"
            )

    def _draw_patient_chunk(
        self,
        all_patient_ids: list[int],
        pool: list[int],
        pointer: int,
        n: int,
        generator,
    ) -> tuple[list[int], list[int], int]:
        """Take next ``n`` distinct patient ids from ``pool``. Reshuffle and
        reset the pointer first if remaining < ``n`` so the chunk is taken
        from a single contiguous slice (guaranteeing uniqueness within the
        chunk because pool size >= n is asserted at init)."""
        if len(pool) - pointer < n:
            pool = _shuffle_list(all_patient_ids, generator)
            pointer = 0
        chunk = pool[pointer : pointer + n]
        pointer += n
        return chunk, pool, pointer

    def _patient_to_random_row(self, patient_id: int, generator) -> int:
        """Pick row ``2k`` or ``2k+1`` for the patient uniformly at random."""
        import torch

        rows = self._patient_to_rows[patient_id]
        choice = int(torch.randint(low=0, high=len(rows), size=(1,), generator=generator).item())
        return rows[choice]

    def _build_epoch_indices(self) -> list[int]:
        import torch

        if self.seed is None:
            self._torch_generator.seed()
        else:
            self._torch_generator.manual_seed(int(self.seed) + self._epoch)

        gen = self._torch_generator

        class0_pool = _shuffle_list(self._class0_patient_ids, gen)
        class1_pool = _shuffle_list(self._class1_patient_ids, gen)
        class0_pointer = 0
        class1_pointer = 0

        # Determine which class is minority (smaller) just for debug-print labelling.
        n0 = len(self._class0_patient_ids)
        n1 = len(self._class1_patient_ids)
        minority_label_for_debug = 0 if n0 <= n1 else 1

        epoch_indices: list[int] = []

        for batch_idx in range(self.num_rollout_batches):
            class0_chunk, class0_pool, class0_pointer = self._draw_patient_chunk(
                self._class0_patient_ids,
                class0_pool,
                class0_pointer,
                self.n_per_class_per_rollout_batch,
                gen,
            )
            class1_chunk, class1_pool, class1_pointer = self._draw_patient_chunk(
                self._class1_patient_ids,
                class1_pool,
                class1_pointer,
                self.n_per_class_per_rollout_batch,
                gen,
            )

            # Convert each patient to one of its two rows uniformly.
            class0_rows = [self._patient_to_random_row(pid, gen) for pid in class0_chunk]
            class1_rows = [self._patient_to_random_row(pid, gen) for pid in class1_chunk]

            # Distribute into mini-batches: each mini-batch gets ``half_mini``
            # class0 rows + ``half_mini`` class1 rows, shuffled within the mini-batch.
            for mini_idx in range(self.mini_batches_per_rollout):
                start = mini_idx * self.half_mini_batch
                end = start + self.half_mini_batch
                mini = class0_rows[start:end] + class1_rows[start:end]
                perm = torch.randperm(len(mini), generator=gen).tolist()
                epoch_indices.extend([mini[i] for i in perm])

            if self._debug_print and batch_idx < 2:
                # Patient-uniqueness within batch sanity check.
                all_patients_in_batch = list(class0_chunk) + list(class1_chunk)
                n_unique = len(set(all_patients_in_batch))
                print(
                    "[ClassBalancedMiniBatchSampler][debug] "
                    f"epoch={self._epoch}, rollout_batch[{batch_idx}] "
                    f"class0={len(class0_chunk)}, class1={len(class1_chunk)}, "
                    f"unique_patients={n_unique} / {len(all_patients_in_batch)} "
                    f"(minority_label_for_debug={minority_label_for_debug})"
                )

        return epoch_indices

    def __iter__(self):
        if not self._current_epoch_indices or self._position >= len(self._current_epoch_indices):
            self._current_epoch_indices = self._build_epoch_indices()
            self._position = 0
            self._epoch += 1

        while self._position < len(self._current_epoch_indices):
            idx = self._current_epoch_indices[self._position]
            self._position += 1
            yield idx

    def __len__(self) -> int:
        return self.num_samples

    def state_dict(self) -> dict[str, Any]:
        return {
            "epoch": self._epoch,
            "position": self._position,
            "current_epoch_indices": self._current_epoch_indices,
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self._epoch = int(state_dict.get("epoch", 0))
        self._position = int(state_dict.get("position", 0))
        indices = state_dict.get("current_epoch_indices", [])
        self._current_epoch_indices = [int(x) for x in indices]


class RolloutClassRatioSampler(AbstractSampler):
    """Patient-level M:1 sampler with within-rollout-batch patient uniqueness.

    Each rollout batch (size ``data.train_batch_size``) contains
    ``n_minority = round(B / (M + 1))`` minority patients and ``B - n_minority``
    majority patients, where ``M = round(|maj_patients| / |min_patients|)``.
    For each drawn patient one of its two rows (``2k`` or ``2k+1``) is picked
    uniformly at random. Within a single rollout batch every ``patient_id``
    appears at most once.

    No oversampling: each *patient* appears at most once per epoch, and
    contributes exactly one row (the other row is unused that epoch). The
    epoch ends when the minority patient pool is exhausted (any remaining
    majority patients are dropped for that epoch). Across multiple epochs
    each patient's two rows are sampled with equal expected frequency due to
    per-draw row randomisation.

    This sampler exposes ``M``, ``minority_label`` and ``majority_label``
    (read by the trainer's ``_apply_class_balance_advantage_scaling`` to
    compensate the M:1 imbalance via per-sample advantage weights).
    """

    def __init__(
        self,
        data_source: Sized,
        data_config: DictConfig,
        seed: int | None = None,
    ):
        import torch

        self.data_source = data_source
        self.data_config = data_config

        self.train_batch_size = int(data_config.get("train_batch_size", 0))
        if self.train_batch_size <= 0:
            raise ValueError(f"train_batch_size must be > 0, got {self.train_batch_size}")

        self.seed = data_config.get("seed", None) if seed is None else seed
        self._torch_generator = torch.Generator()
        self._debug_print = bool(data_config.get("rollout_class_ratio_sampler_debug_print", False))

        # Verify paired-patient layout and collect patient-level pools.
        (
            self._patient_to_rows,
            self._patient_to_label,
            class0_patient_ids,
            class1_patient_ids,
        ) = _collect_patient_pools(self.data_source, sampler_name="RolloutClassRatioSampler")

        if not class0_patient_ids:
            raise ValueError('No patients found with class "0".')
        if not class1_patient_ids:
            raise ValueError('No patients found with class "1".')

        if len(class0_patient_ids) >= len(class1_patient_ids):
            self.majority_label = 0
            self.minority_label = 1
            self.majority_patient_ids = class0_patient_ids
            self.minority_patient_ids = class1_patient_ids
        else:
            self.majority_label = 1
            self.minority_label = 0
            self.majority_patient_ids = class1_patient_ids
            self.minority_patient_ids = class0_patient_ids

        ratio = len(self.majority_patient_ids) / max(1, len(self.minority_patient_ids))
        self.M = max(1, int(round(ratio)))

        self.n_minority_per_batch = max(1, int(round(self.train_batch_size / (self.M + 1))))
        self.n_majority_per_batch = self.train_batch_size - self.n_minority_per_batch
        if self.n_majority_per_batch <= 0:
            raise ValueError(
                f"Computed n_majority_per_batch={self.n_majority_per_batch} <= 0 from "
                f"train_batch_size={self.train_batch_size}, M={self.M}. "
                "Increase train_batch_size or check class counts."
            )

        # Within-batch patient uniqueness requires each pool to be at least
        # as large as the per-rollout-batch chunk size.
        if len(self.minority_patient_ids) < self.n_minority_per_batch:
            raise ValueError(
                f"Minority patient pool has {len(self.minority_patient_ids)} patients, but each "
                f"rollout batch needs {self.n_minority_per_batch}. Within-batch patient uniqueness "
                "cannot be guaranteed. Reduce train_batch_size or check class counts."
            )
        if len(self.majority_patient_ids) < self.n_majority_per_batch:
            raise ValueError(
                f"Majority patient pool has {len(self.majority_patient_ids)} patients, but each "
                f"rollout batch needs {self.n_majority_per_batch}. Within-batch patient uniqueness "
                "cannot be guaranteed."
            )

        # Two-pass mode: when True, each patient emits BOTH of its paired
        # rows per epoch (one row per pass, popped from a per-patient queue
        # shuffled at epoch start). num_rollout_batches_per_epoch doubles.
        # Within-batch patient uniqueness still holds because each batch is
        # built entirely inside one pass.
        self._use_all_rows = bool(
            data_config.get("rollout_class_ratio_sampler_use_all_rows", False)
        )
        self.n_passes_per_epoch = 2 if self._use_all_rows else 1
        self.num_rollout_batches_per_pass = (
            len(self.minority_patient_ids) // self.n_minority_per_batch
        )
        self.num_rollout_batches = self.n_passes_per_epoch * self.num_rollout_batches_per_pass
        self.num_samples = self.num_rollout_batches * self.train_batch_size
        if self.num_samples == 0:
            raise ValueError(
                f"Dataset too small for one rollout batch: "
                f"len(minority_patients)={len(self.minority_patient_ids)}, "
                f"n_minority_per_batch={self.n_minority_per_batch}."
            )

        # Per-row reasoning-order label (survival_first / death_first / unknown).
        # Populated once at startup; used by the debug print in _build_epoch_indices
        # to show whether each batch is a mix of both reasoning orders or skewed
        # to one. Cheap: one O(N) scan over the prompt column at init.
        self._row_to_reasoning_order: dict[int, str] = {}
        for row_pos, content in _iter_prompt_first_content(self.data_source):
            self._row_to_reasoning_order[row_pos] = _classify_reasoning_order(content)

        # Stateful iterator information for checkpoint-friendly restoration.
        self._epoch = 0
        self._position = 0
        self._current_epoch_indices: list[int] = []

        if self._debug_print:
            order_counts = {"survival_first": 0, "death_first": 0, "unknown": 0}
            for label in self._row_to_reasoning_order.values():
                order_counts[label] = order_counts.get(label, 0) + 1
            print(
                "[RolloutClassRatioSampler] "
                f"majority={self.majority_label} (n_patients={len(self.majority_patient_ids)}), "
                f"minority={self.minority_label} (n_patients={len(self.minority_patient_ids)}), "
                f"M={self.M}, n_minority_per_batch={self.n_minority_per_batch}, "
                f"n_majority_per_batch={self.n_majority_per_batch}, "
                f"use_all_rows={self._use_all_rows}, n_passes_per_epoch={self.n_passes_per_epoch}, "
                f"num_rollout_batches_per_pass={self.num_rollout_batches_per_pass}, "
                f"num_rollout_batches_per_epoch={self.num_rollout_batches}, "
                f"row_reasoning_order_counts={order_counts}"
            )

    def _patient_to_random_row(self, patient_id: int, generator) -> int:
        import torch

        rows = self._patient_to_rows[patient_id]
        choice = int(torch.randint(low=0, high=len(rows), size=(1,), generator=generator).item())
        return rows[choice]

    def _build_epoch_indices(self) -> list[int]:
        import torch

        if self.seed is None:
            self._torch_generator.seed()
        else:
            self._torch_generator.manual_seed(int(self.seed) + self._epoch)

        gen = self._torch_generator

        # Per-patient row queue: shuffled at epoch start. Each time a patient
        # is drawn its next row is popped from the front. After
        # ``n_passes_per_epoch`` passes a patient drawn in every pass has
        # emitted all of its rows in random order. A patient that misses a
        # pass (because its pool's floor-division left it out) simply emits
        # fewer rows this epoch; the per-epoch reshuffle balances that out.
        row_queues: dict[int, list[int]] = {}
        for pid, rows in self._patient_to_rows.items():
            perm = torch.randperm(len(rows), generator=gen).tolist()
            row_queues[pid] = [rows[i] for i in perm]

        epoch_indices: list[int] = []
        global_batch_idx = 0
        for pass_idx in range(self.n_passes_per_epoch):
            # Reshuffle the patient pools independently per pass so the
            # second pass is NOT a literal mirror of the first. Within a
            # pass every batch is a single contiguous slice of the
            # shuffled pool, so within-batch patient uniqueness holds.
            minority_pool = _shuffle_list(self.minority_patient_ids, gen)
            majority_pool = _shuffle_list(self.majority_patient_ids, gen)
            minority_pointer = 0
            majority_pointer = 0

            for batch_idx in range(self.num_rollout_batches_per_pass):
                # No oversampling within a pass: drop the tail batch if either
                # pool can't fill its chunk. (With floor-division above this
                # only fires on the very last batch of the pass.)
                if len(minority_pool) - minority_pointer < self.n_minority_per_batch:
                    break
                mini_min_patients = minority_pool[
                    minority_pointer : minority_pointer + self.n_minority_per_batch
                ]
                minority_pointer += self.n_minority_per_batch

                if len(majority_pool) - majority_pointer < self.n_majority_per_batch:
                    break
                mini_maj_patients = majority_pool[
                    majority_pointer : majority_pointer + self.n_majority_per_batch
                ]
                majority_pointer += self.n_majority_per_batch

                mini_min_rows = [row_queues[pid].pop(0) for pid in mini_min_patients]
                mini_maj_rows = [row_queues[pid].pop(0) for pid in mini_maj_patients]

                mixed_rows = mini_min_rows + mini_maj_rows
                perm = torch.randperm(len(mixed_rows), generator=gen).tolist()
                epoch_indices.extend([mixed_rows[i] for i in perm])

                if self._debug_print:
                    all_patients_in_batch = list(mini_min_patients) + list(mini_maj_patients)
                    n_unique_patients = len(set(all_patients_in_batch))
                    n_surv = sum(
                        1 for r in mixed_rows
                        if self._row_to_reasoning_order.get(r) == "survival_first"
                    )
                    n_death = sum(
                        1 for r in mixed_rows
                        if self._row_to_reasoning_order.get(r) == "death_first"
                    )
                    n_unknown = len(mixed_rows) - n_surv - n_death
                    print(
                        "[RolloutClassRatioSampler][debug] "
                        f"epoch={self._epoch}, pass={pass_idx}, "
                        f"rollout_batch[{batch_idx}] (global={global_batch_idx}) "
                        f"minority({self.minority_label})={len(mini_min_patients)}, "
                        f"majority({self.majority_label})={len(mini_maj_patients)}, "
                        f"unique_patients={n_unique_patients}/{len(all_patients_in_batch)} "
                        f"(distinct={n_unique_patients == len(all_patients_in_batch)}), "
                        f"reasoning_order: survival_first={n_surv}, death_first={n_death}, "
                        f"unknown={n_unknown}"
                    )
                global_batch_idx += 1

        return epoch_indices

    def __iter__(self):
        if not self._current_epoch_indices or self._position >= len(self._current_epoch_indices):
            self._current_epoch_indices = self._build_epoch_indices()
            self._position = 0
            self._epoch += 1
        while self._position < len(self._current_epoch_indices):
            idx = self._current_epoch_indices[self._position]
            self._position += 1
            yield idx

    def __len__(self) -> int:
        return self.num_samples

    def state_dict(self) -> dict[str, Any]:
        return {
            "epoch": self._epoch,
            "position": self._position,
            "current_epoch_indices": self._current_epoch_indices,
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self._epoch = int(state_dict.get("epoch", 0))
        self._position = int(state_dict.get("position", 0))
        indices = state_dict.get("current_epoch_indices", [])
        self._current_epoch_indices = [int(x) for x in indices]
