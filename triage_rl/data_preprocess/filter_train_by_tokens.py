#!/usr/bin/env python
"""Drop over-length stays from a prompt-only RL train JSON.

verl does not truncate prompts anywhere in the RL path, so a cap below the data maximum
means records have to be removed before training rather than cut during it. They must be
removed in pairs: the sampler asserts that every patient contributes exactly two rows with
one shared label, so dropping one row of a pair makes it fail with "orphans". This script
therefore keys on the stay id and drops both rows together.

Tokenising every prompt is the slow part and does not depend on the cap, so the per-stay
lengths are cached once with the build-cache mode and any cap is then applied instantly.

    python filter_train_by_tokens.py build-cache --rl-dir DIR --prefix eicu
    python filter_train_by_tokens.py report      --caps 12288 16384 24576
    python filter_train_by_tokens.py apply       --rl-dir DIR --prefix eicu --cap 16384

Output feeds TRIAGE_dataset.py as --train_data_source.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter

import numpy as np

_TOK = None


def _init(ckpt):
    global _TOK
    from transformers import AutoTokenizer

    _TOK = AutoTokenizer.from_pretrained(ckpt)


def _len(text):
    # Exactly what AgentLoopBase.apply_chat_template produces, which is what
    # _agent_loop_postprocess measures against the cap.
    return len(
        _TOK.apply_chat_template(
            [{"role": "user", "content": text}], add_generation_prompt=True, tokenize=True
        )
    )


def load_pair(main_json, index_json):
    recs = json.load(open(main_json))
    idx = json.load(open(index_json))
    if len(recs) != len(idx):
        raise SystemExit(f"length mismatch: {len(recs)} records vs {len(idx)} index entries")
    for i, e in enumerate(idx):
        if e["record_idx"] != i:
            raise SystemExit(f"index file is not positional at {i} (record_idx={e['record_idx']})")
    # bundle invariant: 2k / 2k+1 share a stay_id, orders survival_first then death_first
    for k in range(len(idx) // 2):
        a, b = idx[2 * k], idx[2 * k + 1]
        if a["stay_id"] != b["stay_id"]:
            raise SystemExit(f"pair {k} spans two stays ({a['stay_id']} vs {b['stay_id']})")
        if (a["prompt_order"], b["prompt_order"]) != ("survival_first", "death_first"):
            raise SystemExit(f"pair {k} order is {(a['prompt_order'], b['prompt_order'])}")
        if recs[2 * k]["MOR_label"] != recs[2 * k + 1]["MOR_label"]:
            raise SystemExit(f"pair {k} carries two labels")
    return recs, idx


def build_cache(args):
    from multiprocessing import Pool

    cache = {}
    for split in args.splits:
        main = os.path.join(args.rl_dir, f"{args.prefix}_rl_prompt_only_train_{split}.json")
        side = os.path.join(args.rl_dir, f"{args.prefix}_rl_prompt_only_train_{split}_index.json")
        recs, idx = load_pair(main, side)
        ckpt = args.tokenizer or _default_ckpt(split, args.ckpt_root)
        with Pool(args.workers, initializer=_init, initargs=(ckpt,)) as pool:
            lens = pool.map(_len, [r["prompt"][0]["content"] for r in recs], chunksize=64)
        per_stay: dict[str, int] = {}
        per_stay_label: dict[str, int] = {}
        for r, e, n in zip(recs, idx, lens):
            s = str(e["stay_id"])
            per_stay[s] = max(per_stay.get(s, 0), n)          # the binding variant
            per_stay_label[s] = int(r["MOR_label"])
        sd = [b - a for a, b in zip(lens[0::2], lens[1::2])]
        cache[split] = {
            "tokenizer": ckpt,
            "n_records": len(recs),
            "n_stays": len(per_stay),
            "max_minus_min_survival_death": [int(min(sd)), int(max(sd))],
            "stay_len": per_stay,
            "stay_label": per_stay_label,
        }
        print(
            f"[cache] {split}: {len(recs)} records / {len(per_stay)} stays, "
            f"max={max(per_stay.values())}, survival-vs-death token delta "
            f"range={min(sd)}..{max(sd)}"
        )
    os.makedirs(os.path.dirname(args.cache) or ".", exist_ok=True)
    json.dump(cache, open(args.cache, "w"))
    print(f"[cache] wrote {args.cache}")


def _default_ckpt(split, root=None):
    root = root or os.environ.get("CKPT_ROOT", "./ckpt")
    p = f"{root}/{split}_sft_for_RL"
    return p if os.path.isdir(p) else f"{root}/{split}"


def report(args):
    cache = json.load(open(args.cache))
    caps = args.caps or [8192, 10240, 12288, 14336, 16384, 18432, 20480, 22528, 24576]
    print(f"{'cap':>7} | " + " | ".join(f"{s:>18}" for s in args.splits) + " |    5-fold mean")
    print("-" * (9 + 21 * len(args.splits) + 20))
    for cap in caps:
        cells, dp, dpos = [], [], []
        for split in args.splits:
            c = cache[split]
            lens = c["stay_len"]
            labs = c["stay_label"]
            drop = [s for s, n in lens.items() if n > cap]
            npos = sum(1 for s in labs.values() if s == 1)
            dpos_n = sum(1 for s in drop if labs[s] == 1)
            pa, po = 100 * len(drop) / len(lens), 100 * dpos_n / max(1, npos)
            cells.append(f"{len(drop):5d} {pa:5.2f}%/{po:5.2f}%")
            dp.append(pa)
            dpos.append(po)
        print(f"{cap:>7} | " + " | ".join(cells) + f" |  {np.mean(dp):5.2f}% / {np.mean(dpos):5.2f}%")
    print("\nEach cell: stays dropped, % of all stays, % of positives")
    for split in args.splits:
        c = cache[split]
        v = np.array(list(c["stay_len"].values()))
        print(f"  {split}: stays={len(v)} p50={int(np.percentile(v,50))} "
              f"p90={int(np.percentile(v,90))} p99={int(np.percentile(v,99))} max={int(v.max())}")


def apply_cap(args):
    cache = json.load(open(args.cache))
    for split in args.splits:
        main = os.path.join(args.rl_dir, f"{args.prefix}_rl_prompt_only_train_{split}.json")
        side = os.path.join(args.rl_dir, f"{args.prefix}_rl_prompt_only_train_{split}_index.json")
        recs, idx = load_pair(main, side)
        lens = cache[split]["stay_len"]
        if len(lens) != len(recs) // 2:
            raise SystemExit(f"cache/{split} covers {len(lens)} stays but file has {len(recs)//2}")

        keep_stay = {s for s, n in lens.items() if n <= args.cap}
        out, kept_labels, dropped_labels = [], [], []
        for k in range(len(recs) // 2):
            s = str(idx[2 * k]["stay_id"])
            lab = recs[2 * k]["MOR_label"]
            if s in keep_stay:
                out.append(recs[2 * k])
                out.append(recs[2 * k + 1])
                kept_labels.append(lab)
            else:
                dropped_labels.append(lab)

        if len(out) % 2:
            raise SystemExit("odd record count after filtering -- pairing broken")
        if len(out) != 2 * len(kept_labels):
            raise SystemExit("record/stay count disagree after filtering")

        outp = os.path.join(args.out_dir, f"{args.prefix}_rl_prompt_only_train_{split}_le{args.cap}.json")
        os.makedirs(args.out_dir, exist_ok=True)
        json.dump(out, open(outp, "w"))
        kc, dc = Counter(kept_labels), Counter(dropped_labels)
        h = hashlib.sha256(open(outp, "rb").read()).hexdigest()
        print(
            f"[filter] {split} cap={args.cap}: kept {len(kept_labels)}/{len(lens)} stays "
            f"({100*len(kept_labels)/len(lens):.2f}%), dropped {len(dropped_labels)} "
            f"(pos {dc[1]}, neg {dc[0]}); kept pos-rate "
            f"{kc[1]/max(1,len(kept_labels)):.4f} -> {len(out)} records"
        )
        print(f"           {outp}\n           sha256 {h}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["build-cache", "report", "apply"])
    parser.add_argument("--rl-dir",
                    help="directory holding <prefix>_rl_prompt_only_train_split<N>.json; "
                         "required for build-cache and apply")
    parser.add_argument("--cache", default="./stay_token_lengths.json")
    parser.add_argument("--splits", nargs="+", default=[f"split{i}" for i in range(1, 6)])
    parser.add_argument("--cap", type=int, help="apply: keep stays whose longer variant is <= this")
    parser.add_argument("--caps", type=int, nargs="+", help="report: caps to tabulate")
    parser.add_argument("--out-dir", default="./filtered_train_json")
    parser.add_argument("--prefix", default="eicu",
                    help="filename prefix: eicu -> eicu_rl_prompt_only_train_split1.json, mimic4 -> mimic4_...")
    parser.add_argument("--ckpt-root", default=None,
                    help="root of the SFT checkpoints, one directory per split; used only as a tokenizer source")
    parser.add_argument("--tokenizer", help="override; default is each split's SFT_for_RL checkpoint")
    parser.add_argument("--workers", type=int, default=min(36, os.cpu_count() or 8))
    args = parser.parse_args()

    if args.mode in ("build-cache", "apply") and not args.rl_dir:
        raise SystemExit(f"--rl-dir is required for {args.mode}")

    if args.mode == "build-cache":
        build_cache(args)
    elif args.mode == "report":
        report(args)
    else:
        if not args.cap:
            raise SystemExit("--cap is required for apply")
        apply_cap(args)


if __name__ == "__main__":
    main()
