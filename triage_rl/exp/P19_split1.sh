#!/bin/bash

: "${HF_TOKEN:?export HF_TOKEN before running}"
: "${WANDB_API_KEY:?export WANDB_API_KEY before running}"

set -x

export ACCELERATE_LOG_LEVEL=info
export HYDRA_FULL_ERROR=1

# Skip vLLM's deep_gemm FP8 kernel warmup. The deep_gemm package needs the CUDA
# toolkit (nvcc + headers) discoverable via CUDA_HOME, which isn't installed in
# this conda env. We don't use FP8 (BF16 model, fp32_logits=False), so this is safe.
export VLLM_USE_DEEP_GEMM=0
export VLLM_SKIP_DEEP_GEMM_WARMUP=1

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"

NUM_GPUS=2
NUM_NODES=1

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
LOG_FILENAME="${SCRIPT_DIR}/P19_Split1.txt"
export TORCHINDUCTOR_CACHE_DIR="${SCRIPT_DIR}/.torchinductor_cache"
export TRITON_CACHE_DIR="${SCRIPT_DIR}/.triton_cache"
mkdir -p "${TORCHINDUCTOR_CACHE_DIR}" "${TRITON_CACHE_DIR}"

# Patient draw order. verl passes only data.seed to the sampler and falls back to OS
# entropy when it is unset, so without this a resumed run reshuffles the batches.
data_seed=42

# Prepare data
train_files="${SCRIPT_DIR}/data/p19_split1_train.parquet"
test_files="${SCRIPT_DIR}/data/p19_split1_test.parquet"
rollout_data_dir="${SCRIPT_DIR}/rollout_save_folder_p19_split1/"
validation_data_dir="${SCRIPT_DIR}/validation_save_folder_p19_split1/"

# Sampler: rollout-batch-level natural M:1 ratio (mutually exclusive with class-balanced).
use_rollout_class_ratio_sampler=True
rollout_class_ratio_sampler_debug_print=True
# Two passes per epoch so each patient's paired rows are BOTH used per epoch
# (one row per pass via a per-patient queue shuffled at epoch start). Doubles
# num_rollout_batches_per_epoch (e.g. 36 -> 72 for split1 train).
rollout_class_ratio_sampler_use_all_rows=True

# Set exp name and checkpoint
# Path to the SFT checkpoint this RL run starts from. See README section 3 for which
# checkpoint the reported runs used.
model_path="${SFT_CHECKPOINT:?set SFT_CHECKPOINT to your P19 SFT checkpoint}"
project_name="Triage"
exp_name="P19_Split1"
STORAGE_DIR="ckpts"
CHECKPOINTS_DIR="$STORAGE_DIR/triage/$exp_name"
echo "Checkpoints dir: $CHECKPOINTS_DIR"

# Set max prompt and response length
max_prompt_length=$((1024 * 8))
val_max_prompt_length=$((1024 * 10))
max_response_length=$((1024 * 2))
max_token_len_per_gpu=$((2 * (max_prompt_length + max_response_length)))
train_prompt_bsz=256
total_training_steps=150
save_freq=15
test_freq=30
rollout_data_freq=$((save_freq / 2))
warmup_steps=10

train_max_samples=-1
val_max_samples=-1
train_prompt_mini_bsz=32
use_dynamic_bsz=True
n_resp_per_prompt=8
n_resp_per_prompt_val=1
if [ "$n_resp_per_prompt_val" -gt 1 ]; then
    val_do_sample=True
else
    val_do_sample=False
fi

lr=1e-6
weight_decay=0.0
use_kl_in_reward=False
kl_coef=0.001
use_kl_loss=True
kl_loss_coef=0.001
kl_loss_type=mse # unbiased k2 gradient
entropy_coeff=0.0

# Naive GRPO on the batch separation reward only.
# We reuse the GDPO advantage estimator with a single reward key (separation_reward)
# so the batch-level separation_reward computed in the trainer is picked up via
# ``algorithm.gdpo_reward_keys``. With one key this reduces to per-group GRPO
# on separation_reward (modulo a final masked_whiten step inside the GDPO estimator).
# The per-group std division of the GRPO advantage is disabled via
# ``algorithm.norm_adv_by_std_in_grpo=False`` (Dr. GRPO style).
adv_estimator=gdpo
gdpo_reward_keys='[separation_reward]'
gdpo_reward_weights='[1.0]'

norm_adv_by_std_in_grpo=False # Dr. GRPO Style

# Cross-entropy weight λ for the decision-token CE term. The total actor loss
# is L_GDPO + λ·L_CE + kl_loss_coef·L_KL, where L_CE = -log P_actor(gt | prefix)
# at the single token after `## Final Decision\n`. Set 0 to disable.
ce_loss_coef=0.25

loss_mode=vanilla
clip_ratio_low=0.2
clip_ratio_high=0.28


param_offload=False
optim_offload=False
activation_offload=False
ulysses_tp=2
worker_master_port_start=${VERL_WORKER_MASTER_PORT_START:-47100}
worker_master_port_end=${VERL_WORKER_MASTER_PORT_END:-47200}
vllm_master_port_start=${VERL_VLLM_MASTER_PORT_START:-48100}
vllm_master_port_end=${VERL_VLLM_MASTER_PORT_END:-48200}
vllm_master_port_stride=${VERL_VLLM_MASTER_PORT_STRIDE:-10}

rollout_engine=vllm
gpu_memory_utilization=0.8
infer_tp=1
temperature=1.0
top_p=1.0
val_temperature=0.0
val_top_p=1.0
calculate_log_probs=False
num_return_logprobs=0
decision_marker="'## Final Decision'"
# Zero response_mask at the single "0"/"1" token after `## Final Decision\n`
# so that the sampled answer contributes no gradient to the policy loss.
# The reward still depends on the decision via the constrained 2-stage logprobs.
# Set False to keep the answer token in the loss (e.g. for ablation).
mask_decision_answer_token=True

# Reward. The module is resolved from data.data_source, which the parquets set to "triage"
# (or "triage_p19"), i.e. verl/utils/reward_score/triage.py. gdpo_reward_keys selects the single
# batch separation reward, which is the whole learning signal.
reward_manager_name=gdpo
# No sliding pool: the separation reward ranks within the rollout batch only, with no comparison
# against previous batches. Set >0 to keep that many past batches in the pool.
sliding_pool_size=0
sliding_pool_warmup_steps=0
separation_margin=1.0
sample_reward_clip_low=-5.0
label_smoothing_epsilon=0.0

# create log dir
[ ! -d log ] && mkdir -p log

# PREFLIGHT=1 resolves the hydra config and exits without touching a GPU. Worth running after
# editing any override below; a mistyped key otherwise only surfaces at launch.
PREFLIGHT_ARGS=()
if [ "${PREFLIGHT:-0}" = 1 ]; then
    PREFLIGHT_ARGS=(--cfg job --resolve)
    LOG_FILENAME=/dev/null
fi

PYTHONUNBUFFERED=1 python3 -m verl.trainer.main_ppo "${PREFLIGHT_ARGS[@]}" \
    actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes=2560 \
    algorithm.adv_estimator=$adv_estimator \
    +algorithm.gdpo_reward_keys=${gdpo_reward_keys} \
    +algorithm.gdpo_reward_weights=${gdpo_reward_weights} \
    algorithm.norm_adv_by_std_in_grpo=${norm_adv_by_std_in_grpo} \
    algorithm.use_kl_in_reward=$use_kl_in_reward \
    algorithm.kl_ctrl.kl_coef=$kl_coef \
    reward.reward_manager.name=${reward_manager_name} \
    reward.sliding_pool_size=${sliding_pool_size} \
    reward.sliding_pool_warmup_steps=${sliding_pool_warmup_steps} \
    reward.separation_margin=${separation_margin} \
    reward.sample_reward_clip_low=${sample_reward_clip_low} \
    reward.label_smoothing_epsilon=${label_smoothing_epsilon} \
    data.rollout_class_ratio_sampler=${use_rollout_class_ratio_sampler} \
    data.rollout_class_ratio_sampler_debug_print=${rollout_class_ratio_sampler_debug_print} \
    data.rollout_class_ratio_sampler_use_all_rows=${rollout_class_ratio_sampler_use_all_rows} \
    data.shuffle=False \
    data.seed=${data_seed} \
    data.train_files="$train_files" \
    data.val_files="$test_files" \
    data.train_max_samples=${train_max_samples} \
    data.val_max_samples=${val_max_samples} \
    data.train_batch_size=$train_prompt_bsz \
    data.max_prompt_length=${max_prompt_length} \
    data.val_max_prompt_length=${val_max_prompt_length} \
    data.max_response_length=${max_response_length} \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.model.path=$model_path \
    actor_rollout_ref.model.use_fused_kernels=False \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=${ulysses_tp} \
    actor_rollout_ref.actor.fsdp_config.fsdp_size=-1 \
    actor_rollout_ref.actor.fsdp_config.param_offload=${param_offload} \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=${optim_offload} \
    actor_rollout_ref.actor.optim.lr=${lr} \
    actor_rollout_ref.actor.optim.weight_decay=${weight_decay} \
    actor_rollout_ref.actor.optim.lr_scheduler_type=constant \
    actor_rollout_ref.actor.optim.lr_warmup_steps=${warmup_steps} \
    actor_rollout_ref.actor.ppo_mini_batch_size=${train_prompt_mini_bsz} \
    actor_rollout_ref.actor.use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${max_token_len_per_gpu} \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${max_token_len_per_gpu} \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${max_token_len_per_gpu} \
    actor_rollout_ref.actor.use_kl_loss=${use_kl_loss} \
    actor_rollout_ref.actor.kl_loss_coef=${kl_loss_coef} \
    actor_rollout_ref.actor.kl_loss_type=${kl_loss_type} \
    +actor_rollout_ref.actor.ce_loss_coef=${ce_loss_coef} \
    actor_rollout_ref.actor.clip_ratio_low=${clip_ratio_low} \
    actor_rollout_ref.actor.clip_ratio_high=${clip_ratio_high} \
    actor_rollout_ref.actor.entropy_coeff=${entropy_coeff} \
    actor_rollout_ref.actor.policy_loss.loss_mode=${loss_mode} \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.model.enable_activation_offload=${activation_offload} \
    actor_rollout_ref.ref.fsdp_config.param_offload=${param_offload} \
    actor_rollout_ref.rollout.name=${rollout_engine} \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.rollout.gpu_memory_utilization=${gpu_memory_utilization} \
    actor_rollout_ref.rollout.calculate_log_probs=${calculate_log_probs} \
    actor_rollout_ref.rollout.num_return_logprobs=${num_return_logprobs} \
    +actor_rollout_ref.rollout.vllm_master_port_range="[${vllm_master_port_start},${vllm_master_port_end}]" \
    +actor_rollout_ref.rollout.vllm_master_port_stride=${vllm_master_port_stride} \
    +actor_rollout_ref.rollout.decision_marker="${decision_marker}" \
    +actor_rollout_ref.rollout.mask_decision_answer_token=${mask_decision_answer_token} \
    +actor_rollout_ref.rollout.fp32_logits=False \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${infer_tp} \
    actor_rollout_ref.rollout.n=${n_resp_per_prompt} \
    actor_rollout_ref.rollout.temperature=${temperature} \
    actor_rollout_ref.rollout.top_p=${top_p} \
    actor_rollout_ref.rollout.val_kwargs.n=${n_resp_per_prompt_val} \
    actor_rollout_ref.rollout.val_kwargs.do_sample=${val_do_sample} \
    actor_rollout_ref.rollout.val_kwargs.temperature=${val_temperature} \
    actor_rollout_ref.rollout.val_kwargs.top_p=${val_top_p} \
    trainer.balance_batch=False \
    trainer.rollout_data_dir=${rollout_data_dir} \
    trainer.rollout_data_freq=${rollout_data_freq} \
    trainer.validation_data_dir=${validation_data_dir} \
    trainer.val_before_train=False \
    trainer.n_gpus_per_node=${NUM_GPUS} \
    trainer.nnodes=${NUM_NODES} \
    +trainer.worker_master_port_range="[${worker_master_port_start},${worker_master_port_end}]" \
    trainer.logger=['console','wandb'] \
    trainer.project_name="${project_name}" \
    trainer.experiment_name="${exp_name}" \
    trainer.default_local_dir="${CHECKPOINTS_DIR}" \
    trainer.save_freq=${save_freq} \
    trainer.test_freq=${test_freq} \
    trainer.default_hdfs_dir=null \
    trainer.total_training_steps=${total_training_steps} "${@:1}" > >(tee -a ${LOG_FILENAME})
