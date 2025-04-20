#!/bin/bash
set -x

export VLLM_ATTENTION_BACKEND=XFORMERS
export VLLM_USE_V1=0
# Ensure API Key is handled appropriately
# export WANDB_API_KEY=25c95cfb8dfe322ae6d944a369d2ae63b65d9ece

MODEL_PATH=Qwen/Qwen2.5-VL-3B-Instruct #

# Assuming the modified YAML is named grpo_example.yaml in the config directory
# If you renamed it, use --config-name=your_new_name
python3 -m verl.trainer.main_tvg \
    data.train_files=/home/chendong/video-rl/charades_sta/charades_train_verl_abs.json \
    data.val_files=/home/chendong/video-rl/charades_sta/charades_stas_test_reformatted.json \
    data.max_prompt_length=4096 \
    data.max_response_length=2048 \
    data.train_batch_size=8 \
    data.shuffle=true \
    data.filter_overlong_prompts=false \
    actor_rollout_ref.model.path=${MODEL_PATH} \
    actor_rollout_ref.actor.strategy=fsdp \
    actor_rollout_ref.actor.ppo_mini_batch_size=4 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.actor.entropy_coeff=1e-3 \
    actor_rollout_ref.actor.use_kl_loss=true \
    actor_rollout_ref.actor.kl_loss_coef=1e-2 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.ppo_epochs=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.n=4  \
    actor_rollout_ref.rollout.top_k=-1  \
    actor_rollout_ref.rollout.top_p=1.0  \
    actor_rollout_ref.rollout.do_sample=true \
    actor_rollout_ref.rollout.enforce_eager=false \
    actor_rollout_ref.rollout.enable_chunked_prefill=false  \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    critic.ppo_max_token_len_per_gpu=16384 \
    critic.rollout_n=2 \
    trainer.project_name=video_tvg \
    trainer.experiment_name=qwen2_5_vl_3b_tvg \
    trainer.nnodes=1 \
    trainer.n_gpus_per_node=4 \
    trainer.val_before_train=False \
    trainer.save_freq=10 \
    trainer.logger='["console", "wandb"]' 2>&1 | tee tvg.log