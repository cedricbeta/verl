set -x

gsm8k_train_path=/home/ubuntu/chendong/data/gsm8k/train.parquet
gsm8k_test_path=/home/ubuntu/chendong/data/gsm8k/test.parquet
math_train_path=/home/ubuntu/chendong/data/math/train.parquet
math_test_path=/home/ubuntu/chendong/data/math/test.parquet

train_files="['$gsm8k_train_path', '$math_train_path']"
test_files="['$gsm8k_test_path', '$math_test_path']"

export VLLM_ATTENTION_BACKEND=XFORMERS

huggingface-cli download Qwen/Qwen2.5-7B-Instruct --local-dir /home/ubuntu/chendong/.cache/huggingface/hub/models/Qwen2.5-7B-Instruct &
huggingface-cli download sfairXC/FsfairX-LLaMA3-RM-v0.1 --local-dir /home/ubuntu/chendong/.cache/huggingface/hub/models/FsfairX-LLaMA3-RM-v0.1 &
wait

CUDA_VISIBLE_DEVICES=4,5,6,7 python3 -m verl.trainer.main_ppo \
    data.train_files="$train_files" \
    data.val_files="$test_files" \
    data.train_batch_size=512 \
    data.max_prompt_length=512 \
    data.max_response_length=256 \
    data.filter_overlong_prompts=True \
    data.truncation=error \
    data.return_raw_chat=True \
    actor_rollout_ref.model.path="Qwen/Qwen2.5-7B-Instruct" \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.1 \
    actor_rollout_ref.actor.ppo_mini_batch_size=8 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.7 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    critic.optim.lr=1e-5 \
    critic.model.use_remove_padding=True \
    critic.optim.lr_warmup_steps_ratio=0.05 \
    critic.model.path="Qwen/Qwen2.5-7B-Instruct" \
    critic.model.enable_gradient_checkpointing=True \
    critic.ppo_micro_batch_size_per_gpu=4 \
    critic.model.fsdp_config.param_offload=False \
    critic.model.fsdp_config.optimizer_offload=False \
    reward_model.enable=True \
    reward_model.model.path="sfairXC/FsfairX-LLaMA3-RM-v0.1" \
    reward_model.model.use_remove_padding=True \
    reward_model.model.fsdp_config.param_offload=True \
    reward_model.micro_batch_size_per_gpu=4 \
    algorithm.kl_ctrl.kl_coef=0.001 \
    algorithm.use_kl_in_reward=True \
    trainer.critic_warmup=0 \
    trainer.logger="['console','wandb']" \
    trainer.project_name='verl_examples' \
    trainer.experiment_name='gsm8k_spin' \
    trainer.val_before_train=False \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=1 \
    trainer.save_freq=-1 \
    trainer.test_freq=-1 \
    trainer.total_epochs=30 $@
