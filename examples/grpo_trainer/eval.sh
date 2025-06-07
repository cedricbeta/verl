set -x
export VLLM_ATTENTION_BACKEND=XFORMERS
export VLLM_USE_V1=0

# MODEL_PATH=/home/chendong/video-rl/Temporal-R1/Temporal-R1-3B-Charades
MODEL_PATH=Qwen/Qwen2.5-VL-7B-Instruct
CHECKPOINT_TO_EVALUATE=/home/chendong/video-rl/verl-video/checkpoints/video_vqa/qwen2_5_vl_7b_vqa_reward_451_0605_rpp/global_step_200 # <<< REPLACE
CG_BENCH_JSON_FILE=/home/chendong/video-rl/cg-bench/reformatted_cg_bench_mini.jsonl # <<< REPLACE
CG_BENCH_VIDEOS_DIR=/home/chendong/video-rl/cg-bench/cg_videos_720p # <<< REPLACE
# CG_BENCH_JSON_FILE=/home/chendong/video-rl/charades_sta/charades_vqa_val.jsonl # <<< REPLACE
# CG_BENCH_VIDEOS_DIR=/home/chendong/video-rl/charades_sta/Charades_v1_480 # <<< REPLACE

VISIBLE_DEVICES="0,1,2,3" # Or just "0" if evaluating on a single GPU

# Base command using main_vqa and its default config/vqa.yaml
# We then override parameters for CG-Bench evaluation.
CUDA_VISIBLE_DEVICES=${VISIBLE_DEVICES} python3 -m verl.trainer.eval_cgbench \
    hydra.run.dir=./cg_bench_hydra_outputs/eval_step_$(basename ${CHECKPOINT_TO_EVALUATE}) \
    +trainer.run_cg_bench_evaluation=true \
    +trainer.cg_bench_json_path=${CG_BENCH_JSON_FILE} \
    +trainer.val_only=true \
    +trainer.val_only_after_cg_bench=true \
    +trainer.cg_bench_output_dir=/home/chendong/video-rl/cg-bench \
    trainer.val_before_train=false \
    trainer.resume_mode="resume_path" \
    trainer.resume_from_path=${CHECKPOINT_TO_EVALUATE} \
    +data.video_base_path=${CG_BENCH_VIDEOS_DIR} \
    data.val_files=null \
    data.train_files=/home/chendong/video-rl/charades_sta/charades_vqa_train.jsonl \
    data.val_files=/home/chendong/video-rl/charades_sta/charades_vqa_val.jsonl \
    data.max_prompt_length=25284 \
    data.max_response_length=2048 \
    actor_rollout_ref.model.path=${MODEL_PATH} \
    actor_rollout_ref.actor.strategy=fsdp \
    actor_rollout_ref.rollout.n=1 \
    +actor_rollout_ref.rollout.max_new_tokens_grounding_eval=70 \
    +actor_rollout_ref.rollout.do_sample_grounding_eval=false \
    +actor_rollout_ref.rollout.max_new_tokens_qa_eval=50 \
    +actor_rollout_ref.rollout.do_sample_qa_eval=false \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=1 \
    +trainer.grounding_weight=0.4 \
    +trainer.qa_accuracy_weight=0.4 \
    +trainer.qa_format_weight=0.2 \
    +trainer.thinking_tag_bonus=0.05 \
    trainer.logger='["console"]' 2>&1 | tee cgbench_eval_$(basename ${CHECKPOINT_TO_EVALUATE}).log