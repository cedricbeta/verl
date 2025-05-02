python /home/chendong/video-rl/verl-video/scripts/model_merger.py \
  --backend fsdp \
  --hf_model_path Qwen/Qwen2.5-VL-3B-Instruct \
  --local_dir /home/chendong/video-rl/verl-video/checkpoints/video_tvg/qwen2_5_vl_3b_tvg/global_step_500/actor \
  --target_dir /home/chendong/video-rl/verl-video/hf_step_500