import os
import sys
import ray
import hydra
from omegaconf import OmegaConf, open_dict
from pprint import pprint, pformat
from copy import deepcopy
import torch
import numpy as np
import json
from collections import defaultdict
from tqdm import tqdm
from typing import Dict, Any, Optional

# VERL project structure imports (adjust paths if necessary)
# Assuming this script is placed where these imports work, e.g., in the same dir as main_vqa.py
# or sys.path is adjusted.
from verl.trainer.ppo.ray_trainer_mc import (
    RayPPOTrainer, ResourcePoolManager, Role, 
    parse_grounding_times, prepare_stage2_inputs_for_item, # Make sure these helpers are accessible
    convert_numpy_to_native # And this one
)
from verl.utils import hf_tokenizer, hf_processor
from verl.utils.fs import copy_to_local
from verl.protocol import DataProto, pad_dataproto_to_divisor, unpad_dataproto
from verl.utils.dataset.vqa_dataset import fetch_video, collate_fn # For S2 clipping
from verl.utils.model import compute_position_id_with_mask
import verl.utils.torch_functional as verl_F
from tensordict import TensorDict # For DataProto batch construction

# Worker classes (dynamic import based on config)
# Reward Manager
try:
    from verl.workers.reward_manager import VQAMultiStageRewardManager
    print("Imported VQAMultiStageRewardManager.")
except ImportError:
    print("ERROR: VQAMultiStageRewardManager not found. Please ensure it's in verl/workers/ or accessible.")
    VQAMultiStageRewardManager = None
    # exit(1) # Comment out exit for importability, but ensure it's available at runtime

# Helper for manual collation if not using DataLoader's collate_fn directly
def manual_collate_fn(data_list: list[dict]) -> dict:
    if not data_list: return {}
    keys = data_list[0].keys()
    batch = {k: [] for k in keys}
    for data_item in data_list:
        for k in keys:
            batch[k].append(data_item[k])
    
    final_batch_tensors = {}
    final_batch_non_tensors = {}

    for key, val_list in batch.items():
        if all(isinstance(v, torch.Tensor) for v in val_list):
            try:
                final_batch_tensors[key] = torch.stack(val_list, dim=0)
            except Exception as e:
                print(f"Warning: Could not stack tensor key '{key}': {e}. Keeping as list in non_tensors.")
                final_batch_non_tensors[key] = val_list
        elif all(isinstance(v, np.ndarray) for v in val_list):
            # For np.array of dicts (like multi_modal_inputs), np.array(val_list, dtype=object) is good
            if val_list and isinstance(val_list[0], dict):
                 final_batch_non_tensors[key] = np.array(val_list, dtype=object)
            else:
                 try: # Try to stack if they are numerical arrays of same shape
                     final_batch_non_tensors[key] = np.stack(val_list)
                 except: # Fallback
                     final_batch_non_tensors[key] = np.array(val_list, dtype=object)
        elif all(isinstance(v, dict) for v in val_list): 
            final_batch_non_tensors[key] = np.array(val_list, dtype=object)
        else:
            try:
                final_batch_non_tensors[key] = np.array(val_list, dtype=object)
            except: 
                final_batch_non_tensors[key] = val_list
    
    final_collated_dict = {}
    final_collated_dict.update(final_batch_tensors)
    final_collated_dict.update(final_batch_non_tensors) # non_tensor_batch in DataProto expects np.arrays for lists
    return final_collated_dict


def transform_jsonl_item_for_pipeline(
    jsonl_item_dict: Dict[str, Any], # This is one line from the user's JSONL, parsed as a dict
    video_base_path: str, 
    video_extension: str,
    s1_grounding_prompt_template: str,
    s2_system_prompt: Optional[str]
) -> Optional[Dict[str, Any]]:
    """
    Transforms a single item from the provided JSONL format to a dictionary
    containing raw components needed for the two-stage pipeline.
    """
    try:
        # video_id = jsonl_item_dict.get("video_id") 
        video_uid = jsonl_item_dict.get("video_id")
        
        # S1 prompt uses the 'action' field from JSONL, which needs to be formatted by the template
        s1_raw_action_text = jsonl_item_dict.get("action")
        
        # S2 prompt uses the 'question' field from JSONL (this field already includes choices)
        s2_user_prompt_text_with_choices = jsonl_item_dict.get("question")
        
        s2_gt_answer_letter = jsonl_item_dict.get("answer")
        temporal_grounding_dict = jsonl_item_dict.get("temporal_grounding")

        if not all([video_uid is not None, video_uid is not None, s1_raw_action_text, 
                    s2_user_prompt_text_with_choices, s2_gt_answer_letter, 
                    temporal_grounding_dict]):
            # print(f"Warning: Skipping JSONL item with video_id {video_id} due to missing essential fields. Item: {jsonl_item_dict}")
            return None

        video_path = os.path.join(video_base_path, f"{video_uid}{video_extension}")
        if not os.path.exists(video_path):
            # print(f"Warning: Video file not found for video_id {video_id}, path: {video_path}. Skipping.")
            return None

        gt_s1_start_time_raw = temporal_grounding_dict.get("start_time")
        gt_s1_end_time_raw = temporal_grounding_dict.get("end_time")

        gt_s1_start_time, gt_s1_end_time = None, None
        if isinstance(gt_s1_start_time_raw, (int, float)) and isinstance(gt_s1_end_time_raw, (int, float)):
            gt_s1_start_time = float(gt_s1_start_time_raw)
            gt_s1_end_time = float(gt_s1_end_time_raw)
            if not (gt_s1_start_time >= 0 and gt_s1_start_time <= gt_s1_end_time):
                # print(f"Warning: Invalid start/end times in temporal_grounding for video_id {video_id}: {temporal_grounding_dict}. Setting to 0,0.")
                gt_s1_start_time, gt_s1_end_time = 0.0, 0.0 
        else:
            # print(f"Warning: Non-numeric or missing start/end times in temporal_grounding for video_id {video_id}: {temporal_grounding_dict}. Setting to 0,0.")
            gt_s1_start_time, gt_s1_end_time = 0.0, 0.0
        
        # Format S1 user prompt using the 'action' field from JSONL
        s1_user_prompt_text_formatted = s1_grounding_prompt_template.format(s1_raw_action_text)
        
        # The S2 prompt from JSONL ('question' field) is already formatted with choices
        s2_user_prompt_text_final = s2_user_prompt_text_with_choices
        
        return {
            # "video_id": video_id, 
            "video_id": video_uid, 
            "video_path": video_path,
            "s1_user_prompt_text": s1_user_prompt_text_formatted,
            "s1_gt_start_time": gt_s1_start_time,     
            "s1_gt_end_time": gt_s1_end_time,         
            # "s1_gt_clue_intervals_all": None, # Not available in this JSONL format directly.
                                               # VQAMultiStageRewardManager doesn't directly use it if start/end provided.
            "s2_user_prompt_text": s2_user_prompt_text_final, 
            "s2_gt_answer_letter": s2_gt_answer_letter, 
            "s2_system_prompt": s2_system_prompt 
        }
    except Exception as e:
        # print(f"Error transforming JSONL item (video_id: {jsonl_item_dict.get('video_id', 'Unknown')}): {e}")
        import traceback
        traceback.print_exc()
        return None

@hydra.main(config_path='config', config_name='vqa', version_base=None) # Uses vqa.yaml by default
def run_cg_bench_evaluation_standalone(config: OmegaConf):
    """
    Main function to set up and run the CG-Bench evaluation.
    """
    if not ray.is_initialized():
        ray.init(runtime_env={'env_vars': {'TOKENIZERS_PARALLELISM': 'true',
                                           'NCCL_DEBUG': config.trainer.get("nccl_debug", "WARN"),
                                           'VLLM_LOGGING_LEVEL': config.trainer.get("vllm_logging_level", "WARN"),
                                           'CUDA_VISIBLE_DEVICES': os.environ.get('CUDA_VISIBLE_DEVICES', '')},
                             "working_dir": os.getcwd()})
    print(f"Ray initialized: {ray.is_initialized()}")

    print("--- Effective Configuration for CG-Bench Evaluation ---")
    pprint(OmegaConf.to_container(config, resolve=True))
    OmegaConf.resolve(config)
    torch.autograd.set_detect_anomaly(config.trainer.get("detect_anomaly", False))

    # --- Get Paths from Config ---
    cg_bench_json_path = config.trainer.cg_bench_json_path
    cg_bench_video_base_path = config.data.video_base_path
    video_extension = config.data.get("video_extension", ".mp4")
    output_dir_template = config.trainer.cg_bench_output_dir

    if not all([cg_bench_json_path, cg_bench_video_base_path, output_dir_template]):
        raise ValueError("Missing required paths in config: trainer.cg_bench_json_path, data.video_base_path, or trainer.cg_bench_output_dir")

    # --- Load and Transform Data ---
    s1_prompt_template = config.data.get("grounding_prompt_template", "Give the query: {}, when does the described content occur in the video?")
    s2_system_prompt_text = config.data.get("system_prompt", '''You FIRST think about the reasoning process as an internal monologue and then provide the final answer.
                        The reasoning process MUST BE enclosed within <think> </think> tags.
                        The final answer MUST BE put in <answer> </answer> tags, containing only the letter of the correct option.''')
    
    print(f"Loading CG-Bench data from JSONL file: {cg_bench_json_path}")
    raw_jsonl_item_dicts = []
    if not os.path.exists(cg_bench_json_path):
        print(f"ERROR: CG-Bench JSONL file not found at {cg_bench_json_path}")
        return
        
    with open(cg_bench_json_path, 'r', encoding='utf-8') as f_in:
        for line_number, line in enumerate(f_in):
            line = line.strip()
            if not line: # Skip empty lines
                continue
            try:
                raw_jsonl_item_dicts.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON from line {line_number + 1} in {cg_bench_json_path}: {e}")
                print(f"Problematic line content: >>>{line}<<<")
                continue # Skip problematic lines
    
    if not raw_jsonl_item_dicts:
        print(f"No data loaded from {cg_bench_json_path}. Please check the file content and path.")
        return

    print(f"Transforming {len(raw_jsonl_item_dicts)} raw items from JSONL...")
    transformed_cg_data = [
        transform_jsonl_item_for_pipeline( # Use the corrected transformation function
            item_dict, 
            cg_bench_video_base_path, 
            video_extension, 
            s1_prompt_template, 
            s2_system_prompt_text
        )
        for item_dict in raw_jsonl_item_dicts # Iterate over the list of dicts from JSONL
    ]
    transformed_cg_data = [item for item in transformed_cg_data if item is not None]


    if not transformed_cg_data:
        print("No data to evaluate after transformation (all items might have been skipped due to errors or missing videos/fields). Exiting.")
        return

    print(f"Successfully transformed {len(transformed_cg_data)} items from CG-Bench JSONL.")

    # --- Initialize Tokenizer, Processor, Workers, and Trainer ---
    local_model_path_tokenizer = copy_to_local(config.actor_rollout_ref.model.path)
    tokenizer = hf_tokenizer(local_model_path_tokenizer, trust_remote_code=config.data.get("trust_remote_code", False))
    processor = hf_processor(local_model_path_tokenizer, use_fast=True, trust_remote_code=config.data.get("trust_remote_code", False))

    strategy = config.actor_rollout_ref.actor.strategy
    ActorRolloutRefWorkerCls, CriticWorkerCls, RewardModelWorkerCls, RayWorkerGroupCls = None, None, None, None
    if strategy == 'fsdp':
        from verl.workers.fsdp_workers import ActorRolloutRefWorker, CriticWorker, RewardModelWorker
        from verl.single_controller.ray import RayWorkerGroup
        ActorRolloutRefWorkerCls, CriticWorkerCls, RewardModelWorkerCls, RayWorkerGroupCls = ActorRolloutRefWorker, CriticWorker, RewardModelWorker, RayWorkerGroup
    elif strategy == 'megatron':
        from verl.workers.megatron_workers import ActorRolloutRefWorker, CriticWorker, RewardModelWorker
        from verl.single_controller.ray.megatron import NVMegatronRayWorkerGroup
        ActorRolloutRefWorkerCls, CriticWorkerCls, RewardModelWorkerCls, RayWorkerGroupCls = ActorRolloutRefWorker, CriticWorker, RewardModelWorker, NVMegatronRayWorkerGroup
    else: raise NotImplementedError(f"Unsupported strategy: {strategy}")

    role_worker_mapping = {Role.ActorRollout: ray.remote(ActorRolloutRefWorkerCls)}
    if config.algorithm.adv_estimator == "gae":
        if CriticWorkerCls: 
            role_worker_mapping[Role.Critic] = ray.remote(CriticWorkerCls)
        else:
            print("Warning: GAE estimator selected, but CriticWorkerCls is None. Critic will not be used.")


    global_pool_id = config.trainer.get("eval_resource_pool_id", 'cg_bench_eval_pool')
    resource_pool_spec = {global_pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes}
    mapping = {Role.ActorRollout: global_pool_id}
    if Role.Critic in role_worker_mapping: mapping[Role.Critic] = global_pool_id
    if config.algorithm.use_kl_in_reward or config.actor_rollout_ref.actor.use_kl_loss: # or if ref policy needed for other reasons
        if hasattr(config.actor_rollout_ref, 'ref'):
             role_worker_mapping[Role.RefPolicy] = ray.remote(ActorRolloutRefWorkerCls)
             mapping[Role.RefPolicy] = global_pool_id
    if config.reward_model.enable:
        if RewardModelWorkerCls:
            role_worker_mapping[Role.RewardModel] = ray.remote(RewardModelWorkerCls)
            mapping[Role.RewardModel] = global_pool_id
        else:
            print("Warning: Reward model enabled in config, but RewardModelWorkerCls is None. RM will not be used.")

    resource_pool_manager = ResourcePoolManager(resource_pool_spec=resource_pool_spec, mapping=mapping)

    if VQAMultiStageRewardManager is None:
        raise ImportError("VQAMultiStageRewardManager is not available. Cannot proceed with evaluation.")

    vqa_eval_reward_fn = VQAMultiStageRewardManager(
        tokenizer=tokenizer,
        grounding_weight=config.trainer.grounding_weight,
        qa_accuracy_weight=config.trainer.qa_accuracy_weight,
        qa_format_weight=config.trainer.qa_format_weight,
        thinking_tag_bonus=config.trainer.get("thinking_tag_bonus", 0.05),
        s1_grounding_tag_regex=config.trainer.get("s1_grounding_tag_regex", r"<answer>(.*?)</answer>"),
        s2_answer_tag_regex=config.trainer.get("s2_answer_tag_regex", r"<answer>\s*([A-Z])\s*</answer>"),
        num_examine=3
    )
    
    eval_config = deepcopy(config)
    # Ensure eval specific settings if needed, e.g. disable training aspects for pure eval
    with open_dict(eval_config): 
        eval_config.trainer.run_cg_bench_evaluation = True # Example flag, not used by RayPPOTrainer directly
        eval_config.trainer.run_custom_evaluation_only = True # Example flag

    trainer_instance = RayPPOTrainer(
        config=eval_config, tokenizer=tokenizer, processor=processor,
        role_worker_mapping=role_worker_mapping, resource_pool_manager=resource_pool_manager,
        ray_worker_group_cls=RayWorkerGroupCls,
        reward_fn=vqa_eval_reward_fn, # For PPO, not directly used in this eval flow if we manually call reward_fn
        val_reward_fn=vqa_eval_reward_fn # Used for _validate or direct calls
    )
    
    print("\n--- Initializing Workers for CG-Bench Evaluation ---")
    trainer_instance.init_workers() 
    
    print("\n--- Loading Checkpoint for CG-Bench Evaluation ---")
    if not (config.trainer.resume_mode == "resume_path" and config.trainer.resume_from_path):
        raise ValueError("For evaluation, config.trainer.resume_mode must be 'resume_path' and "
                         "config.trainer.resume_from_path must point to the checkpoint folder.")
    trainer_instance._load_checkpoint() 
    print(f"--- Checkpoint loaded. Trainer global_steps: {0} ---")
    
    output_dir_final = output_dir_template.format(global_steps=0)
    os.makedirs(output_dir_final, exist_ok=True)

    # --- Perform Evaluation Directly ---
    all_eval_run_metrics_accumulator = defaultdict(list)
    detailed_results_for_saving = []
    
    video_proc_cfg_for_pipeline = {
         "fps": config.data.get("video_sample_fps", 2.0),
         "nframes": config.data.get("video_sample_nframes", None), # If None, smart_nframes uses fps
         "min_frames": config.data.get("video_min_frames", 4),
         "max_frames": config.data.get("video_max_frames",32),
         "min_pixels": config.data.get("video_min_pixels", 128 * 28 * 28), # Example values
         "total_pixels": config.data.get("video_total_pixels", int(128000 * 28 * 28 * 0.9)), # Example values
         "image_factor": config.data.get("video_image_factor", 28), # Example value
    }
    # Ensure one of nframes or fps is primary for fetch_video smart_nframes logic
    if video_proc_cfg_for_pipeline.get("nframes") is not None:
        video_proc_cfg_for_pipeline.pop("fps", None)
    elif video_proc_cfg_for_pipeline.get("fps") is None: # Neither specified
        video_proc_cfg_for_pipeline["fps"] = 2.0 # Default if both are None


    eval_batch_size = config.trainer.get("cg_bench_eval_batch_size", 1) 
    num_pipeline_batches = (len(transformed_cg_data) + eval_batch_size - 1) // eval_batch_size
    
    step_info_prefix_eval_loop = f"[CG-Bench EVAL - Trainer Step {0}]"


    for i_pipeline_batch in tqdm(range(num_pipeline_batches), desc=f"{step_info_prefix_eval_loop} Pipeline Batches"):
        batch_start = i_pipeline_batch * eval_batch_size
        batch_end = min((i_pipeline_batch + 1) * eval_batch_size, len(transformed_cg_data))
        current_pipeline_batch_dicts = transformed_cg_data[batch_start:batch_end]
        current_actual_pipeline_batch_size = len(current_pipeline_batch_dicts)
        print(f"{step_info_prefix_eval_loop} Processing pipeline batch {i_pipeline_batch + 1}/{num_pipeline_batches}, size: {current_actual_pipeline_batch_size}")

        if current_actual_pipeline_batch_size == 0: continue
        
        # Initialize result containers for the current batch
        s1_decoded_texts_for_batch = ["<S1 Gen Fail>"] * current_actual_pipeline_batch_size
        s1_predicted_times_for_batch = [(None, None)] * current_actual_pipeline_batch_size
        s2_decoded_texts_for_batch = ["<S2 Gen Fail>"] * current_actual_pipeline_batch_size
        s2_responses_tensor_for_reward = None
        
        # --- Stage 1: Grounding Generation ---
        s1_gen_inputs_list = []
        original_indices_map_s1 = [] 

        for item_batch_idx, item_data_transformed in enumerate(current_pipeline_batch_dicts):
            try:
                # Stage 1 video uses full video, video_proc_cfg_for_pipeline applies (fps or nframes)
                ele_s1_video = {"video": item_data_transformed["video_path"], **video_proc_cfg_for_pipeline}
                s1_full_video_frames, sample_fps, total_frames = fetch_video(ele_s1_video, image_factor=video_proc_cfg_for_pipeline.get("image_factor", 28), return_video_sample_fps=True)
                
                if s1_full_video_frames is None or s1_full_video_frames.nelement() == 0: 
                    print(f"{step_info_prefix_eval_loop} Warning: S1 full video failed for video_id {item_data_transformed['video_id']}. Skipping this item for S1 gen.")
                    continue 
                
                item_data_transformed["original_video_nframes_s1"] = s1_full_video_frames.shape[0]
                s1_multi_modal_data = {"video": [s1_full_video_frames.numpy()]}

                s1_messages_pipeline = []
                s1_system_p = config.data.get("system_prompt", None)
                if s1_system_p: s1_messages_pipeline.append({"role": "system", "content": s1_system_p})
                s1_message_content = [{"type": "video", "video": item_data_transformed["video_path"]}, {"type": "text", "text": item_data_transformed["s1_user_prompt_text"]}]
                s1_messages_pipeline.append({"role": "user", "content": s1_message_content})
                
                s1_raw_prompt = processor.apply_chat_template(s1_messages_pipeline, add_generation_prompt=True, tokenize=False)
                
                s1_model_inputs_pipeline = processor(text=[s1_raw_prompt], image=None, videos=[s1_full_video_frames], return_tensors="pt")
                s1_input_ids_raw_pipe = s1_model_inputs_pipeline.pop("input_ids")
                s1_attn_mask_raw_pipe = s1_model_inputs_pipeline.pop("attention_mask")
                if "second_per_grid_ts" in s1_model_inputs_pipeline:
                    s1_model_inputs_pipeline.pop("second_per_grid_ts")
                    
                from verl.models.transformers.qwen2_vl import get_rope_index
                s1_pos_ids_pipe_val = [
                        get_rope_index(
                                processor,
                                input_ids=s1_input_ids_raw_pipe[0],
                                image_grid_thw=s1_model_inputs_pipeline.get("image_grid_thw"),
                                video_grid_thw=s1_model_inputs_pipeline.get("video_grid_thw"),
                                second_per_grid_ts=s1_model_inputs_pipeline.get("second_per_grid_ts"),
                                attention_mask=s1_attn_mask_raw_pipe[0],
                            )
                    ]
                s1_pos_ids_pipe_val = s1_pos_ids_pipe_val[0]
                
                s1_input_ids_pipe, s1_attn_mask_pipe, s1_pos_ids_pipe_val = verl_F.postprocess_data(
                    s1_input_ids_raw_pipe, s1_attn_mask_raw_pipe,position_ids=s1_pos_ids_pipe_val,
                    max_length=config.data.max_prompt_length,
                    pad_token_id=tokenizer.pad_token_id, left_pad=True,
                    truncation=config.data.get("truncation", "error")
                )
                # s1_pos_ids_pipe_val = compute_position_id_with_mask(s1_attn_mask_pipe)[0] 
                
                s1_multi_modal_inputs_final = dict(s1_model_inputs_pipeline)
                raw_prompt_id = tokenizer.encode(s1_raw_prompt, add_special_tokens=False)

                s1_gen_inputs_list.append({
                    "input_ids": s1_input_ids_pipe[0], "attention_mask": s1_attn_mask_pipe[0],
                    "position_ids": s1_pos_ids_pipe_val, "multi_modal_data": s1_multi_modal_data,
                    "multi_modal_inputs": s1_multi_modal_inputs_final, "raw_prompt_ids": raw_prompt_id,
                })
                original_indices_map_s1.append(item_batch_idx)
                
            except Exception as e_s1_prep_pipe:
                print(f"{step_info_prefix_eval_loop} Error S1 prep for video_id {item_data_transformed.get('video_id', 'Unknown')}: {e_s1_prep_pipe}")
        
        # --- Run S1 and S2 Pipeline for the batch ---
        if s1_gen_inputs_list: 
            # S1 Generation
            s1_collated_for_gen = manual_collate_fn(s1_gen_inputs_list)
            s1_dataproto_for_gen = DataProto.from_single_dict(s1_collated_for_gen)
            s1_dataproto_for_gen.meta_info = {
                'eos_token_id': tokenizer.eos_token_id, 'pad_token_id': tokenizer.pad_token_id,
                'max_new_tokens': config.actor_rollout_ref.rollout.get("max_new_tokens_grounding_eval", 2048),
                'do_sample': config.actor_rollout_ref.rollout.get("do_sample_grounding_eval", False),
                'temperature': config.actor_rollout_ref.rollout.get("temperature_grounding_eval", 1.0),
            }
            s1_input_padded_pipe, s1_pad_pipe = pad_dataproto_to_divisor(s1_dataproto_for_gen, trainer_instance.actor_rollout_wg.world_size)
            s1_output_padded_pipe = trainer_instance.actor_rollout_wg.generate_sequences(s1_input_padded_pipe)
            s1_gen_output_proto_pipe = unpad_dataproto(s1_output_padded_pipe, pad_size=s1_pad_pipe)

            # S1 Post-processing
            if s1_gen_output_proto_pipe.batch.get('responses') is not None:
                s1_responses_valid_items = s1_gen_output_proto_pipe.batch.get('responses')
                decoded_s1_texts_valid_items = tokenizer.batch_decode(s1_responses_valid_items, skip_special_tokens=True)
                print("test: ", len(decoded_s1_texts_valid_items), decoded_s1_texts_valid_items)
                
                predicted_s1_times_valid_items, raw_times = parse_grounding_times(decoded_s1_texts_valid_items, [sample_fps], [32])
                print(f"{step_info_prefix_eval_loop} S1 generation output for batch {i_pipeline_batch}: {decoded_s1_texts_valid_items}")
                
                for i, original_batch_item_idx in enumerate(original_indices_map_s1):
                    s1_decoded_texts_for_batch[original_batch_item_idx] = decoded_s1_texts_valid_items[i]
                    s1_predicted_times_for_batch[original_batch_item_idx] = predicted_s1_times_valid_items[i]
            else:
                print(f"{step_info_prefix_eval_loop} Warning: S1 generation output missing 'responses' for batch {i_pipeline_batch}")

            # --- Stage 2: VQA Generation (using S1 results) ---
            s2_gen_inputs_list = []
            original_indices_map_s2 = [] 

            for item_batch_idx, item_data_transformed in enumerate(current_pipeline_batch_dicts):
                try:
                    clip_start_s2_pipe, clip_end_s2_pipe = item_data_transformed["s1_gt_start_time"], item_data_transformed["s1_gt_end_time"]
                    print(f"{step_info_prefix_eval_loop} S1 GT times for video_id {item_data_transformed['video_id']}: {clip_start_s2_pipe}, {clip_end_s2_pipe}")
                    print(f"{step_info_prefix_eval_loop} GT for video_id {item_data_transformed['s2_gt_answer_letter']}")
                    
                    s1_pred_start_pipe, s1_pred_end_pipe = s1_predicted_times_for_batch[item_batch_idx]
                    
                    print(f"{step_info_prefix_eval_loop} S1 predicted times for video_id {item_data_transformed['video_id']}: {s1_pred_start_pipe}, {s1_pred_end_pipe}")
                    
                    if isinstance(s1_pred_start_pipe, (float,int)) and isinstance(s1_pred_end_pipe, (float,int)) and s1_pred_start_pipe <= s1_pred_end_pipe:
                        clip_start_s2_pipe, clip_end_s2_pipe = s1_pred_start_pipe, s1_pred_end_pipe
                    else: 
                        clip_start_s2_pipe, clip_end_s2_pipe = 0.0, None 
                        print(f"{step_info_prefix_eval_loop} Warning: S2 clipping for video_id {item_data_transformed['video_id']} defaulting to full video.")

                    ele_s2_video_segment_cfg = {"video": item_data_transformed["video_path"], "video_start": clip_start_s2_pipe, "video_end": clip_end_s2_pipe, **video_proc_cfg_for_pipeline} 
                    s2_clipped_video_frames = fetch_video(ele_s2_video_segment_cfg, image_factor=video_proc_cfg_for_pipeline.get("image_factor", 28))
                    print(f"{step_info_prefix_eval_loop} S2 video segment for video_id {item_data_transformed['video_id']} from {clip_start_s2_pipe} to {clip_end_s2_pipe} frames: {s2_clipped_video_frames.shape if s2_clipped_video_frames is not None else 'None'}")
                    
                         
                    if s2_clipped_video_frames is None or s2_clipped_video_frames.nelement() == 0: 
                        print(f"{step_info_prefix_eval_loop} Warning: S2 clipped video failed for video_id {item_data_transformed['video_id']}. Skipping this item for S2 gen.")
                        continue
                    
                    item_data_for_s2_proc_pipe = {"question_text": item_data_transformed["s2_user_prompt_text"], "clipped_video": s2_clipped_video_frames, "original_index": item_batch_idx}
                    processed_s2_item = prepare_stage2_inputs_for_item(item_data_for_s2_proc_pipe, processor, tokenizer, config) 
                    if processed_s2_item:
                        s2_gen_inputs_list.append(processed_s2_item)
                        original_indices_map_s2.append(item_batch_idx) 
                except Exception as e_s2_prep_pipe:
                    print(f"{step_info_prefix_eval_loop} Error S2 prep for video_id {item_data_transformed.get('video_id', 'Unknown')}: {e_s2_prep_pipe}")

            # S2 Generation
            if s2_gen_inputs_list:
                s2_collated_for_gen = manual_collate_fn(s2_gen_inputs_list)
                s2_dataproto_for_gen = DataProto.from_single_dict(s2_collated_for_gen)
                s2_dataproto_for_gen.meta_info = {
                    'eos_token_id': tokenizer.eos_token_id, 'pad_token_id': tokenizer.pad_token_id,
                    'max_new_tokens': config.actor_rollout_ref.rollout.get("max_new_tokens_qa_eval", 50),
                    'do_sample': config.actor_rollout_ref.rollout.get("do_sample_qa_eval", False),
                    'temperature': config.actor_rollout_ref.rollout.get("temperature_qa_eval", 1.0),
                }
                s2_input_padded_pipe, s2_pad_pipe = pad_dataproto_to_divisor(s2_dataproto_for_gen, trainer_instance.actor_rollout_wg.world_size)
                s2_output_padded_pipe = trainer_instance.actor_rollout_wg.generate_sequences(s2_input_padded_pipe)
                s2_gen_output_proto_pipe = unpad_dataproto(s2_output_padded_pipe, pad_size=s2_pad_pipe)

                # S2 Post-processing
                if s2_gen_output_proto_pipe.batch.get('responses') is not None:
                    s2_responses_tensor_for_reward = s2_gen_output_proto_pipe.batch.get('responses')
                    decoded_s2_texts_valid_items = tokenizer.batch_decode(s2_responses_tensor_for_reward, skip_special_tokens=True)
                    print(f"{step_info_prefix_eval_loop} S2 generation output for batch {i_pipeline_batch}: {decoded_s2_texts_valid_items}")
                    for i, original_batch_item_idx in enumerate(original_indices_map_s2):
                        s2_decoded_texts_for_batch[original_batch_item_idx] = decoded_s2_texts_valid_items[i]
                else:
                    print(f"{step_info_prefix_eval_loop} Warning: S2 generation output missing 'responses' for batch {i_pipeline_batch}")
        else:
            print(f"{step_info_prefix_eval_loop} Warning: No valid S1 inputs to generate for batch {i_pipeline_batch}, skipping S2 as well.")

        
        # --- Scoring and Aggregation for the batch ---
        metrics_from_reward_fn_this_batch = {}
        if s2_responses_tensor_for_reward is not None: 
            reward_non_tensor_data = defaultdict(list)
            items_to_score_indices_in_s2_tensor = [] 
            
            # Note: s2_responses_tensor_for_reward has length len(original_indices_map_s2)
            for i_s2_tensor, original_batch_item_idx_of_s2_item in enumerate(original_indices_map_s2):
                item_data_for_reward_assembly = current_pipeline_batch_dicts[original_batch_item_idx_of_s2_item]
                s1_text_for_reward = s1_decoded_texts_for_batch[original_batch_item_idx_of_s2_item]

                if s1_text_for_reward != "<S1 Gen Fail>":
                    reward_non_tensor_data['decoded_grounding_texts'].append(s1_text_for_reward)
                    reward_non_tensor_data['start_time'].append(item_data_for_reward_assembly['s1_gt_start_time'])
                    reward_non_tensor_data['end_time'].append(item_data_for_reward_assembly['s1_gt_end_time'])
                    reward_non_tensor_data['ground_truth_answer'].append(item_data_for_reward_assembly['s2_gt_answer_letter'])
                    reward_non_tensor_data['video_length'].append(32 / sample_fps)
                    reward_non_tensor_data['total_frames'].append(32)
                    items_to_score_indices_in_s2_tensor.append(i_s2_tensor)
            
            if reward_non_tensor_data['decoded_grounding_texts']: 
                reward_non_tensor_data_np = {k: np.array(v, dtype=object) for k, v in reward_non_tensor_data.items()}
                
                filtered_s2_responses = s2_responses_tensor_for_reward[items_to_score_indices_in_s2_tensor]
                filtered_s2_masks = (filtered_s2_responses != tokenizer.pad_token_id).long()

                eval_input_proto_for_reward = DataProto(
                    batch=TensorDict({
                        "responses": filtered_s2_responses, "response_mask": filtered_s2_masks
                    }, batch_size=[len(items_to_score_indices_in_s2_tensor)]),
                    non_tensor_batch=reward_non_tensor_data_np
                )
                
                result_eval_fn = vqa_eval_reward_fn(eval_input_proto_for_reward, return_dict=True)
                metrics_from_reward_fn_this_batch = result_eval_fn.get("reward_extra_info", {})
                for metric_key, metric_values_list in metrics_from_reward_fn_this_batch.items():
                    all_eval_run_metrics_accumulator[metric_key].extend(list(metric_values_list))
        
        # --- Aggregation of detailed results for the batch ---
        default_scores_keys = ['s1_grounding_iou', 's1_grounding_format_score', 's1_base_format_correct', 's1_thinking_present',
                               's1_grounding_combined_score_unweighted', 's2_qa_accuracy_correct', 's2_qa_format_score',
                               's2_base_format_correct', 's2_thinking_present', 'final_weighted_score']
        default_scores = {k: 0.0 for k in default_scores_keys}
        reward_metrics_ptr = 0 

        for item_batch_idx, item_data_transformed in enumerate(current_pipeline_batch_dicts):
            item_scores_to_save = default_scores.copy()

            if item_batch_idx in original_indices_map_s2 and \
               s1_decoded_texts_for_batch[item_batch_idx] != "<S1 Gen Fail>" and \
               s2_decoded_texts_for_batch[item_batch_idx] != "<S2 Gen Fail>" and \
               metrics_from_reward_fn_this_batch:

                all_keys_have_data_at_ptr = True
                for key_metric_save_check in metrics_from_reward_fn_this_batch:
                    if reward_metrics_ptr >= len(metrics_from_reward_fn_this_batch[key_metric_save_check]):
                        all_keys_have_data_at_ptr = False; break
                
                if all_keys_have_data_at_ptr:
                    for key_metric_save_assign in metrics_from_reward_fn_this_batch:
                        if key_metric_save_assign in item_scores_to_save: 
                           item_scores_to_save[key_metric_save_assign] = metrics_from_reward_fn_this_batch[key_metric_save_assign][reward_metrics_ptr]
                    reward_metrics_ptr += 1
            
            detailed_results_for_saving.append({
                "video_id": item_data_transformed["video_id"],
                "video_uid": item_data_transformed["video_id"],
                "s1_gt_start": item_data_transformed["s1_gt_start_time"],
                "s1_gt_end": item_data_transformed["s1_gt_end_time"],
                "s1_pred_start": s1_predicted_times_for_batch[item_batch_idx][0] if s1_predicted_times_for_batch[item_batch_idx] else None,
                "s1_pred_end": s1_predicted_times_for_batch[item_batch_idx][1] if s1_predicted_times_for_batch[item_batch_idx] else None,
                "s1_input_prompt_for_gen": item_data_transformed["s1_user_prompt_text"],
                "s1_generated_text": s1_decoded_texts_for_batch[item_batch_idx],
                "s2_gt_answer_letter": item_data_transformed["s2_gt_answer_letter"],
                "s2_question_with_choices": item_data_transformed["s2_user_prompt_text"],
                "s2_generated_text": s2_decoded_texts_for_batch[item_batch_idx],
                **item_scores_to_save
            })

    # --- Final Output and Summary ---
    detailed_results_file_path = os.path.join(output_dir_final, f"cg_bench_detailed_results_step_{0}.jsonl")
    print(f"{step_info_prefix_eval_loop} Saving detailed CG-Bench results ({len(detailed_results_for_saving)} items) to: {detailed_results_file_path}")
    with open(detailed_results_file_path, 'w', encoding='utf-8') as f_out:
        for res_item in detailed_results_for_saving:
            f_out.write(json.dumps(convert_numpy_to_native(res_item)) + '\n')

    final_eval_metrics_summary = {}
    for metric_name in default_scores_keys: 
        values = all_eval_run_metrics_accumulator.get(metric_name, [])
        numeric_values = [v for v in values if isinstance(v, (int, float, np.number)) and np.isfinite(v)]
        if numeric_values:
            log_prefix = f"eval_cg_bench/{metric_name}"
            final_eval_metrics_summary[f"{log_prefix}/mean"] = float(np.mean(numeric_values))
            final_eval_metrics_summary[f"{log_prefix}/std"] = float(np.std(numeric_values))
            final_eval_metrics_summary[f"{log_prefix}/count"] = len(numeric_values)
            if metric_name == 's2_qa_accuracy_correct': 
                 final_eval_metrics_summary["eval_cg_bench/overall_s2_accuracy"] = float(np.mean(numeric_values))
            if metric_name == 's1_grounding_iou': 
                 final_eval_metrics_summary["eval_cg_bench/overall_s1_iou"] = float(np.mean(numeric_values))
        else:
            print(f"{step_info_prefix_eval_loop} Warning: No valid numeric values for metric '{metric_name}' in CG-Bench summary.")
    
    print(f"\n--- CG-Bench Evaluation Metrics (Step {0}) ---")
    pprint(final_eval_metrics_summary)
    
    print(f"\n--- Detailed results saved in: {output_dir_final} ---")
    print("--- CG-Bench Standalone Evaluation Complete ---")
    ray.shutdown()


if __name__ == '__main__':
    if VQAMultiStageRewardManager is None and "verl.workers.reward_manager" in sys.modules : # Check if import was attempted
        print("CRITICAL ERROR: VQAMultiStageRewardManager failed to import. Cannot proceed.")
        exit(1) 
    run_cg_bench_evaluation_standalone()