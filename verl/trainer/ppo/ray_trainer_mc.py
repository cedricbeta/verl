# Copyright 2024 Bytedance Ltd. and/or its affiliates
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
"""
FSDP PPO Trainer with Ray-based single controller.
This trainer supports model-agonistic model initialization with huggingface
"""

import os
import uuid
import warnings
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from pprint import pprint
from typing import Type, Dict
from copy import deepcopy
from collections import defaultdict
from functools import partial
from tqdm import tqdm

import ray
import numpy as np
from codetiming import Timer
from omegaconf import OmegaConf, open_dict
from verl import DataProto
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.single_controller.base import Worker
from verl.single_controller.ray import RayResourcePool, RayWorkerGroup, RayClassWithInitArgs
from verl.single_controller.ray.base import create_colocated_worker_cls
from verl.trainer.ppo import core_algos
from verl.trainer.ppo.core_algos import agg_loss
from verl.utils.py_functional import append_to_dict
from verl.trainer.ppo.metric_utils import compute_data_metrics, compute_throughout_metrics, compute_timing_metrics, reduce_metrics, bootstrap_metric, calc_maj_val, process_validation_metrics
from verl.utils.seqlen_balancing import get_seqlen_balanced_partitions, log_seqlen_unbalance
from verl.utils.checkpoint.checkpoint_manager import find_latest_ckpt_path
from verl.utils.dataset.video_rl_dataset_mc import RLHFDataset, collate_fn
# from verl.utils.dataset.video_rl_dataset import VideoRLHFDataset, collate_fn
from verl.utils.tracking import ValidationGenerationsLogger
from torch.utils.data import Dataset, RandomSampler, SequentialSampler
from torchdata.stateful_dataloader import StatefulDataLoader
import json
from collections import defaultdict
from typing import List, Dict, Any, Optional, Union
import verl.utils.torch_functional as verl_F
from verl.utils.model import compute_position_id_with_mask
from verl.utils.dataset.vqa_dataset import fetch_video
from pprint import pprint, pformat
from tensordict import TensorDict

WorkerType = Type[Worker]


DEBUG_PRINT_FIT_VQA = True

class Role(Enum):
    """
    To create more roles dynamically, you can subclass Role and add new members
    """
    Actor = 0
    Rollout = 1
    ActorRollout = 2
    Critic = 3
    RefPolicy = 4
    RewardModel = 5
    ActorRolloutRef = 6


class AdvantageEstimator(str, Enum):
    """
    Using an enumeration class to avoid spelling errors in adv_estimator
    """
    GAE = 'gae'
    GRPO = 'grpo'
    REINFORCE_PLUS_PLUS = 'reinforce_plus_plus'
    REINFORCE_PLUS_PLUS_BASELINE = 'reinforce_plus_plus_baseline'
    REMAX = 'remax'
    RLOO = 'rloo'

def process_validation_metrics(
        data_sources: np.ndarray, # Or list
        sample_inputs: list[str], # Unused in this simplified version, but kept for signature
        infos_dict: dict[str, list[Any]],
        target_metrics: List[str] = ["reward", "tvg_accuracy", "tvg_format", "tvg_combined"] # Metrics to process
    ) -> dict[str, float]:
        """
        Process validation metrics to calculate overall mean, max, and min for target variables.

        Args:
            data_sources: Array/List of data source identifiers for each sample.
            sample_inputs: List of input prompts (unused here).
            infos_dict: Dictionary where keys are variable names (e.g., 'reward', 'tvg_accuracy')
                        and values are lists of corresponding values for ALL samples.
            target_metrics: List of keys from infos_dict to calculate stats for.

        Returns:
            dict[str, float]: A flat dictionary with keys like 'val/reward/mean',
                            'val/tvg_accuracy/max', etc.
        """
        output_metrics = {}
        unique_sources = set(data_sources) if data_sources is not None and len(data_sources) > 0 else {"unknown"}

        # Calculate overall metrics first
        for var_name in target_metrics:
            if var_name in infos_dict:
                values = infos_dict[var_name]
                # Filter out non-numeric types and non-finite values if necessary
                numeric_values = [v for v in values if isinstance(v, (int, float, np.number)) and np.isfinite(v)]

                if numeric_values:
                    prefix = f"val/all_sources/{var_name}" # Use a common prefix
                    output_metrics[f"{prefix}/mean"] = float(np.mean(numeric_values))
                    output_metrics[f"{prefix}/max"] = float(np.max(numeric_values))
                    output_metrics[f"{prefix}/min"] = float(np.min(numeric_values))
                    output_metrics[f"{prefix}/std"] = float(np.std(numeric_values)) # Keep std deviation
                    output_metrics[f"{prefix}/count"] = len(numeric_values)
                else:
                    print(f"Warning: No valid numeric values found for metric '{var_name}' during validation processing.")

        # Optional: Calculate per-data-source metrics (if needed)
        # This part can be added if you still want per-source breakdown
        # for data_source in unique_sources:
        #     indices = [i for i, src in enumerate(data_sources) if src == data_source]
        #     if not indices: continue
        #     for var_name in target_metrics:
        #          if var_name in infos_dict:
        #              source_values = [infos_dict[var_name][i] for i in indices]
        #              numeric_values = [v for v in source_values if isinstance(v, (int, float, np.number)) and np.isfinite(v)]
        #              if numeric_values:
        #                  prefix = f"val/{data_source}/{var_name}"
        #                  output_metrics[f"{prefix}/mean"] = float(np.mean(numeric_values))
        #                  # Add max/min/std per source if desired
        #                  output_metrics[f"{prefix}/count"] = len(numeric_values)


        return output_metrics

def convert_numpy_to_native(obj):
    """Recursively converts numpy arrays and numbers in dicts/lists to native types."""
    if isinstance(obj, dict):
        return {k: convert_numpy_to_native(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_to_native(elem) for elem in obj]
    elif isinstance(obj, np.ndarray):
        return obj.tolist() # Convert numpy array to Python list
    elif isinstance(obj, (np.int_, np.intc, np.intp, np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16, np.uint32, np.uint64)):
        return int(obj) # Convert numpy int types to Python int
    elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
        # Check for NaN/Infinity which are not standard JSON
        if np.isnan(obj): return None # Represent NaN as null
        if np.isinf(obj): return None # Represent Inf as null (or choose a string like 'Infinity')
        return float(obj) # Convert numpy float types to Python float
    elif isinstance(obj, (np.bool_)):
        return bool(obj) # Convert numpy bool types to Python bool
    elif isinstance(obj, (np.void)): # Handle potential void types if present
        return None # Represent void as null
    return obj # Return object itself if not a numpy type needing conversion
@dataclass
class ResourcePoolManager:
    """
    Define a resource pool specification. Resource pool will be initialized first.
    Mapping
    """
    resource_pool_spec: dict[str, list[int]]
    mapping: dict[Role, str]
    resource_pool_dict: dict[str, RayResourcePool] = field(default_factory=dict)

    def create_resource_pool(self):
        for resource_pool_name, process_on_nodes in self.resource_pool_spec.items():
            # max_colocate_count means the number of WorkerGroups (i.e. processes) in each RayResourcePool
            # For FSDP backend, we recommend using max_colocate_count=1 that merge all WorkerGroups into one.
            # For Megatron backend, we recommend using max_colocate_count>1 that can utilize different WorkerGroup for differnt models
            resource_pool = RayResourcePool(process_on_nodes=process_on_nodes,
                                            use_gpu=True,
                                            max_colocate_count=1,
                                            name_prefix=resource_pool_name)
            self.resource_pool_dict[resource_pool_name] = resource_pool

        self._check_resource_available()

    def get_resource_pool(self, role: Role) -> RayResourcePool:
        """Get the resource pool of the worker_cls"""
        return self.resource_pool_dict[self.mapping[role]]

    def get_n_gpus(self) -> int:
        """Get the number of gpus in this cluster."""
        return sum([n_gpus for process_on_nodes in self.resource_pool_spec.values() for n_gpus in process_on_nodes])

    def _check_resource_available(self):
        """Check if the resource pool can be satisfied in this ray cluster."""
        node_available_resources = ray.state.available_resources_per_node()
        node_available_gpus = {node: node_info.get('GPU', 0) for node, node_info in node_available_resources.items()}

        # check total required gpus can be satisfied
        total_available_gpus = sum(node_available_gpus.values())
        total_required_gpus = sum(
            [n_gpus for process_on_nodes in self.resource_pool_spec.values() for n_gpus in process_on_nodes])
        if total_available_gpus < total_required_gpus:
            raise ValueError(
                f"Total available GPUs {total_available_gpus} is less than total desired GPUs {total_required_gpus}")

        # check each resource pool can be satisfied, O(#resource_pools * #nodes)
        for resource_pool_name, process_on_nodes in self.resource_pool_spec.items():
            num_gpus, num_nodes = process_on_nodes[0], len(process_on_nodes)
            for node, available_gpus in node_available_gpus.items():
                if available_gpus >= num_gpus:
                    node_available_gpus[node] -= num_gpus
                    num_nodes -= 1
                    if num_nodes == 0:
                        break
            if num_nodes > 0:
                raise ValueError(
                    f"Resource pool {resource_pool_name}: {num_gpus}*{num_nodes} cannot be satisfied in this ray cluster"
                )


import torch
from verl.utils.torch_functional import masked_mean


def apply_kl_penalty(data: DataProto, kl_ctrl: core_algos.AdaptiveKLController, kl_penalty='kl'):
    responses = data.batch['responses']
    response_length = responses.size(1)
    token_level_scores = data.batch['token_level_scores']
    batch_size = data.batch.batch_size[0]
    attention_mask = data.batch['attention_mask']
    response_mask = attention_mask[:, -response_length:]

    # compute kl between ref_policy and current policy
    # When apply_kl_penalty, algorithm.use_kl_in_reward=True, so the reference model has been enabled.
    kld = core_algos.kl_penalty(data.batch['old_log_probs'], data.batch['ref_log_prob'],
                                kl_penalty=kl_penalty)  # (batch_size, response_length)
    kld = kld * response_mask
    beta = kl_ctrl.value

    token_level_rewards = token_level_scores - beta * kld

    current_kl = masked_mean(kld, mask=response_mask, axis=-1)  # average over sequence
    current_kl = torch.mean(current_kl, dim=0).item()

    # according to https://github.com/huggingface/trl/blob/951ca1841f29114b969b57b26c7d3e80a39f75a0/trl/trainer/ppo_trainer.py#L837
    kl_ctrl.update(current_kl=current_kl, n_steps=batch_size)
    data.batch['token_level_rewards'] = token_level_rewards

    metrics = {'actor/reward_kl_penalty': current_kl, 'actor/reward_kl_penalty_coeff': beta}

    return data, metrics


def compute_response_mask(data: DataProto):
    responses = data.batch['responses']
    response_length = responses.size(1)
    attention_mask = data.batch['attention_mask']
    return attention_mask[:, -response_length:]


def compute_advantage(data: DataProto, adv_estimator, gamma=1.0, lam=1.0, num_repeat=1):
    # Back-compatible with trainers that do not compute response mask in fit
    if "response_mask" not in data.batch.keys():
        data.batch['response_mask'] = compute_response_mask(data)
    # prepare response group
    # TODO: add other ways to estimate advantages
    if adv_estimator == AdvantageEstimator.GAE:
        values = data.batch['values']
        advantages, returns = core_algos.compute_gae_advantage_return(
            token_level_rewards=data.batch['token_level_rewards'],
            values=data.batch['values'],
            response_mask=data.batch['response_mask'],
            gamma=gamma,
            lam=lam)
        data.batch['advantages'] = advantages
        data.batch['returns'] = returns
    elif adv_estimator == AdvantageEstimator.GRPO:
        advantages, returns = core_algos.compute_grpo_outcome_advantage(
            token_level_rewards=data.batch['token_level_rewards'],
            response_mask=data.batch['response_mask'],
            index=data.non_tensor_batch['uid'])
        data.batch['advantages'] = advantages
        data.batch['returns'] = returns
    elif adv_estimator == AdvantageEstimator.REINFORCE_PLUS_PLUS_BASELINE:
        advantages, returns = core_algos.compute_reinforce_plus_plus_baseline_outcome_advantage(
            token_level_rewards=data.batch['token_level_rewards'],
            response_mask=data.batch['response_mask'],
            index=data.non_tensor_batch['uid'])
        data.batch['advantages'] = advantages
        data.batch['returns'] = returns
    elif adv_estimator == AdvantageEstimator.REINFORCE_PLUS_PLUS:
        advantages, returns = core_algos.compute_reinforce_plus_plus_outcome_advantage(
            token_level_rewards=data.batch['token_level_rewards'],
            response_mask=data.batch['response_mask'],
            gamma=gamma)
        data.batch['advantages'] = advantages
        data.batch['returns'] = returns
    elif adv_estimator == AdvantageEstimator.REMAX:
        advantages, returns = core_algos.compute_remax_outcome_advantage(
            token_level_rewards=data.batch['token_level_rewards'],
            reward_baselines=data.batch['reward_baselines'],
            response_mask=data.batch['response_mask'])

        data.batch['advantages'] = advantages
        data.batch['returns'] = returns
    elif adv_estimator == AdvantageEstimator.RLOO:
        advantages, returns = core_algos.compute_rloo_outcome_advantage(
            token_level_rewards=data.batch['token_level_rewards'],
            response_mask=data.batch['response_mask'],
            index=data.non_tensor_batch['uid'])
        data.batch['advantages'] = advantages
        data.batch['returns'] = returns
    else:
        raise NotImplementedError
    return data


def parse_grounding_times(decoded_texts: list[str]) -> list[tuple[Optional[float], Optional[float]]]:
    """Parses start/end times from Stage 1 generated text. Robust implementation needed."""
    print(f"PARSING {len(decoded_texts)} grounding responses...") # Add print
    parsed_times = []
    for i, text in enumerate(decoded_texts):
        import re # Keep import local if only used here
        # print(f"Parsing text: {text}") # Add print
        try: # Simple example parsing, needs improvement
            content_answer_match = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL | re.IGNORECASE)
            content_answer = content_answer_match.group(1).strip()
            if not (content_answer.startswith('{') and content_answer.endswith('}')): return 0.0

            answer_data = json.loads(content_answer)
            if not isinstance(answer_data, dict): return 0.0
            if "start_time" not in answer_data or "end_time" not in answer_data: return 0.0

            start_time_str = answer_data["start_time"]
            end_time_str = answer_data["end_time"]
            start_time = float(start_time_str) if start_time_str else None
            end_time = float(end_time_str) if end_time_str else None
            
            parsed_times.append((start_time, end_time))
            
        except Exception as e:
            # print(f"Error parsing times from text (Item {i}): '{text}' - {e}")
            parsed_times.append((None, None))
            
    print(f"PARSING finished. Got {len([t for t in parsed_times if t[0] is not None])} valid time pairs.") # Add print
    return parsed_times

def resample_and_process_video_segment(video_path: str, start_time: float, end_time: float, config: dict, processor) -> Optional[dict]:
    """
    **PLACEHOLDER:** This function needs real implementation, likely involving
    calls to Ray workers to perform distributed video loading, segment
    extraction (using _read_video_torchvision), frame sampling (smart_nframes),
    resizing (process_video_frames), and processor feature extraction for vision.
    """
    print(f"PLACEHOLDER: Resample {os.path.basename(video_path)} [{start_time:.2f}-{end_time if end_time is not None else 'end'}]")
    # Simulate success/failure and dummy output structure
    if np.random.rand() < 0.1: # Simulate 10% failure rate
        print("PLACEHOLDER: Resampling failed.")
        return None
    dummy_visual_features = {
        'pixel_values_videos': torch.randn(1, 16, 3, 224, 224), # Example shape TCHW
        # Add other keys the processor/model might expect e.g.
        # 'video_grid_thw': torch.tensor([[16, 14, 14]])
    }
    print("PLACEHOLDER: Resampling successful.")
    return dummy_visual_features

# def prepare_stage2_inputs_for_item(item_data, processor, tokenizer, config):
#     # ... (implementation from previous response) ...
#     """Prepares tokenized text and combines with video features for one item."""
#     question_text = item_data["question_text"]
#     clipped_video_frames_tensor = item_data["clipped_video"] # Assumes video is already processed

#     qa_system_prompt = "Answer the following multiple-choice question based on the video clip by providing only the letter of the correct option." # Example
#     stage2_messages = []
#     if qa_system_prompt:
#          stage2_messages.append({"role": "system", "content": qa_system_prompt})
#     stage2_messages.append({"role": "user", "content": '<video> ' + question_text})

#     # Ensure processor is available (passed from trainer or accessed via self)
#     stage2_raw_prompt = processor.apply_chat_template(stage2_messages, add_generation_prompt=True, tokenize=False)

#     stage2_model_inputs = processor(
#         text=[stage2_raw_prompt],
#         images=None,
#         videos=[clipped_video_frames_tensor],
#         return_tensors="pt"
#     )
#     #  # --- MODIFIED DEBUG PRINT ---
#     # item_original_idx = item_data.get('original_index', -1)
#     # print(f"DEBUG P2S: Item Idx {item_original_idx} - Processor Output Keys: {list(stage2_model_inputs.keys())}")
#     # if 'grid_thw' in stage2_model_inputs:
#     #     grid_thw_val = stage2_model_inputs['grid_thw']
#     #     print(f"DEBUG P2S: Item Idx {item_original_idx} - grid_thw = {grid_thw_val} (type: {type(grid_thw_val)})")
#     #     if isinstance(grid_thw_val, (list, torch.Tensor, np.ndarray)):
#     #         try:
#     #             print(f"DEBUG P2S: Item Idx {item_original_idx} - grid_thw length: {len(grid_thw_val)}")
#     #             if len(grid_thw_val) > 0:
#     #                 print(f"DEBUG P2S: Item Idx {item_original_idx} - grid_thw[0] = {grid_thw_val[0]} (type: {type(grid_thw_val[0])})")
#     #                 if isinstance(grid_thw_val[0], (list, torch.Tensor, np.ndarray)):
#     #                      print(f"DEBUG P2S: Item Idx {item_original_idx} - grid_thw[0] length: {len(grid_thw_val[0])}")

#     #         except:
#     #             pass # Error during detailed print
#     # else:
#     #     print(f"DEBUG P2S: Item Idx {item_original_idx} - 'grid_thw' NOT in stage2_model_inputs")
#     # if 'pixel_values_videos' in stage2_model_inputs:
#     #     pvv = stage2_model_inputs['pixel_values_videos']
#     #     print(f"DEBUG P2S: Item Idx {item_original_idx} - pixel_values_videos type: {type(pvv)}")
#     #     if isinstance(pvv, torch.Tensor):
#     #         print(f"DEBUG P2S: Item Idx {item_original_idx} - pixel_values_videos shape: {pvv.shape}")
#     #     elif isinstance(pvv, list) and pvv and isinstance(pvv[0], torch.Tensor):
#     #          print(f"DEBUG P2S: Item Idx {item_original_idx} - pixel_values_videos is list of tensors, first shape: {pvv[0].shape}")

#     s2_input_ids_raw = stage2_model_inputs.pop("input_ids")
#     s2_attn_mask_raw = stage2_model_inputs.pop("attention_mask")

#     # Ensure tokenizer and config are available
#     stage2_input_ids, stage2_attention_mask = verl_F.postprocess_data(
#          input_ids=s2_input_ids_raw, attention_mask=s2_attn_mask_raw,
#          max_length=config.data.get("max_prompt_length", 1024),
#          pad_token_id=tokenizer.pad_token_id, left_pad=True,
#          truncation=config.data.get("truncation", "error")
#      )

#     # Calculate Stage 2 Position IDs
#     stage2_position_ids = compute_position_id_with_mask(stage2_attention_mask)[0]

#     stage2_multi_modal_inputs = dict(stage2_model_inputs) # Other features like pixel_values_videos

#     return {
#         'input_ids': stage2_input_ids[0],
#         'attention_mask': stage2_attention_mask[0],
#         'position_ids': stage2_position_ids,
#         'multi_modal_inputs': stage2_multi_modal_inputs,
#     }

# def prepare_stage2_inputs_for_item(item_data, processor, tokenizer, config):
#     """Prepares tokenized text and combines with video features for one item."""
#     question_text = item_data["question_text"]
#     clipped_video_frames_tensor = item_data["clipped_video"] # Assumes video is already processed

#     qa_system_prompt = "Answer the following multiple-choice question based on the video clip by providing only the letter of the correct option." # Example
#     stage2_messages = []
#     if qa_system_prompt:
#          stage2_messages.append({"role": "system", "content": qa_system_prompt})
#     stage2_messages.append({"role": "user", "content": question_text})

#     # Processor call automatically adds <video> token if videos are provided
#     stage2_raw_prompt = processor.apply_chat_template(stage2_messages, add_generation_prompt=True, tokenize=False)

#     # Call processor ONCE to get all outputs needed for RoPE calculation
#     stage2_model_inputs_all = processor(
#         text=[stage2_raw_prompt],
#         images=None,
#         videos=[clipped_video_frames_tensor],
#         return_tensors="pt"
#     )

#     if "input_ids" not in stage2_model_inputs_all or "attention_mask" not in stage2_model_inputs_all:
#          raise ValueError("Processor output missing keys for Stage 2")

#     s2_input_ids_raw = stage2_model_inputs_all.pop("input_ids") # Keep for postprocessing
#     s2_attn_mask_raw = stage2_model_inputs_all.pop("attention_mask") # Keep for postprocessing

#     # --- MODIFICATION: Use RIGHT padding and RIGHT truncation ---
#     stage2_input_ids, stage2_attention_mask = verl_F.postprocess_data(
#          input_ids=s2_input_ids_raw, attention_mask=s2_attn_mask_raw,
#          max_length=config.data.get("max_prompt_length", 4096),
#          pad_token_id=tokenizer.pad_token_id,
#          left_pad=True, # Use Right Padding
#          truncation='error' # Use Right Truncation
#      )
#     # --- END MODIFICATION ---

#     # --- MODIFICATION: Calculate Position IDs using get_rope_index ---
#     stage2_position_ids = None
#     stage2_model_inputs_for_rope = dict(stage2_model_inputs_all) # Create a copy to pass non-tensor features

#     try:
       
#         # from verl.models.transformers.qwen2_vl import get_rope_index
#         # s2_pos_ids_list =  [
#         #                     get_rope_index(
#         #                         processor,
#         #                         input_ids=stage2_input_ids[0],
#         #                         image_grid_thw=stage2_model_inputs_for_rope.get("image_grid_thw"),
#         #                         video_grid_thw=stage2_model_inputs_for_rope.get("video_grid_thw"),
#         #                         second_per_grid_ts=stage2_model_inputs_for_rope.get("second_per_grid_ts"),
#         #                         attention_mask=stage2_attention_mask[0],
#         #                     )
#         #                 ]  # (1, 3, seq_len)
#         # stage2_position_ids = s2_pos_ids_list[0] # get_rope_index might return list or tensor directly
#         # print(f"DEBUG P2S: Calculated RoPE Position IDs shape: {stage2_position_ids.shape}")
#         stage2_position_ids = compute_position_id_with_mask(stage2_attention_mask)[0]

#     except ImportError:
#         print("WARN P2S: verl.models.transformers.qwen2_vl.get_rope_index not found. Falling back to basic position IDs.")
#         stage2_position_ids = compute_position_id_with_mask(stage2_attention_mask)[0]
#     except Exception as rope_err:
#         print(f"ERROR P2S: Error calculating RoPE Position IDs: {rope_err}. Falling back to basic position IDs.")
#         import traceback; traceback.print_exc()
#         stage2_position_ids = compute_position_id_with_mask(stage2_attention_mask)[0] # Basic fallback

#     if stage2_position_ids is None: # Ensure fallback if all else fails
#         print("ERROR P2S: Position ID calculation failed completely. Using basic IDs.")
#         stage2_position_ids = compute_position_id_with_mask(stage2_attention_mask)[0]

#     # --- END MODIFICATION ---

#     # Prepare the final multi_modal_inputs dict to be returned (for collation)
#     # This should contain the features needed later by the actor (pixel_values, grid_thw)
#     # Remove keys that are not needed or handled elsewhere (like input_ids, attn_mask, pos_ids)
#     stage2_multi_modal_inputs_final = dict(stage2_model_inputs_all) # Start with all outputs from processor
#     stage2_multi_modal_inputs_final.pop("input_ids", None)
#     stage2_multi_modal_inputs_final.pop("attention_mask", None)
#     stage2_multi_modal_inputs_final.pop("position_ids", None) # We calculated this separately


#     return {
#         'input_ids': stage2_input_ids[0],       # Processed IDs (Right-padded)
#         'attention_mask': stage2_attention_mask[0], # Processed Mask (Right-padded)
#         'position_ids': stage2_position_ids,    # RoPE or Fallback Position IDs
#         'multi_modal_inputs': stage2_multi_modal_inputs_final, # Features for actor (e.g., pixel_values)
#     }

def prepare_stage2_inputs_for_item(item_data, processor, tokenizer, config):
    """Prepares tokenized text and combines with video features for one item."""
    question_text = item_data["question_text"]
    clipped_video_frames_tensor = item_data["clipped_video"] # Assumes this is a Tensor TCHW

    qa_system_prompt = '''You FIRST think about the reasoning process as an internal monologue and then provide the final answer.
                        The reasoning process MUST BE enclosed within <think> </think> tags.
                        The final answer MUST BE put in <answer> </answer> tags, containing only the letter of the correct option.'''
    stage2_messages = []
    if qa_system_prompt: # Assuming system_prompt is a string or None
         stage2_messages.append({"role": "system", "content": qa_system_prompt})
    stage2_messages.append({"role": "user", "content": question_text}) # Processor will add <video> if videos are passed

    # Call processor ONCE to get all outputs
    # `processor_outputs_all` will contain 'input_ids', 'attention_mask', 
    # 'pixel_values_videos', 'video_grid_thw', 'second_per_grid_ts', etc.
    processor_outputs_all = processor(
        text=[processor.apply_chat_template(stage2_messages, add_generation_prompt=True, tokenize=False)],
        images=None, # No separate images for Stage 2 in this flow
        videos=[clipped_video_frames_tensor],
        return_tensors="pt"
    )

    if "input_ids" not in processor_outputs_all or "attention_mask" not in processor_outputs_all:
         raise ValueError("Processor output missing 'input_ids' or 'attention_mask' for Stage 2")

    s2_input_ids_raw = processor_outputs_all.pop("input_ids") 
    s2_attn_mask_raw = processor_outputs_all.pop("attention_mask")

    # `processor_outputs_all` now contains remaining items like 'pixel_values_videos', 'video_grid_thw', 'second_per_grid_ts'

    # Pad/Truncate S2 prompt (text part)
    stage2_input_ids_padded, stage2_attention_mask_padded = verl_F.postprocess_data(
         input_ids=s2_input_ids_raw, 
         attention_mask=s2_attn_mask_raw,
         max_length=config.data.get("max_prompt_length", 4096), # Use a relevant max length for S2 prompt
         pad_token_id=tokenizer.pad_token_id,
         left_pad=True, 
         truncation='error' # Or 'right' as needed
     )

    # Calculate Position IDs using get_rope_index
    # `get_rope_index` needs access to `second_per_grid_ts` from the original processor output.
    stage2_position_ids = None
    # Create a temporary dict that mimics `model_inputs` for `get_rope_index`
    temp_model_inputs_for_rope = dict(processor_outputs_all) # Has 'video_grid_thw', 'second_per_grid_ts' etc.

    try:
        from verl.models.transformers.qwen2_vl import get_rope_index # Ensure correct import
        stage2_position_ids_list = [
            get_rope_index(
                processor,
                input_ids=stage2_input_ids_padded[0], # Use the padded input_ids
                image_grid_thw=temp_model_inputs_for_rope.get("image_grid_thw"),
                video_grid_thw=temp_model_inputs_for_rope.get("video_grid_thw"),
                second_per_grid_ts=temp_model_inputs_for_rope.get("second_per_grid_ts"), # Crucial: Use value from processor_outputs_all
                attention_mask=stage2_attention_mask_padded[0], # Use the padded attention_mask
            )
        ]
        stage2_position_ids = stage2_position_ids_list[0] 
        # print(f"[Prepare S2 DEBUG] Calculated RoPE Position IDs shape: {stage2_position_ids.shape if stage2_position_ids is not None else 'None'}")
    except ImportError:
        print("[Prepare S2 WARN] verl.models.transformers.qwen2_vl.get_rope_index not found. Falling back to basic position IDs.")
        stage2_position_ids = compute_position_id_with_mask(stage2_attention_mask_padded)[0]
    except Exception as rope_err:
        print(f"[Prepare S2 ERROR] Error calculating RoPE Position IDs: {rope_err}. Falling back to basic position IDs.")
        import traceback; traceback.print_exc()
        stage2_position_ids = compute_position_id_with_mask(stage2_attention_mask_padded)[0]

    if stage2_position_ids is None: 
        print("[Prepare S2 ERROR] Position ID calculation failed completely. Using basic IDs.")
        stage2_position_ids = compute_position_id_with_mask(stage2_attention_mask_padded)[0]

    # --- Prepare the final `multi_modal_inputs` dictionary for the PPO actor ---
    # This dictionary should ONLY contain TENSOR features that the actor model's `forward` method expects.
    # `processor_outputs_all` currently holds Tensors like 'pixel_values_videos', 'video_grid_thw'
    # AND potentially list-based items like 'second_per_grid_ts'.
    
    final_multi_modal_inputs_for_actor = {}
    if isinstance(processor_outputs_all, dict):
        for key, value in processor_outputs_all.items():
            if isinstance(value, torch.Tensor): # Only keep Tensors
                final_multi_modal_inputs_for_actor[key] = value
            # else:
                # print(f"[Prepare S2 DEBUG] Skipping non-tensor key '{key}' (type: {type(value)}) from final multi_modal_inputs for actor.")
    
    # print(f"[Prepare S2 DEBUG] Final multi_modal_inputs for actor keys: {list(final_multi_modal_inputs_for_actor.keys())}")

    return {
        'input_ids': stage2_input_ids_padded[0],       # Padded S2 prompt tokens
        'attention_mask': stage2_attention_mask_padded[0], # Padded S2 prompt mask
        'position_ids': stage2_position_ids,           # RoPE or Fallback Position IDs for S2 prompt
        'multi_modal_inputs': final_multi_modal_inputs_for_actor, # DICT of TENSOR features for PPO actor
    }
    

@contextmanager
def _timer(name: str, timing_raw: Dict[str, float]):
    with Timer(name=name, logger=None) as timer:
        yield
    if name not in timing_raw:
        timing_raw[name] = 0
    timing_raw[name] += timer.last


class RayPPOTrainer(object):
    """
    Note that this trainer runs on the driver process on a single CPU/GPU node.
    """

    # TODO: support each role have individual ray_worker_group_cls,
    # i.e., support different backend of different role
    def __init__(self,
                 config,
                 tokenizer,
                 role_worker_mapping: dict[Role, WorkerType],
                 resource_pool_manager: ResourcePoolManager,
                 ray_worker_group_cls: RayWorkerGroup = RayWorkerGroup,
                 processor=None,
                 reward_fn=None,
                 val_reward_fn=None):

        # assert torch.cuda.is_available(), 'cuda must be available on driver'

        self.tokenizer = tokenizer
        self.processor = processor
        self.config = config
        self.reward_fn = reward_fn
        self.val_reward_fn = val_reward_fn

        self.hybrid_engine = config.actor_rollout_ref.hybrid_engine
        assert self.hybrid_engine, 'Currently, only support hybrid engine'

        if self.hybrid_engine:
            assert Role.ActorRollout in role_worker_mapping, f'{role_worker_mapping.keys()=}'

        self.role_worker_mapping = role_worker_mapping
        self.resource_pool_manager = resource_pool_manager
        self.use_reference_policy = Role.RefPolicy in role_worker_mapping
        self.use_rm = Role.RewardModel in role_worker_mapping
        self.ray_worker_group_cls = ray_worker_group_cls
        self.validation_generations_logger = ValidationGenerationsLogger()

        # define in-reward KL control
        # kl loss control currently not suppoorted
        if config.algorithm.use_kl_in_reward:
            self.kl_ctrl_in_reward = core_algos.get_kl_controller(config.algorithm.kl_ctrl)

        if self.config.algorithm.adv_estimator == AdvantageEstimator.GAE:
            self.use_critic = True
        elif self.config.algorithm.adv_estimator in [
                AdvantageEstimator.GRPO, AdvantageEstimator.REINFORCE_PLUS_PLUS, AdvantageEstimator.REMAX,
                AdvantageEstimator.RLOO, AdvantageEstimator.REINFORCE_PLUS_PLUS_BASELINE
        ]:
            self.use_critic = False
        else:
            raise NotImplementedError

        self._validate_config()
        self._create_dataloader()
        print(f"DEBUG CONFIG Trainer Init Check: self.config.actor_rollout_ref.actor.ppo_mini_batch_size = {self.config.actor_rollout_ref.actor.ppo_mini_batch_size}")


    def _validate_config(self):
        config = self.config
        # number of GPUs total
        n_gpus = config.trainer.n_gpus_per_node * config.trainer.nnodes

        # 1. Check total batch size for data correctness
        real_train_batch_size = config.data.train_batch_size * config.actor_rollout_ref.rollout.n
        assert real_train_batch_size % n_gpus == 0, \
            f"real_train_batch_size ({real_train_batch_size}) must be divisible by total n_gpus ({n_gpus})."

        # A helper function to check "micro_batch_size" vs "micro_batch_size_per_gpu"
        # We throw an error if the user sets both. The new convention is "..._micro_batch_size_per_gpu".
        def check_mutually_exclusive(mbs, mbs_per_gpu, name: str):
            settings = {
                "actor_rollout_ref.actor": "micro_batch_size",
                "critic": "micro_batch_size",
                "reward_model": "micro_batch_size",
                "actor_rollout_ref.ref": "log_prob_micro_batch_size",
                "actor_rollout_ref.rollout": "log_prob_micro_batch_size",
            }

            if name in settings:
                param = settings[name]
                param_per_gpu = f"{param}_per_gpu"

                if mbs is None and mbs_per_gpu is None:
                    raise ValueError(
                        f"[{name}] Please set at least one of '{name}.{param}' or '{name}.{param_per_gpu}'.")

                if mbs is not None and mbs_per_gpu is not None:
                    raise ValueError(
                        f"[{name}] You have set both '{name}.{param}' AND '{name}.{param_per_gpu}'. "
                        f"Please remove '{name}.{param}' because only '*_{param_per_gpu}' is supported (the former is deprecated)."
                    )

        if not config.actor_rollout_ref.actor.use_dynamic_bsz:
            # actor: ppo_micro_batch_size vs. ppo_micro_batch_size_per_gpu
            check_mutually_exclusive(config.actor_rollout_ref.actor.ppo_micro_batch_size,
                                     config.actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu,
                                     "actor_rollout_ref.actor")

            if self.use_reference_policy:
                # reference: log_prob_micro_batch_size vs. log_prob_micro_batch_size_per_gpu
                check_mutually_exclusive(config.actor_rollout_ref.ref.log_prob_micro_batch_size,
                                         config.actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu,
                                         "actor_rollout_ref.ref")

            #  The rollout section also has log_prob_micro_batch_size vs. log_prob_micro_batch_size_per_gpu
            check_mutually_exclusive(config.actor_rollout_ref.rollout.log_prob_micro_batch_size,
                                     config.actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu,
                                     "actor_rollout_ref.rollout")

        if self.use_critic and not config.critic.use_dynamic_bsz:
            # Check for critic micro-batch size conflicts
            check_mutually_exclusive(config.critic.ppo_micro_batch_size, config.critic.ppo_micro_batch_size_per_gpu,
                                     "critic")

        # Check for reward model micro-batch size conflicts
        if config.reward_model.enable and not config.reward_model.use_dynamic_bsz:
            check_mutually_exclusive(config.reward_model.micro_batch_size, config.reward_model.micro_batch_size_per_gpu,
                                     "reward_model")

        # Actor
        # check if train_batch_size is larger than ppo_mini_batch_size
        # if NOT dynamic_bsz, we must ensure:
        #    ppo_mini_batch_size is divisible by ppo_micro_batch_size
        #    ppo_micro_batch_size * sequence_parallel_size >= n_gpus
        if not config.actor_rollout_ref.actor.use_dynamic_bsz:
            assert config.data.train_batch_size >= config.actor_rollout_ref.actor.ppo_mini_batch_size
            sp_size = config.actor_rollout_ref.actor.get('ulysses_sequence_parallel_size', 1)
            if config.actor_rollout_ref.actor.ppo_micro_batch_size is not None:
                assert config.actor_rollout_ref.actor.ppo_mini_batch_size % config.actor_rollout_ref.actor.ppo_micro_batch_size == 0
                assert config.actor_rollout_ref.actor.ppo_micro_batch_size * sp_size >= n_gpus

        assert config.actor_rollout_ref.actor.loss_agg_mode in [
            "token-mean", "seq-mean-token-sum", "seq-mean-token-mean"
        ], f"Invalid loss_agg_mode: {config.actor_rollout_ref.actor.loss_agg_mode}"

        if config.algorithm.use_kl_in_reward and config.actor_rollout_ref.actor.use_kl_loss:
            print(f"NOTICE: You have both enabled in-reward kl and kl loss.")

        # critic
        if self.use_critic and not config.critic.use_dynamic_bsz:
            assert config.data.train_batch_size >= config.critic.ppo_mini_batch_size
            sp_size = config.critic.get('ulysses_sequence_parallel_size', 1)
            if config.critic.ppo_micro_batch_size is not None:
                assert config.critic.ppo_mini_batch_size % config.critic.ppo_micro_batch_size == 0
                assert config.critic.ppo_micro_batch_size * sp_size >= n_gpus

        # Check if use_remove_padding is enabled when using sequence parallelism for fsdp
        if config.actor_rollout_ref.actor.strategy == 'fsdp':
            if config.actor_rollout_ref.actor.get('ulysses_sequence_parallel_size', 1) > 1 or \
                    config.actor_rollout_ref.ref.get('ulysses_sequence_parallel_size', 1) > 1:
                assert config.actor_rollout_ref.model.use_remove_padding, \
                    "When using sequence parallelism for actor/ref policy, you must enable `use_remove_padding`."

        if self.use_critic and config.critic.strategy == 'fsdp':
            if config.critic.get('ulysses_sequence_parallel_size', 1) > 1:
                assert config.critic.model.use_remove_padding, \
                    "When using sequence parallelism for critic, you must enable `use_remove_padding`."

        if config.data.get('val_batch_size', None) is not None:
            print(
                f"WARNING: val_batch_size is deprecated. Validation datasets are sent to inference engines as a whole batch, which will schedule the memory themselves."
            )

        # check eval config
        if config.actor_rollout_ref.rollout.val_kwargs.do_sample:
            assert config.actor_rollout_ref.rollout.temperature > 0, \
                "validation gen temperature should be greater than 0 when enabling do_sample"

        print("[validate_config] All configuration checks passed successfully!")

    def _create_dataloader(self):
        # TODO: we have to make sure the batch size is divisible by the dp size
        from verl.utils.import_utils import load_extern_type
        if "custom_cls" in self.config.data and self.config.data.custom_cls.get("path", None) is not None:
            dataset_cls = load_extern_type(self.config.data.custom_cls.path, self.config.data.custom_cls.name)
            if not issubclass(dataset_cls, Dataset):
                raise TypeError(f"The custom dataset class '{self.config.data.custom_cls.name}' from "
                                f"'{self.config.data.custom_cls.path}' must inherit from torch.utils.data.Dataset")
        else:
            dataset_cls = RLHFDataset

        self.train_dataset = dataset_cls(
            data_files=self.config.data.train_files,
            tokenizer=self.tokenizer,
            processor=self.processor,
            config=self.config.data,
        )
        # print("train_datset", self.train_dataset)

        # use sampler for better ckpt resume
        if self.config.data.shuffle:
            train_dataloader_generator = torch.Generator()
            train_dataloader_generator.manual_seed(self.config.data.get('seed', 1))
            sampler = RandomSampler(data_source=self.train_dataset, generator=train_dataloader_generator)
        else:
            sampler = SequentialSampler(data_source=self.train_dataset)

        self.train_dataloader = StatefulDataLoader(dataset=self.train_dataset,
                                                   batch_size=self.config.data.get('gen_batch_size',
                                                                                   self.config.data.train_batch_size),
                                                   num_workers=8,
                                                   drop_last=True,
                                                   collate_fn=collate_fn,
                                                   sampler=sampler)
        print(f"train_dataloader: {self.train_dataloader}")
        self.val_dataset = dataset_cls(
            data_files=self.config.data.val_files,
            tokenizer=self.tokenizer,
            processor=self.processor,
            config=self.config.data,
        )
        self.val_dataloader = StatefulDataLoader(
            dataset=self.val_dataset,
            # Validation datasets are sent to inference engines as a whole batch,
            # which will schedule the memory themselves.
            batch_size=len(self.val_dataset),
            num_workers=8,
            shuffle=False,
            drop_last=False,
            collate_fn=collate_fn)

        assert len(self.train_dataloader) >= 1
        assert len(
            self.val_dataloader
        ) == 1, "Validation dataloader must have a single batch, which inference engines will schedule the memory themselves."

        print(f'Size of train dataloader: {len(self.train_dataloader)}')

        # inject total_training_steps to actor/critic optim_config. This is hacky.
        total_training_steps = len(self.train_dataloader) * self.config.trainer.total_epochs

        if self.config.trainer.total_training_steps is not None:
            total_training_steps = self.config.trainer.total_training_steps

        self.total_training_steps = total_training_steps
        print(f'Total training steps: {self.total_training_steps}')

        OmegaConf.set_struct(self.config, True)
        with open_dict(self.config):
            self.config.actor_rollout_ref.actor.optim.total_training_steps = total_training_steps
            self.config.critic.optim.total_training_steps = total_training_steps
        
        
        from verl.utils.dataset.vqa_dataset import TwoStageVideoQADataset
        
        two_stage_dataset = TwoStageVideoQADataset(
            data_files=self.config.data.train_files,
            tokenizer=self.tokenizer,
            processor=self.processor,
            config=self.config.data,
        )
        
        self.two_stage_dataloader = StatefulDataLoader(
            dataset=two_stage_dataset,
            batch_size=self.config.data.get('gen_batch_size',
                                            self.config.data.train_batch_size),
            num_workers=8,
            drop_last=True,
            collate_fn=collate_fn,
            sampler=sampler
        )
        
            
        # inject total_training_steps to actor/critic optim_config. This is hacky.
        total_training_steps = len(self.two_stage_dataloader) * self.config.trainer.total_epochs

        if self.config.trainer.total_training_steps is not None:
            total_training_steps = self.config.trainer.total_training_steps

        self.total_training_steps = total_training_steps
        print(f'Total training steps: {self.total_training_steps}')

        OmegaConf.set_struct(self.config, True)
        with open_dict(self.config):
            self.config.actor_rollout_ref.actor.optim.total_training_steps = total_training_steps
            self.config.critic.optim.total_training_steps = total_training_steps
        
        try:
             # Use validation files specified in config
             vqa_val_files = self.config.data.get("vqa_val_files", self.config.data.val_files)
             if not vqa_val_files:
                  print("[WARN] No specific 'vqa_val_files' found in config.data, falling back to 'val_files'. Ensure these contain VQA data.")
                  vqa_val_files = self.config.data.val_files
             if not vqa_val_files:
                  raise ValueError("No validation files found for VQA validation.")

             self.vqa_val_dataset = TwoStageVideoQADataset(
                 data_files=vqa_val_files,
                 tokenizer=self.tokenizer,
                 processor=self.processor,
                 config=self.config.data, # Use the same data config
             )

             # Validation: typically no shuffle, run as one large batch
             self.vqa_val_dataloader = StatefulDataLoader(
                 dataset=self.vqa_val_dataset,
                 batch_size=len(self.vqa_val_dataset), # Process all validation data at once
                 num_workers=self.config.data.get("val_num_workers", 4), # Use separate config or default
                 shuffle=False,
                 drop_last=False,
                 collate_fn=collate_fn # Use the same collate function
             )
             print(f"VQA Validation Dataloader created with {len(self.vqa_val_dataset)} samples.")
             assert len(self.vqa_val_dataloader) == 1, "VQA Validation dataloader should process all data in one batch."

        except Exception as e:
             print(f"[ERROR] Failed to create VQA validation dataloader: {e}")
             print("[WARN] VQA Validation will be skipped.")
             self.vqa_val_dataloader = None # Ensure it's None if creation fails
        # --- End VQA Validation Dataloader ---
        
        
    def _validate_vqa(self):
        """
        Performs validation using the two-stage VQA process (Generation + QA)
        and calls self.val_reward_fn. Aligns with fit_vqa's generation flow.
        Tracks frame number ratio.
        """
        step_info_prefix_val = f"[VQA VAL - Step {self.global_steps}]" # Use current global step for context
        print(f"{step_info_prefix_val} --- Running _validate_vqa (using self.val_reward_fn) ---")
        
        if not hasattr(self, 'vqa_val_dataloader') or self.vqa_val_dataloader is None:
             print(f"{step_info_prefix_val} [WARN] VQA Validation Dataloader not available. Skipping VQA validation.")
             return {}

        # To collect metrics for averaging at the end
        # Each key will hold a list of values from all processed items in the validation batch
        all_val_run_metrics_accumulator = defaultdict(list)

        # Validation loop should only run once as batch_size is len(dataset)
        for val_i, batch_dict_from_loader_val in enumerate(self.vqa_val_dataloader):
            if val_i > 0: 
                print(f"{step_info_prefix_val} [WARN] VQA Validation dataloader has more than one batch! Processing first only."); 
                break
            if not batch_dict_from_loader_val: 
                print(f"{step_info_prefix_val} [DEBUG] Empty batch_dict in validation dataloader. Skipping."); 
                continue

            print(f"{step_info_prefix_val} --- Processing Validation Batch {val_i+1} ---")
            
            try:
                full_batch_proto_val = DataProto.from_single_dict(batch_dict_from_loader_val)
                if not full_batch_proto_val or len(full_batch_proto_val) == 0:
                     print(f"{step_info_prefix_val} [DEBUG] Empty DataProto from validation dataloader. Skipping."); 
                     continue
                current_batch_size_val = len(full_batch_proto_val)
                print(f"{step_info_prefix_val} [INFO] Loaded Validation Batch - Size: {current_batch_size_val}")
            except Exception as e:
                 print(f"{step_info_prefix_val} [ERROR] DataProto creation from validation batch: {e}. Skipping."); 
                 continue

            # --- Stage 1: Grounding Generation (Validation Mode) ---
            print(f"{step_info_prefix_val} -- Stage 1 (Validation): Generating Grounding --")
            decoded_grounding_texts_s1_val = ["<S1 VAL Gen Fail>"] * current_batch_size_val
            predicted_times_s1_val = [(None, None)] * current_batch_size_val
            s1_gen_output_proto_val = None # To check if S1 succeeded

            try:
                s1_input_batch_keys_val = ['input_ids', 'attention_mask', 'position_ids']
                # Assuming 'multi_modal_inputs' contains the FULL video features for Stage 1
                s1_input_non_tensor_keys_val = ['multi_modal_inputs'] 
                actual_s1_keys_val = [k for k in s1_input_batch_keys_val if k in full_batch_proto_val.batch]
                actual_s1_nt_keys_val = [k for k in s1_input_non_tensor_keys_val if k in full_batch_proto_val.non_tensor_batch]

                if not actual_s1_keys_val: raise ValueError("Missing essential Stage 1 tensor keys for validation.")

                s1_input_proto_for_gen_val = DataProto(
                   batch=full_batch_proto_val.batch.select(*actual_s1_keys_val),
                   non_tensor_batch={k: full_batch_proto_val.non_tensor_batch[k] for k in actual_s1_nt_keys_val}
                )
                s1_input_proto_for_gen_val.meta_info = {
                    'eos_token_id': self.tokenizer.eos_token_id, 
                    'pad_token_id': self.tokenizer.pad_token_id,
                    'max_new_tokens': self.config.actor_rollout_ref.rollout.get("max_new_tokens_grounding_val", 
                                       self.config.actor_rollout_ref.rollout.get("max_new_tokens_grounding", 50)),
                    'do_sample': self.config.actor_rollout_ref.rollout.get("do_sample_grounding_val", False), 
                    'temperature': self.config.actor_rollout_ref.rollout.get("temperature_grounding_val", 1.0),
                    'validate': True 
                }
                s1_input_padded_val, s1_pad_val = pad_dataproto_to_divisor(s1_input_proto_for_gen_val, self.actor_rollout_wg.world_size)
                s1_output_padded_val = self.actor_rollout_wg.generate_sequences(s1_input_padded_val)
                s1_gen_output_proto_val = unpad_dataproto(s1_output_padded_val, pad_size=s1_pad_val)

                s1_responses_val = s1_gen_output_proto_val.batch.get('responses')
                if s1_responses_val is None: raise ValueError("S1 VAL generation failed (no 'responses' tensor in output).")
                
                decoded_grounding_texts_s1_val = self.tokenizer.batch_decode(s1_responses_val, skip_special_tokens=True)
                predicted_times_s1_val = parse_grounding_times(decoded_grounding_texts_s1_val)
            except Exception as e:
                print(f"{step_info_prefix_val} [ERROR VAL S1] {e}"); import traceback; traceback.print_exc()
                # If S1 fails, all items in this batch will effectively have S1 fail for reward calculation later

            # --- Stage 2: Prepare Inputs & Generate QA (Validation Mode) ---
            print(f"{step_info_prefix_val} -- Stage 2 (Validation): Preparing Inputs & Generating QA --")
            s2_items_for_collation_val = [] 
            s2_valid_original_indices_val = [] 
            
            # For frame ratio metric - correspond to items that *attempt* S2 prep
            s2_attempted_actual_frames_val = []
            s2_attempted_original_2fps_frames_val = []

            for original_idx_val_loop in range(current_batch_size_val):
                item_s2_prep_prefix = f"{step_info_prefix_val} [S2 Prep VAL Idx {original_idx_val_loop}]"
                try:
                    pred_start_s1_val, pred_end_s1_val = predicted_times_s1_val[original_idx_val_loop]
                    gt_start_val = full_batch_proto_val.non_tensor_batch['start_time'][original_idx_val_loop]
                    gt_end_val = full_batch_proto_val.non_tensor_batch['end_time'][original_idx_val_loop]
                    
                    final_clip_start_val, final_clip_end_val = gt_start_val, gt_end_val
                    clip_source_log_val = "GT_Times_Val"
                    if final_clip_start_val is None or final_clip_end_val is None or \
                       not (isinstance(final_clip_start_val, (int, float)) and isinstance(final_clip_end_val, (int, float)) and final_clip_start_val >= 0 and final_clip_start_val < final_clip_end_val):
                        final_clip_start_val, final_clip_end_val = pred_start_s1_val, pred_end_s1_val
                        clip_source_log_val = "PredictedS1_Val"
                        if final_clip_start_val is None or final_clip_end_val is None or \
                           not (isinstance(final_clip_start_val, (int, float)) and isinstance(final_clip_end_val, (int, float)) and final_clip_start_val >= 0 and final_clip_start_val < final_clip_end_val):
                            final_clip_start_val, final_clip_end_val = 0.0, None
                            clip_source_log_val = "FullVideoFallback_Val"
                    # print(f"{item_s2_prep_prefix} Clip source: {clip_source_log_val}, Times: [{final_clip_start_val}, {final_clip_end_val}]")

                    video_path_s2_val = full_batch_proto_val.non_tensor_batch["video_path"][original_idx_val_loop]
                    
                    video_proc_config_item_val = full_batch_proto_val.non_tensor_batch["video_processing_config"][original_idx_val_loop]
                    # print(f"{item_s2_prep_prefix} Type of video_proc_config_item_val (from array) IS {type(video_proc_config_item_val)}")
                    if not isinstance(video_proc_config_item_val, dict):
                        if isinstance(video_proc_config_item_val, np.ndarray) and video_proc_config_item_val.ndim == 0 and hasattr(video_proc_config_item_val, 'item') and isinstance(video_proc_config_item_val.item(), dict):
                            # print(f"{item_s2_prep_prefix} It's a 0-dim ndarray containing a dict. Extracting dict using .item().")
                            video_proc_config_item_val = video_proc_config_item_val.item()
                        else:
                            raise TypeError(f"video_proc_config_item_val is {type(video_proc_config_item_val)}, not dict.")
                    
                    current_clip_video_proc_config_val = dict(video_proc_config_item_val) # Use dict() for a new, clean dict
                    # print(f"{item_s2_prep_prefix} Type of current_clip_video_proc_config_val (after dict()) IS {type(current_clip_video_proc_config_val)}")
                    # print(f"{item_s2_prep_prefix} Contents of current_clip_video_proc_config_val: {current_clip_video_proc_config_val}")

                    if not isinstance(current_clip_video_proc_config_val, dict): raise TypeError("current_clip_video_proc_config_val became non-dict before nframes get.")
                    nframes_val_check_s2 = current_clip_video_proc_config_val.get("nframes")
                    if nframes_val_check_s2 is not None:
                        if not isinstance(current_clip_video_proc_config_val, dict): raise TypeError("became non-dict before fps pop.")
                        current_clip_video_proc_config_val.pop("fps", None)
                    else: 
                        if not isinstance(current_clip_video_proc_config_val, dict): raise TypeError("became non-dict before nframes pop/setdefault.")
                        current_clip_video_proc_config_val.pop("nframes", None)
                        current_clip_video_proc_config_val.setdefault("fps", 2)

                    ele_segment_s2_val = {"video": video_path_s2_val, 
                                          "video_start": final_clip_start_val, "video_end": final_clip_end_val,
                                          "original_idx": original_idx_val_loop, **current_clip_video_proc_config_val}
                    
                    if not isinstance(current_clip_video_proc_config_val, dict): raise TypeError("became non-dict before image_factor get.")
                    image_factor_val_s2 = current_clip_video_proc_config_val.get("image_factor", 28)
                    
                    clipped_video_frames_s2_val = fetch_video(ele_segment_s2_val, image_factor=image_factor_val_s2)
                    if clipped_video_frames_s2_val is None or clipped_video_frames_s2_val.nelement() == 0:
                        raise ValueError(f"Clipped video for validation item {original_idx_val_loop} is empty.")

                    # Frame Ratio Calculation
                    num_actual_s2_frames = clipped_video_frames_s2_val.shape[0]
                    s2_attempted_actual_frames_val.append(num_actual_s2_frames)
                    
                    # Get original_video_nframes (total frames of the Stage 1 input video)
                    # This should have been added to non_tensor_batch by TwoStageVideoQADataset
                    s1_original_total_frames = full_batch_proto_val.non_tensor_batch.get("original_video_nframes")[original_idx_val_loop]

                    if s1_original_total_frames is not None:
                        s2_attempted_original_2fps_frames_val.append(s1_original_total_frames)
                        # print("type(s1_original_total_frames)", type(s1_original_total_frames))
                        # print(f"{item_s2_prep_prefix} Frame Ratio: Actual S2 frames = {num_actual_s2_frames}, Original 2fps frames = {s1_original_total_frames}.")
                    else:
                        s2_attempted_original_2fps_frames_val.append(np.nan)
                        print(f"{item_s2_prep_prefix} Frame Ratio: Could not calculate original 2fps frames (missing S1 duration/fps for item). Actual S2 frames = {num_actual_s2_frames}.")

                    question_text_s2_val = full_batch_proto_val.non_tensor_batch["question_text"][original_idx_val_loop]
                    item_data_for_s2_processor_val = {"question_text": question_text_s2_val, "clipped_video": clipped_video_frames_s2_val, "original_index": original_idx_val_loop}
                    processed_s2_item_dict_val = prepare_stage2_inputs_for_item(item_data_for_s2_processor_val, self.processor, self.tokenizer, self.config)
                    
                    if processed_s2_item_dict_val:
                        s2_items_for_collation_val.append(processed_s2_item_dict_val)
                        s2_valid_original_indices_val.append(original_idx_val_loop)
                except Exception as e_s2_prep_val:
                    print(f"{item_s2_prep_prefix} [WARN VAL S2 Prep Error] {e_s2_prep_val}. Item skipped.")
                    # Add NaN for frame counts if item is skipped here
                    if len(s2_attempted_actual_frames_val) == original_idx_val_loop: # only append if not already appended by a partial success
                        s2_attempted_actual_frames_val.append(np.nan) 
                        s2_attempted_original_2fps_frames_val.append(np.nan)


            s2_qa_gen_output_proto_val = None
            s2_input_proto_for_qa_gen_val = None # Will hold the input to S2 QA generation for valid items
            if s2_items_for_collation_val:
                print(f"{step_info_prefix_val} -- Stage 2 (Validation): Generating QA for {len(s2_items_for_collation_val)} valid items --")
                try:
                    collated_s2_val_input_dict = collate_fn(s2_items_for_collation_val)
                    s2_input_proto_for_qa_gen_val = DataProto.from_single_dict(collated_s2_val_input_dict)
                    s2_input_proto_for_qa_gen_val.meta_info = {
                        'eos_token_id': self.tokenizer.eos_token_id, 'pad_token_id': self.tokenizer.pad_token_id,
                        'max_new_tokens': self.config.actor_rollout_ref.rollout.get("max_new_tokens_qa_val", 
                                           self.config.actor_rollout_ref.rollout.get("max_new_tokens_qa", 10)),
                        'do_sample': self.config.actor_rollout_ref.rollout.get("do_sample_qa_val", False),
                        'temperature': self.config.actor_rollout_ref.rollout.get("temperature_qa_val", 1.0),
                        'validate': True
                    }
                    s2_qa_gen_input_padded_val, s2_qa_gen_pad_val = pad_dataproto_to_divisor(s2_input_proto_for_qa_gen_val, self.actor_rollout_wg.world_size)
                    s2_qa_gen_output_padded_val = self.actor_rollout_wg.generate_sequences(s2_qa_gen_input_padded_val)
                    s2_qa_gen_output_proto_val = unpad_dataproto(s2_qa_gen_output_padded_val, pad_size=s2_qa_gen_pad_val)
                    if 'responses' not in s2_qa_gen_output_proto_val.batch:
                        raise ValueError("S2 VAL QA generation failed (no 'responses' tensor).")
                except Exception as e_s2_gen_val:
                    print(f"{step_info_prefix_val} [ERROR VAL S2 Gen] {e_s2_gen_val}"); import traceback; traceback.print_exc()
                    s2_qa_gen_output_proto_val = None
            else:
                print(f"{step_info_prefix_val} [INFO VAL S2 Gen] No valid items prepped for Stage 2 QA generation.")


            # --- Assemble DataProto for Validation Reward Function ---
            val_reward_input_proto = None
            if s2_qa_gen_output_proto_val and s2_valid_original_indices_val and 'responses' in s2_qa_gen_output_proto_val.batch:
                try:
                    print(f"{step_info_prefix_val} [VQA VAL DEBUG] Assembling DataProto for val_reward_fn using {len(s2_valid_original_indices_val)} valid S2 items.")
                    s2_val_responses_tensor = s2_qa_gen_output_proto_val.batch['responses']
                    s2_val_attn_mask_full = s2_qa_gen_output_proto_val.batch.get('attention_mask')
                    s2_val_response_len = s2_val_responses_tensor.shape[1]
                    s2_val_response_masks_tensor = (s2_val_responses_tensor != self.tokenizer.pad_token_id).long()
                    if s2_val_attn_mask_full is not None and s2_val_attn_mask_full.shape[1] >= s2_val_response_len:
                         # Use actual response mask from generation output if available and covers the response part
                        s2_val_response_masks_tensor = s2_val_attn_mask_full[:, -s2_val_response_len:]
                    
                    reward_non_tensor_data_val = defaultdict(list)
                    for valid_s2_orig_idx in s2_valid_original_indices_val:
                        reward_non_tensor_data_val['decoded_grounding_texts'].append(decoded_grounding_texts_s1_val[valid_s2_orig_idx])
                        reward_non_tensor_data_val['start_time'].append(full_batch_proto_val.non_tensor_batch['start_time'][valid_s2_orig_idx])
                        reward_non_tensor_data_val['end_time'].append(full_batch_proto_val.non_tensor_batch['end_time'][valid_s2_orig_idx])
                        reward_non_tensor_data_val['ground_truth_answer'].append(full_batch_proto_val.non_tensor_batch['ground_truth_answer'][valid_s2_orig_idx])
                    
                    num_valid_s2_items_for_reward = s2_val_responses_tensor.shape[0]
                    for k_nt_val, v_list_nt_val in reward_non_tensor_data_val.items():
                        if len(v_list_nt_val) != num_valid_s2_items_for_reward:
                            raise ValueError(f"Length mismatch for non-tensor key '{k_nt_val}' in validation reward prep. Expected {num_valid_s2_items_for_reward}, got {len(v_list_nt_val)}")
                    
                    reward_non_tensor_data_np_val = {k: np.array(v, dtype=object) for k, v in reward_non_tensor_data_val.items()}

                    val_reward_input_proto = DataProto(
                        batch=TensorDict({"responses": s2_val_responses_tensor, "response_mask": s2_val_response_masks_tensor}, 
                                         batch_size=[num_valid_s2_items_for_reward]), 
                        non_tensor_batch=reward_non_tensor_data_np_val
                    )
                except Exception as val_reward_prep_err:
                    print(f"{step_info_prefix_val} [ERROR VAL Reward Prep] {val_reward_prep_err}"); import traceback; traceback.print_exc()
                    val_reward_input_proto = None

            # --- Call Validation Reward Function ---
            if val_reward_input_proto:
                # print(f"{step_info_prefix_val} [VQA VAL DEBUG] Calling self.val_reward_fn with {len(val_reward_input_proto)} items.")
                try:
                    result_val_reward_fn = self.val_reward_fn(val_reward_input_proto, return_dict=True)
                    if "reward_extra_info" in result_val_reward_fn and isinstance(result_val_reward_fn["reward_extra_info"], dict):
                        for metric_key_val, metric_values_list_val in result_val_reward_fn["reward_extra_info"].items():
                            all_val_run_metrics_accumulator[metric_key_val].extend(list(metric_values_list_val)) 
                    else:
                        print(f"{step_info_prefix_val} [VQA VAL WARN] 'reward_extra_info' from val_reward_fn is missing or not a dict.")
                except Exception as val_reward_call_err:
                    print(f"{step_info_prefix_val} [ERROR VAL Reward Call] {val_reward_call_err}"); import traceback; traceback.print_exc()
            else:
                 print(f"{step_info_prefix_val} [VQA VAL WARN] Skipping val reward calculation (no valid input proto).")
            
            # Add frame ratio for S2 items that were successfully processed for reward
            # s2_valid_original_indices_val maps items in val_reward_input_proto back to their original batch position
            frame_ratios_for_log_val = []
            for i, original_batch_idx in enumerate(s2_valid_original_indices_val): # Iterate through indices that made it to reward_fn
                 # s2_attempted_actual_frames_val and s2_attempted_original_2fps_frames_val are indexed by original_batch_idx
                if original_batch_idx < len(s2_attempted_actual_frames_val) and original_batch_idx < len(s2_attempted_original_2fps_frames_val):
                    actual_f_val = s2_attempted_actual_frames_val[original_batch_idx]
                    orig_2fps_f_val = s2_attempted_original_2fps_frames_val[original_batch_idx]
                    
                    frame_ratios_for_log_val.append(actual_f_val / orig_2fps_f_val)
                    
                else: # Should not happen if lists are built correctly
                    frame_ratios_for_log_val.append(np.nan)

            all_val_run_metrics_accumulator['s2_frame_num_ratio'].extend(frame_ratios_for_log_val)

        # --- Process Aggregated Metrics for the entire validation set ---
        print(f"{step_info_prefix_val} [VQA VAL DEBUG] Processing all aggregated validation metrics from accumulator...")
        final_val_metrics_summary = {}
        vqa_target_metrics_for_summary = ['grounding_accuracy', 'grounding_format', 'grounding_score', 
                                          'qa_accuracy', 'final_score', 's2_frame_num_ratio']
        
        for metric_name_val in vqa_target_metrics_for_summary:
            values_val = all_val_run_metrics_accumulator.get(metric_name_val, [])
            numeric_values_val = [v for v in values_val if isinstance(v, (int, float, np.number)) and np.isfinite(v)]
            if numeric_values_val:
                log_prefix_val = f"val/vqa/{metric_name_val}" 
                final_val_metrics_summary[f"{log_prefix_val}/mean"] = float(np.mean(numeric_values_val))
                final_val_metrics_summary[f"{log_prefix_val}/max"] = float(np.max(numeric_values_val))
                final_val_metrics_summary[f"{log_prefix_val}/min"] = float(np.min(numeric_values_val))
                final_val_metrics_summary[f"{log_prefix_val}/std"] = float(np.std(numeric_values_val))
                final_val_metrics_summary[f"{log_prefix_val}/count"] = len(numeric_values_val)
            else:
                print(f"{step_info_prefix_val} [VQA VAL WARN] No valid numeric values for metric '{metric_name_val}' in summary.")

        print(f"{step_info_prefix_val} [VQA VAL INFO] Validation metrics summary: {pformat(final_val_metrics_summary)}")
        return final_val_metrics_summary
    
    def _maybe_log_val_generations(self, inputs, outputs, scores):
        """Log a table of validation samples to the configured logger (wandb or swanlab)"""

        generations_to_log = self.config.trainer.log_val_generations

        if generations_to_log == 0:
            return

        import numpy as np

        # Create tuples of (input, output, score) and sort by input text
        samples = list(zip(inputs, outputs, scores))
        samples.sort(key=lambda x: x[0])  # Sort by input text

        # Use fixed random seed for deterministic shuffling
        rng = np.random.RandomState(42)
        rng.shuffle(samples)

        # Take first N samples after shuffling
        samples = samples[:generations_to_log]

        # Log to each configured logger
        self.validation_generations_logger.log(self.config.trainer.logger, samples, self.global_steps)
    

    def _validate(self):
        print("[DEBUG] --- Running _validate ---")
        all_val_results = [] # To store detailed results for saving
        data_source_lst = []
        reward_extra_infos_dict: dict[str, list] = defaultdict(list)
        sample_inputs_for_logging = [] # For _maybe_log_val_generations
        sample_outputs_for_logging = [] # For _maybe_log_val_generations
        sample_scores_for_logging = [] # For _maybe_log_val_generations

        # Validation loop should only run once
        for val_i, test_data in enumerate(self.val_dataloader):
            if val_i > 0: print("Warning: Validation dataloader has more than one batch!") # Should not happen

            test_batch = DataProto.from_single_dict(test_data)
            if len(test_batch) == 0: continue # Skip empty batches
            original_batch_size = len(test_batch)

            # Store original prompts and ground truth before modification/repeat
            if "prompts" in test_batch.non_tensor_batch:
                 original_prompts = deepcopy(test_batch.non_tensor_batch["prompts"])
            else: # Fallback if 'prompts' key wasn't added
                 original_prompts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in test_batch.batch['input_ids']]
            original_ground_truth = deepcopy(test_batch.non_tensor_batch.get("ground_truth"))
            # Store other original info if needed for saving all_val_results
            original_videos = deepcopy(test_batch.non_tensor_batch.get("videos"))
            # original_video_lengths = deepcopy(test_batch.non_tensor_batch.get("video_length"))


            # Repeat test batch according to validation config 'n'
            repeat_n = self.config.actor_rollout_ref.rollout.val_kwargs.get('n', 1) # Default to 1 if not set
            if repeat_n > 1: # Avoid repeat if n=1
                test_batch = test_batch.repeat(repeat_times=repeat_n, interleave=True)

            # --- WORKAROUND POP START (from Turn 59/61) ---
            print("[DEBUG VAL WORKAROUND] Popping only essential tensor keys...")
            essential_tensor_keys = ['input_ids', 'attention_mask', 'position_ids']
            popped_tensors_for_gen = {}
            temp_meta_info = test_batch.meta_info
            batch_size_for_td = (0,) # Default for empty case

            for key in essential_tensor_keys:
                 if key in test_batch.batch:
                     popped_tensors_for_gen[key] = test_batch.batch[key]
                     if batch_size_for_td == (0,): # Get batch size from first available tensor
                         batch_size_for_td = popped_tensors_for_gen[key].shape[:1]
                 else: print(f"Warning: Key '{key}' not in test_batch.batch for val pop.")

            # Ensure batch_size_for_td is set correctly if possible
            if not popped_tensors_for_gen: batch_size_for_td = (0,) # No tensors popped
            elif batch_size_for_td == (0,): batch_size_for_td = (len(test_batch),) # Fallback if no tensors had shape

            test_gen_batch = DataProto(
                batch=TensorDict(popped_tensors_for_gen, batch_size=batch_size_for_td),
                non_tensor_batch={}, meta_info=temp_meta_info
            )

            keys_to_add_back = []
            if 'multi_modal_data' in test_batch.non_tensor_batch: keys_to_add_back.append('multi_modal_data')
            if 'multi_modal_inputs' in test_batch.non_tensor_batch: keys_to_add_back.append('multi_modal_inputs')
            for key in keys_to_add_back:
                 if key in test_batch.non_tensor_batch:
                      test_gen_batch.non_tensor_batch[key] = test_batch.non_tensor_batch[key]
                      # Don't delete from original test_batch here, just copy needed info
                      # del test_batch.non_tensor_batch[key]
                 else: print(f"Warning: Key '{key}' to add back not found in val non_tensor_batch.")

            # Exclude the popped tensors from the original test_batch
            test_batch.batch = test_batch.batch.exclude(*essential_tensor_keys)
            # --- WORKAROUND POP END ---

            test_gen_batch.meta_info = {
                'eos_token_id': self.tokenizer.eos_token_id, 'pad_token_id': self.tokenizer.pad_token_id,
                'recompute_log_prob': False, 'do_sample': self.config.actor_rollout_ref.rollout.val_kwargs.do_sample,
                'validate': True,
            }
            print(f'[DEBUG VAL] test_gen_batch meta info: {test_gen_batch.meta_info}')

            # Pad, generate, unpad
            test_gen_batch_padded, pad_size = pad_dataproto_to_divisor(test_gen_batch, self.actor_rollout_wg.world_size)
            test_output_gen_batch_padded = self.actor_rollout_wg.generate_sequences(test_gen_batch_padded)
            test_output_gen_batch = unpad_dataproto(test_output_gen_batch_padded, pad_size=pad_size)
            print('[DEBUG VAL] validation generation end')

            output_ids = test_output_gen_batch.batch['responses']
            output_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in output_ids]

            # Reconstruct batch for reward function
            # Note: test_batch now contains remaining tensors (if any) + original non-tensors
            test_batch = test_batch.union(test_output_gen_batch)
            # Add back ground truth and video length (repeated)
            if original_ground_truth is not None:
                 repeated_gt = np.repeat(original_ground_truth, repeat_n, axis=0)
                 test_batch.non_tensor_batch["ground_truth"] = repeated_gt
            # if original_video_lengths is not None:
            #      repeated_vl = np.repeat(original_video_lengths, repeat_n, axis=0)
            #      test_batch.non_tensor_batch["video_length"] = repeated_vl

            # Evaluate using reward function
            print("[DEBUG VAL] Calculating rewards...")
            result = self.val_reward_fn(test_batch, return_dict=True) # Use val_reward_fn
            print("[DEBUG VAL] Rewards calculated.")
            reward_tensor = result["reward_tensor"]
            scores = reward_tensor.sum(-1).cpu().tolist()

            # Collect detailed results for saving
            current_reward_extra = result.get("reward_extra_info", {})
            for idx in range(original_batch_size):
                 prompt = original_prompts[idx]
                 gt = original_ground_truth[idx] if original_ground_truth is not None else "N/A"
                #  vid_len = original_video_lengths[idx] if original_video_lengths is not None else "N/A"
                 video_ref = original_videos[idx][0] if original_videos is not None and len(original_videos[idx]) > 0 else "N/A"
                 results_for_prompt = []
                 extra_info_for_prompt = defaultdict(list)

                 for repeat_idx in range(repeat_n):
                      global_idx = idx * repeat_n + repeat_idx
                      results_for_prompt.append({
                          "response": output_texts[global_idx],
                          "score": scores[global_idx]
                      })
                      for key, values in current_reward_extra.items():
                           if values is not None and len(values) > global_idx:
                                extra_info_for_prompt[key].append(values[global_idx])
                           else:
                                extra_info_for_prompt[key].append(None) # Append None if missing

                 all_val_results.append({
                     "step": self.global_steps, "prompt": prompt, "ground_truth": gt,
                      "video_ref": video_ref,
                     "outputs": results_for_prompt, "extra_rewards": dict(extra_info_for_prompt)
                 })

            # Populate lists for metrics processing (match lengths)
            sample_inputs_for_logging.extend(original_prompts * repeat_n)
            sample_outputs_for_logging.extend(output_texts)
            sample_scores_for_logging.extend(scores)
            reward_extra_infos_dict["reward"].extend(scores)
            if "reward_extra_info" in result:
                 for key, lst in result["reward_extra_info"].items():
                     # Ensure list has correct length (pad with NaN maybe?)
                     expected_len = len(scores)
                     actual_len = len(lst) if lst is not None else 0
                     if actual_len == expected_len:
                          reward_extra_infos_dict[key].extend(lst)
                     else:
                          print(f"Warning: Length mismatch for reward_extra_info key '{key}'. Expected {expected_len}, got {actual_len}. Padding with NaN.")
                          padded_lst = list(lst) + [np.nan] * (expected_len - actual_len) if lst is not None else [np.nan] * expected_len
                          reward_extra_infos_dict[key].extend(padded_lst[:expected_len])

            data_source_lst.append(test_batch.non_tensor_batch.get('data_source', ['unknown'] * len(scores)))


        # --- Save ALL validation results ---
        try:
            val_output_dir = os.path.join(self.config.trainer.default_local_dir, "validation_results")
            os.makedirs(val_output_dir, exist_ok=True)
            self.all_val_samples_file_path = os.path.join(val_output_dir, f"val_step_{self.global_steps}.json")
            print(f"[DEBUG VAL] Preparing validation results for JSON serialization...")
            serializable_val_results = convert_numpy_to_native(all_val_results) # Use helper function
            print(f"[DEBUG VAL] Saving all validation results ({len(serializable_val_results)} unique prompts) to: {self.all_val_samples_file_path}")
            with open(self.all_val_samples_file_path, 'w') as f:
                 json.dump(serializable_val_results, f, indent=2)
            print("[DEBUG VAL] Validation results saved successfully.")
        except Exception as e:
            print(f"Error saving validation results: {e}"); import traceback; traceback.print_exc()


        # --- Log subset to dashboard ---
        self._maybe_log_val_generations(inputs=sample_inputs_for_logging, outputs=sample_outputs_for_logging, scores=sample_scores_for_logging)

        # --- Process and return simplified metrics ---
        print("[DEBUG VAL] Processing metrics...")
        data_sources = np.concatenate(data_source_lst) if data_source_lst else np.array([])
        metric_dict = {}
        if len(data_sources) > 0 and reward_extra_infos_dict:
             # Use the simplified metric processor
             metric_dict = process_validation_metrics(
                  data_sources=data_sources,
                  sample_inputs=sample_inputs_for_logging, # Still needed for data source mapping if used
                  infos_dict=reward_extra_infos_dict,
                  # Define the specific metrics you want stats for
                  target_metrics=["reward", "tvg_accuracy", "tvg_format", "tvg_combined"]
             )
        else: print("[DEBUG VAL] No data sources or reward info found for metric processing.")
        print("[DEBUG VAL] Metrics processing complete.")

        return metric_dict
    
    def init_workers(self):
        """Init resource pool and worker group"""
        self.resource_pool_manager.create_resource_pool()

        self.resource_pool_to_cls = {pool: {} for pool in self.resource_pool_manager.resource_pool_dict.values()}
        actor_config_for_worker = self.config.actor_rollout_ref.actor # Adjust path if needed
        print(f"DEBUG CONFIG Trainer -> Actor Worker Init: Config passed to worker: ppo_mini_batch_size = {actor_config_for_worker.ppo_mini_batch_size}")


        # create actor and rollout
        if self.hybrid_engine:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.ActorRollout)
            actor_rollout_cls = RayClassWithInitArgs(cls=self.role_worker_mapping[Role.ActorRollout],
                                                     config=self.config.actor_rollout_ref,
                                                     role='actor_rollout')
            self.resource_pool_to_cls[resource_pool]['actor_rollout'] = actor_rollout_cls
        else:
            raise NotImplementedError

        # create critic
        if self.use_critic:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.Critic)
            critic_cls = RayClassWithInitArgs(cls=self.role_worker_mapping[Role.Critic], config=self.config.critic)
            self.resource_pool_to_cls[resource_pool]['critic'] = critic_cls

        # create reference policy if needed
        if self.use_reference_policy:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RefPolicy)
            ref_policy_cls = RayClassWithInitArgs(self.role_worker_mapping[Role.RefPolicy],
                                                  config=self.config.actor_rollout_ref,
                                                  role='ref')
            self.resource_pool_to_cls[resource_pool]['ref'] = ref_policy_cls

        # create a reward model if reward_fn is None
        if self.use_rm:
            # we create a RM here
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RewardModel)
            rm_cls = RayClassWithInitArgs(self.role_worker_mapping[Role.RewardModel], config=self.config.reward_model)
            self.resource_pool_to_cls[resource_pool]['rm'] = rm_cls

        # initialize WorkerGroup
        # NOTE: if you want to use a different resource pool for each role, which can support different parallel size,
        # you should not use `create_colocated_worker_cls`. Instead, directly pass different resource pool to different worker groups.
        # See https://github.com/volcengine/verl/blob/master/examples/ray/tutorial.ipynb for more information.
        all_wg = {}
        self.wg_dicts = []
        wg_kwargs = {}  # Setting up kwargs for RayWorkerGroup
        if OmegaConf.select(self.config.trainer, "ray_wait_register_center_timeout") is not None:
            wg_kwargs["ray_wait_register_center_timeout"] = self.config.trainer.ray_wait_register_center_timeout

        for resource_pool, class_dict in self.resource_pool_to_cls.items():
            worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)
            wg_dict = self.ray_worker_group_cls(resource_pool=resource_pool,
                                                ray_cls_with_init=worker_dict_cls,
                                                **wg_kwargs)
            spawn_wg = wg_dict.spawn(prefix_set=class_dict.keys())
            all_wg.update(spawn_wg)
            # keep the referece of WorkerDict to support ray >= 2.31. Ref: https://github.com/ray-project/ray/pull/45699
            self.wg_dicts.append(wg_dict)

        if self.use_critic:
            self.critic_wg = all_wg['critic']
            self.critic_wg.init_model()

        if self.use_reference_policy:
            self.ref_policy_wg = all_wg['ref']
            self.ref_policy_wg.init_model()

        if self.use_rm:
            self.rm_wg = all_wg['rm']
            self.rm_wg.init_model()

        # we should create rollout at the end so that vllm can have a better estimation of kv cache memory
        self.actor_rollout_wg = all_wg['actor_rollout']
        self.actor_rollout_wg.init_model()
        print("DEBUG CONFIG Trainer Init: actor ppo_mini_batch_size =", self.config.actor_rollout_ref.actor.ppo_mini_batch_size)


    def _save_checkpoint(self):
        # path: given_path + `/global_step_{global_steps}` + `/actor`
        local_global_step_folder = os.path.join(self.config.trainer.default_local_dir,
                                                f'global_step_{self.global_steps}')

        print(f'local_global_step_folder: {local_global_step_folder}')
        actor_local_path = os.path.join(local_global_step_folder, 'actor')

        actor_remote_path = None if self.config.trainer.default_hdfs_dir is None else os.path.join(
            self.config.trainer.default_hdfs_dir, f'global_step_{self.global_steps}', 'actor')

        remove_previous_ckpt_in_save = self.config.trainer.get('remove_previous_ckpt_in_save', False)
        if remove_previous_ckpt_in_save:
            print(
                'Warning: remove_previous_ckpt_in_save is deprecated, set max_actor_ckpt_to_keep=1 and max_critic_ckpt_to_keep=1 instead'
            )
        max_actor_ckpt_to_keep = self.config.trainer.get('max_actor_ckpt_to_keep',
                                                         None) if not remove_previous_ckpt_in_save else 1
        max_critic_ckpt_to_keep = self.config.trainer.get('max_critic_ckpt_to_keep',
                                                          None) if not remove_previous_ckpt_in_save else 1

        self.actor_rollout_wg.save_checkpoint(actor_local_path,
                                              actor_remote_path,
                                              self.global_steps,
                                              max_ckpt_to_keep=max_actor_ckpt_to_keep)

        if self.use_critic:
            critic_local_path = os.path.join(local_global_step_folder, 'critic')
            critic_remote_path = None if self.config.trainer.default_hdfs_dir is None else os.path.join(
                self.config.trainer.default_hdfs_dir, f'global_step_{self.global_steps}', 'critic')
            self.critic_wg.save_checkpoint(critic_local_path,
                                           critic_remote_path,
                                           self.global_steps,
                                           max_ckpt_to_keep=max_critic_ckpt_to_keep)

        # save dataloader
        dataloader_local_path = os.path.join(local_global_step_folder, 'data.pt')
        dataloader_state_dict = self.train_dataloader.state_dict()
        torch.save(dataloader_state_dict, dataloader_local_path)

        # latest checkpointed iteration tracker (for atomic usage)
        local_latest_checkpointed_iteration = os.path.join(self.config.trainer.default_local_dir,
                                                           'latest_checkpointed_iteration.txt')
        with open(local_latest_checkpointed_iteration, 'w') as f:
            f.write(str(self.global_steps))

    def _load_checkpoint(self):
        if self.config.trainer.resume_mode == 'disable':
            return 0

        # load from hdfs
        if self.config.trainer.default_hdfs_dir is not None:
            raise NotImplementedError('load from hdfs is not implemented yet')
        else:
            checkpoint_folder = self.config.trainer.default_local_dir  # TODO: check path
            if not os.path.isabs(checkpoint_folder):
                working_dir = os.getcwd()
                checkpoint_folder = os.path.join(working_dir, checkpoint_folder)
            global_step_folder = find_latest_ckpt_path(checkpoint_folder)  # None if no latest

        # find global_step_folder
        if self.config.trainer.resume_mode == 'auto':
            if global_step_folder is None:
                print('Training from scratch')
                return 0
        else:
            if self.config.trainer.resume_mode == "resume_path":
                assert isinstance(self.config.trainer.resume_from_path, str), "resume ckpt must be str type"
                assert 'global_step_' in self.config.trainer.resume_from_path, "resume ckpt must specify the global_steps"
                global_step_folder = self.config.trainer.resume_from_path
                if not os.path.isabs(global_step_folder):
                    working_dir = os.getcwd()
                    global_step_folder = os.path.join(working_dir, global_step_folder)
        print(f'Load from checkpoint folder: {global_step_folder}')
        # set global step
        self.global_steps = int(global_step_folder.split('global_step_')[-1])

        print(f'Setting global step to {self.global_steps}')
        print(f'Resuming from {global_step_folder}')

        actor_path = os.path.join(global_step_folder, 'actor')
        critic_path = os.path.join(global_step_folder, 'critic')
        # load actor
        self.actor_rollout_wg.load_checkpoint(actor_path,
                                              del_local_after_load=self.config.trainer.del_local_ckpt_after_load)
        # load critic
        if self.use_critic:
            self.critic_wg.load_checkpoint(critic_path,
                                           del_local_after_load=self.config.trainer.del_local_ckpt_after_load)

        # load dataloader,
        # TODO: from remote not implemented yet
        dataloader_local_path = os.path.join(global_step_folder, 'data.pt')
        if os.path.exists(dataloader_local_path):
            dataloader_state_dict = torch.load(dataloader_local_path, weights_only=False)
            self.train_dataloader.load_state_dict(dataloader_state_dict)
        else:
            print(f"Warning: No dataloader state found at {dataloader_local_path}, will start from scratch")

    def _balance_batch(self, batch: DataProto, metrics, logging_prefix='global_seqlen'):
        """Reorder the data on single controller such that each dp rank gets similar total tokens"""
        attention_mask = batch.batch['attention_mask']
        batch_size = attention_mask.shape[0]
        global_seqlen_lst = batch.batch['attention_mask'].view(batch_size, -1).sum(-1).tolist()  # (train_batch_size,)
        world_size = self.actor_rollout_wg.world_size
        global_partition_lst = get_seqlen_balanced_partitions(global_seqlen_lst,
                                                              k_partitions=world_size,
                                                              equal_size=True)
        # reorder based on index. The data will be automatically equally partitioned by dispatch function
        global_idx = torch.tensor([j for partition in global_partition_lst for j in partition])
        batch.reorder(global_idx)
        global_balance_stats = log_seqlen_unbalance(seqlen_list=global_seqlen_lst,
                                                    partitions=global_partition_lst,
                                                    prefix=logging_prefix)
        metrics.update(global_balance_stats)

    
    def fit(self):
        """
        The training loop of PPO.
        MODIFIED: Logs extra rewards and saves training samples periodically.
        """
        from verl.utils.tracking import Tracking
        from omegaconf import OmegaConf
        self.all_train_samples = []  # List to store training samples

        logger = Tracking(project_name=self.config.trainer.project_name,
                          experiment_name=self.config.trainer.experiment_name,
                          default_backend=self.config.trainer.logger,
                          config=OmegaConf.to_container(self.config, resolve=True))

        self.global_steps = 0
        print("--- Initializing Fit ---")

        print("--- Loading Checkpoint ---")
        self._load_checkpoint()
        print(f"--- Checkpoint Loaded, Global Step: {self.global_steps} ---")

        if self.val_reward_fn is not None and self.config.trainer.get('val_before_train', True):
            print("--- Performing Initial Validation ---")
            val_metrics = self._validate()
            pprint(f'Initial validation metrics: {val_metrics}')
            logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get('val_only', False): print("--- val_only=True, exiting fit ---"); return

        progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="Training Progress")
        self.global_steps += 1
        last_val_metrics = None

        print("--- Starting Training Loop ---")
        # Removed try...finally block as saving happens during the loop now

        for epoch in range(self.config.trainer.total_epochs):
            print(f"--- Starting Epoch {epoch+1}/{self.config.trainer.total_epochs} ---")
            for i, batch_dict in enumerate(self.two_stage_dataloader):
                print(f"\n--- Starting Global Step {self.global_steps} (Epoch {epoch+1}, Batch {i+1}) ---")
                metrics = {}
                timing_raw = {}

                if not batch_dict: print("[DEBUG] Empty batch_dict. Skipping."); continue
                batch: DataProto = DataProto.from_single_dict(batch_dict)
                if len(batch.batch) == 0 and len(batch.non_tensor_batch) == 0: print("[DEBUG] Empty DataProto. Skipping."); continue

                # Store original prompts before pop
                original_prompts_text = batch.non_tensor_batch.get("prompts")
                original_videos = batch.non_tensor_batch.get("videos") # Store original video refs too

                # pop keys for generation
                # ... (popping logic as before) ...
                pop_batch_keys = ['input_ids', 'attention_mask', 'position_ids']
                pop_non_tensor_keys = ['raw_prompt_ids']
                if 'multi_modal_inputs' in batch.non_tensor_batch: pop_non_tensor_keys.extend(['multi_modal_data', 'multi_modal_inputs'])
                actual_pop_batch_keys = [k for k in pop_batch_keys if k in batch.batch]
                actual_pop_non_tensor_keys = [k for k in pop_non_tensor_keys if k in batch.non_tensor_batch]
                gen_batch = batch.pop(batch_keys=actual_pop_batch_keys, non_tensor_batch_keys=actual_pop_non_tensor_keys)


                is_last_step = self.global_steps >= self.total_training_steps

                with _timer('step', timing_raw):
                    # generate sequences
                    with _timer('gen', timing_raw): gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch)

                    if 'uid' not in batch.non_tensor_batch: # Add UID if not present
                         batch.non_tensor_batch['uid'] = np.array([str(uuid.uuid4()) for _ in range(len(batch.batch))], dtype=object)
                    print(f"[DEBUG] Repeating batch {self.config.actor_rollout_ref.rollout.n} times...") # DEBUG
                    batch = batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
                    print(f"[DEBUG] Batch size after repeat: {len(batch.batch)}") # DEBUG
                    batch = batch.union(gen_batch_output)
                    print("[DEBUG] Merged gen_batch_output. Batch keys:", list(batch.batch.keys())) # DEBUG

                    # --- Store and Log Sample ---
                    if len(batch.batch) > 0:
                        try:
                            prompt_text_to_print = original_prompts_text[0] if original_prompts_text is not None and len(original_prompts_text) > 0 else "N/A"
                            video_ref_to_save = original_videos[0][0] if original_videos is not None and len(original_videos) > 0 and len(original_videos[0]) > 0 else "N/A"
                            response_tokens = batch.batch['responses'][0]
                            decoded_response = self.tokenizer.decode(response_tokens, skip_special_tokens=True)
                            print("-" * 50); 
                            print(f"[DEBUG SAMPLE] Prompt (Index 0): {prompt_text_to_print}"); 
                            print(f"[DEBUG SAMPLE] Output (Index 0): {decoded_response}"); 
                            print("-" * 50)
                            # Append sample to list
                            sample_to_save = { "step": self.global_steps, "prompt": prompt_text_to_print, "response": decoded_response, "video_ref": video_ref_to_save}
                            self.all_train_samples.append(sample_to_save)
                        except Exception as e: print(f"[DEBUG SAMPLE] Error printing/saving sample: {e}")

                    batch.batch['response_mask'] = compute_response_mask(batch)
                    if self.config.trainer.balance_batch: self._balance_batch(batch, metrics=metrics)
                    batch.meta_info['global_token_num'] = torch.sum(batch.batch['attention_mask'], dim=-1).tolist()

                    # --- Log Probs ---
                    with _timer('old_log_prob', timing_raw): old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
                    # ... (entropy calc) ...
                    batch = batch.union(old_log_prob)
                    if self.use_reference_policy:
                        with _timer('ref', timing_raw): ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
                        batch = batch.union(ref_log_prob)

                    # --- Compute Values ---
                    if self.use_critic:
                        with _timer('values', timing_raw): values = self.critic_wg.compute_values(batch)
                        batch = batch.union(values)

                    # --- Advantage Computation ---
                    with _timer('adv', timing_raw):
                        # ... (RM logic) ...
                        try:
                            reward_result = self.reward_fn(batch, return_dict=True)
                            reward_tensor = reward_result['reward_tensor']
                            reward_extra_infos_dict = reward_result.get('reward_extra_info', {})
                        except Exception as e: print(f'Error in reward_fn: {e}'); import traceback; traceback.print_exc(); raise e
                        batch.batch['token_level_scores'] = reward_tensor
                        if reward_extra_infos_dict: batch.non_tensor_batch.update({k: np.array(v) for k, v in reward_extra_infos_dict.items()})

                        # --- Log Extra Reward Metrics ---
                        if reward_extra_infos_dict:
                            for key, val_array in reward_extra_infos_dict.items():
                                if isinstance(val_array, (list, np.ndarray)) and len(val_array) > 0:
                                     if isinstance(val_array, np.ndarray): mean_val = np.mean([v for v in val_array if np.isfinite(v)]) if len(val_array)>0 else 0
                                     else: mean_val = np.mean([v for v in val_array if isinstance(v, (int, float)) and np.isfinite(v)]) if len(val_array)>0 else 0
                                     metrics[f'reward/{key}_mean'] = mean_val
                                else: metrics[f'reward/{key}_mean'] = 0
                        # --- End Log Extra Metrics ---

                        if self.config.algorithm.use_kl_in_reward: # ... (KL penalty) ...
                            batch, kl_metrics = apply_kl_penalty(batch, kl_ctrl=self.kl_ctrl_in_reward, kl_penalty=self.config.algorithm.kl_penalty); metrics.update(kl_metrics)
                        else: batch.batch['token_level_rewards'] = batch.batch['token_level_scores']

                        batch = compute_advantage(batch, adv_estimator=self.config.algorithm.adv_estimator, gamma=self.config.algorithm.gamma, lam=self.config.algorithm.lam, num_repeat=self.config.actor_rollout_ref.rollout.n)

                    # --- Update Critic ---
                    if self.use_critic:
                        with _timer('update_critic', timing_raw): critic_output = self.critic_wg.update_critic(batch)
                        critic_output_metrics = reduce_metrics(critic_output.meta_info['metrics']); metrics.update(critic_output_metrics)

                    # --- Update Actor ---
                    if self.config.trainer.critic_warmup <= self.global_steps:
                        # --- WORKAROUND START (Optional) ---
                        problematic_keys = ['tvg_accuracy', 'tvg_format', 'tvg_combined']
                        # print(f"[DEBUG] WORKAROUND: Removing problematic non-tensor keys before update_actor: {problematic_keys}")
                        for key in problematic_keys:
                            if key in batch.non_tensor_batch: del batch.non_tensor_batch[key]
                        # --- WORKAROUND END ---
                        with _timer('update_actor', timing_raw): actor_output = self.actor_rollout_wg.update_actor(batch)
                        actor_output_metrics = reduce_metrics(actor_output.meta_info['metrics']); metrics.update(actor_output_metrics)

                    # --- Validate ---
                    if self.val_reward_fn is not None and self.config.trainer.test_freq > 0 and \
                        (is_last_step or self.global_steps % self.config.trainer.test_freq == 0):
                        with _timer('testing', timing_raw): val_metrics: dict = self._validate()
                        if is_last_step: last_val_metrics = val_metrics
                        metrics.update(val_metrics)

                    # --- Save Checkpoint & Training Samples ---
                    if self.config.trainer.save_freq > 0 and ( is_last_step or \
                            self.global_steps % self.config.trainer.save_freq == 0):
                        print(f"[DEBUG] Saving checkpoint for step {self.global_steps}...")
                        with _timer('save_checkpoint', timing_raw): self._save_checkpoint()
                        print("[DEBUG] Checkpoint saved.")

                        # --- MODIFICATION: Save training samples periodically ---
                        try:
                            train_output_dir = os.path.join(self.config.trainer.default_local_dir, "training_samples")
                            os.makedirs(train_output_dir, exist_ok=True)
                            # Save samples collected *since last save*
                            train_samples_file_path = os.path.join(train_output_dir, f"train_samples_step_{self.global_steps}.jsonl")
                            print(f"[DEBUG] Saving {len(self.all_train_samples)} training samples to: {train_samples_file_path}")
                            with open(train_samples_file_path, 'w') as f:
                                for sample in self.all_train_samples:
                                     f.write(json.dumps(sample) + '\n')
                            self.all_train_samples.clear() # <<< Clear list after saving
                            print("[DEBUG] Training samples saved and list cleared.")
                        except Exception as e:
                             print(f"Error saving training samples at step {self.global_steps}: {e}")
                        # --- END MODIFICATION ---


                # --- Logging ---
                # ... (logging metrics code as before) ...
                metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
                metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
                n_gpus = self.resource_pool_manager.get_n_gpus()
                metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, n_gpus=n_gpus))
                logger.log(data=metrics, step=self.global_steps)
                print(f"[DEBUG] Step {self.global_steps} Metrics Logged.")


                if is_last_step:
                    pprint(f'Final validation metrics: {last_val_metrics}'); progress_bar.close()
                    print("--- Reached Last Step, Exiting Fit ---"); break # Break inner loop
                    # with _timer('save_checkpoint', timing_raw): self._save_checkpoint()

                progress_bar.update(1)
                self.global_steps += 1
            # End of batch loop

        print("--- Finished Training Loop ---") # DEBUG
    
    def fit_vqa(self):
        """
        Two-Stage VQA Training Loop.
        - Uses self.two_stage_dataloader.
        - Manually assembles inputs for Stage 1 (Grounding) and Stage 2 (QA).
        - Manually assembles the PPO update batch using outputs from Stage 2.
        - Incorporates VQACombinedRewardManager for reward calculation.
        - Includes verbose PPO preparation logging.
        """
        from verl.utils.tracking import Tracking
        from omegaconf import OmegaConf
        logger_backend = Tracking(project_name=self.config.trainer.project_name,
                                  experiment_name=self.config.trainer.experiment_name,
                                  default_backend=self.config.trainer.logger,
                                  config=OmegaConf.to_container(self.config, resolve=True))

        self.global_steps = 0
        print("--- Initializing Fit VQA ---")

        print("--- Loading Checkpoint ---")
        self.global_steps = self._load_checkpoint() # Ensure this method returns the loaded step
        print(f"--- Checkpoint Loaded, Starting from Global Step: {self.global_steps} ---") # Start from current step

        # Initial validation
        if self.val_reward_fn is not None and self.config.trainer.get('val_before_train', True):
             print("--- Performing Initial Validation ---")
             val_metrics = self._validate_vqa() # Make sure _validate_vqa uses self.val_reward_fn
             print(f'Initial validation metrics: {val_metrics}')
             logger_backend.log(data=val_metrics, step=self.global_steps) # Log with current global_steps
             if self.config.trainer.get('val_only', False):
                 print("--- val_only=True, exiting fit_vqa ---"); return

        progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="VQA Training Progress")
        is_last_step_reached = False # Flag to break outer loop

        print("--- Starting VQA Training Loop ---")
        
        # Determine start epoch based on loaded global_steps
        # Add a check for len(self.two_stage_dataloader) to avoid division by zero if empty
        start_epoch = 0
        if len(self.two_stage_dataloader) > 0:
            start_epoch = self.global_steps // len(self.two_stage_dataloader)
        else:
            print("[WARN] Two-stage dataloader is empty. Cannot determine start epoch or run training.")
            progress_bar.close()
            return

        for epoch in range(start_epoch, self.config.trainer.total_epochs):
            print(f"--- Starting Epoch {epoch+1}/{self.config.trainer.total_epochs} (Global Step: {self.global_steps}) ---")
            if hasattr(self.two_stage_dataloader.sampler, 'set_epoch'): 
                 self.two_stage_dataloader.sampler.set_epoch(epoch) # Sampler epoch for distributed training

            for i, batch_dict_from_loader in enumerate(self.two_stage_dataloader):
                # Skip steps if resuming from a checkpoint mid-epoch
                # current_absolute_step is the step number if we started from 0
                current_absolute_step_in_epoch_calc = epoch * len(self.two_stage_dataloader) + i
                if current_absolute_step_in_epoch_calc < self.global_steps:
                    continue
                
                self.global_steps += 1 # Increment global_steps AT THE START of processing a new batch

                if self.global_steps > self.total_training_steps: 
                    is_last_step_reached = True; break # Break inner (batch) loop

                step_info_prefix = f"[VQA Fit - Step {self.global_steps}/{self.total_training_steps}]"
                print(f"\n{step_info_prefix} --- Starting Iteration (Epoch {epoch+1}, Batch {i+1}) ---")
                metrics = {}; timing_raw = {}

                # --- Load and Validate Batch from Dataloader ---
                if not batch_dict_from_loader: 
                    print(f"{step_info_prefix} [WARN] Empty batch_dict from dataloader. Skipping."); 
                    if self.global_steps > 0: self.global_steps -=1 # Decrement if we skip before processing
                    continue
                try:
                    # This is the original, full batch from the dataloader (contains S1 & S2 components)
                    full_batch_proto_from_loader = DataProto.from_single_dict(batch_dict_from_loader)
                    if not full_batch_proto_from_loader or len(full_batch_proto_from_loader) == 0: 
                        print(f"{step_info_prefix} [WARN] Empty DataProto from dataloader. Skipping."); 
                        if self.global_steps > 0: self.global_steps -=1
                        continue
                    current_batch_size = len(full_batch_proto_from_loader)
                    # Add UIDs if not present
                    if 'uid' not in full_batch_proto_from_loader.non_tensor_batch: 
                        full_batch_proto_from_loader.non_tensor_batch['uid'] = np.array([str(uuid.uuid4()) for _ in range(current_batch_size)], dtype=object)
                    print(f"{step_info_prefix} [INFO] Loaded Original Batch - Size: {current_batch_size}")
                except Exception as e: 
                    print(f"{step_info_prefix} [ERROR] DataProto creation from dataloader batch: {e}. Skipping."); 
                    import traceback; traceback.print_exc(); 
                    if self.global_steps > 0: self.global_steps -=1
                    continue
                
                # ================== Start of Core PPO Step ==================
                with _timer('full_step_wall_time', timing_raw): # Overall timer for the step
                    # --- Stage 1: Grounding Generation ---
                    print(f"{step_info_prefix} -- Stage 1: Generating Grounding --")
                    decoded_grounding_texts_s1 = ["<S1 Gen Fail>"] * current_batch_size
                    predicted_times_s1 = [(None, None)] * current_batch_size
                    s1_gen_output_proto = None # Define to check success later

                    try:
                        with _timer('S1_Assemble_Gen_Parse', timing_raw):
                            # Prepare Stage 1 input from `full_batch_proto_from_loader`
                            s1_input_batch_keys = ['input_ids', 'attention_mask', 'position_ids']
                            s1_input_non_tensor_keys = ['multi_modal_inputs'] 
                            
                            actual_s1_input_batch_keys = [k for k in s1_input_batch_keys if k in full_batch_proto_from_loader.batch]
                            actual_s1_input_non_tensor_keys = [k for k in s1_input_non_tensor_keys if k in full_batch_proto_from_loader.non_tensor_batch]

                            if not actual_s1_input_batch_keys: raise ValueError("Missing essential Stage 1 tensor keys in `full_batch_proto_from_loader`.")

                            s1_input_proto_for_gen = DataProto(
                               batch=full_batch_proto_from_loader.batch.select(*actual_s1_input_batch_keys),
                               non_tensor_batch={k: full_batch_proto_from_loader.non_tensor_batch[k] for k in actual_s1_input_non_tensor_keys}
                            )
                            s1_input_proto_for_gen.meta_info = { 
                                'eos_token_id': self.tokenizer.eos_token_id, 'pad_token_id': self.tokenizer.pad_token_id,
                                'max_new_tokens': self.config.actor_rollout_ref.rollout.get("max_new_tokens_grounding", 50),
                                'do_sample': self.config.actor_rollout_ref.rollout.get("do_sample_grounding", True),
                                'temperature': self.config.actor_rollout_ref.rollout.get("temperature_grounding", 0.7),
                            }
                            # Pad for distributed generation if needed
                            s1_input_padded_proto, s1_pad_size = pad_dataproto_to_divisor(s1_input_proto_for_gen, self.actor_rollout_wg.world_size)
                            
                            # Generate grounding sequences
                            s1_output_padded_proto = self.actor_rollout_wg.generate_sequences(s1_input_padded_proto)
                            s1_gen_output_proto = unpad_dataproto(s1_output_padded_proto, pad_size=s1_pad_size)

                            s1_responses_tensor = s1_gen_output_proto.batch.get('responses')
                            if s1_responses_tensor is None: raise ValueError("Stage 1 generation failed (no 'responses' tensor in output).")
                            
                            decoded_grounding_texts_s1 = self.tokenizer.batch_decode(s1_responses_tensor, skip_special_tokens=True)
                            predicted_times_s1 = parse_grounding_times(decoded_grounding_texts_s1) # Ensure this function is robust
                    except Exception as e: 
                        print(f"{step_info_prefix} [ERROR] During Stage 1 (Grounding): {e}. Skipping PPO for this batch."); 
                        import traceback; traceback.print_exc(); progress_bar.update(1); 
                        if self.global_steps > 0: self.global_steps -=1 # Decrement as PPO step is skipped
                        continue # Skip to next item in dataloader

                    # --- Stage 2: Prepare Inputs for QA ---
                    print(f"{step_info_prefix} -- Stage 2: Preparing Inputs for QA --")
                    s2_items_for_collation = [] 
                    s2_valid_original_indices = [] # Indices from `full_batch_proto_from_loader` that are valid for S2
                    gt_times_used_count = 0
                    with _timer('S2_Input_Prep', timing_raw):
                        for original_idx in range(current_batch_size):
                            try:
                                pred_start_s1, pred_end_s1 = predicted_times_s1[original_idx]
                                gt_start_s1 = full_batch_proto_from_loader.non_tensor_batch['start_time'][original_idx]
                                gt_end_s1 = full_batch_proto_from_loader.non_tensor_batch['end_time'][original_idx]
                                
                                # Determine final start/end times for video clipping
                                final_clip_start, final_clip_end = pred_start_s1, pred_end_s1
                                clip_source_log = "PredictedS1"
                                if final_clip_start is None or final_clip_end is None or \
                                   not isinstance(final_clip_start, (int, float)) or \
                                   not isinstance(final_clip_end, (int, float)) or \
                                   final_clip_start < 0 or final_clip_start >= final_clip_end:
                                    final_clip_start, final_clip_end = gt_start_s1, gt_end_s1
                                    clip_source_log = "GT_Times"
                                    gt_times_used_count +=1
                                    if final_clip_start is None or final_clip_end is None or \
                                       not isinstance(final_clip_start, (int, float)) or \
                                       not isinstance(final_clip_end, (int, float)) or \
                                       final_clip_start < 0 or final_clip_start >= final_clip_end:
                                        final_clip_start, final_clip_end = 0.0, None # Fallback to full video
                                        clip_source_log = "FullVideoFallback"
                                
                                  
                                print(f"[Prepare Inputs for QA] {original_idx}: clip start and end: {final_clip_start}, {final_clip_end}")

                                video_path_s2 = full_batch_proto_from_loader.non_tensor_batch["video_path"][original_idx]
                                video_proc_config_s2 = full_batch_proto_from_loader.non_tensor_batch["video_processing_config"][original_idx]
                                
                                # Ensure video_proc_config_s2 has either 'nframes' or 'fps', but not both for fetch_video's smart_nframes
                                current_clip_video_proc_config = video_proc_config_s2.copy()
                                if current_clip_video_proc_config.get("nframes") is not None: 
                                    current_clip_video_proc_config.pop("fps", None)
                                else: 
                                    current_clip_video_proc_config.pop("nframes", None)
                                    current_clip_video_proc_config.setdefault("fps", 2) # Default FPS if nframes not used

                                ele_segment_s2 = {"video": video_path_s2, 
                                                  "video_start": final_clip_start, 
                                                  "video_end": final_clip_end, 
                                                  "original_idx": original_idx,
                                                  **current_clip_video_proc_config}
                                clipped_video_frames_s2 = fetch_video(ele_segment_s2, image_factor=current_clip_video_proc_config.get("image_factor", 28))
                                if clipped_video_frames_s2 is None or clipped_video_frames_s2.nelement() == 0:
                                    raise ValueError(f"Clipped video for item {original_idx} (source: {clip_source_log}) resulted in empty tensor.")

                                question_text_s2 = full_batch_proto_from_loader.non_tensor_batch["question_text"][original_idx]
                                
                                # `prepare_stage2_inputs_for_item` is a helper you need to define/import
                                # It takes item_data (question, clipped_video), processor, tokenizer, config
                                # and returns a dict with 'input_ids', 'attention_mask', 'position_ids', 'multi_modal_inputs'
                                item_data_for_s2_processor = {
                                    "question_text": question_text_s2, 
                                    "clipped_video": clipped_video_frames_s2, 
                                    "original_index": original_idx # For debugging/tracking
                                }
                                processed_s2_item_dict = prepare_stage2_inputs_for_item(item_data_for_s2_processor, self.processor, self.tokenizer, self.config)
                                
                                if processed_s2_item_dict:
                                    s2_items_for_collation.append(processed_s2_item_dict)
                                    s2_valid_original_indices.append(original_idx)
                            except Exception as e: 
                                print(f"{step_info_prefix} [WARN] S2 Input Prep for original_idx {original_idx}: {e}. Item will be skipped for PPO.")
                    
                    metrics['stage2_prep/gt_times_used_rate'] = gt_times_used_count / current_batch_size if current_batch_size > 0 else 0
                    metrics['stage2_prep/valid_item_for_s2_gen_rate'] = len(s2_valid_original_indices) / current_batch_size if current_batch_size > 0 else 0
                    
                    # --- Stage 2: QA Generation ---
                    print(f"{step_info_prefix} -- Stage 2: Generating QA --")
                    s2_qa_gen_output_proto = None # Define to check success later
                    s2_input_proto_for_qa_gen = None # Define for PPO assembly

                    if s2_items_for_collation:
                        try:
                            with _timer('S2_Collate_Gen_Decode', timing_raw):
                                collated_s2_qa_gen_input_dict = collate_fn(s2_items_for_collation) # Use your collate_fn
                                s2_input_proto_for_qa_gen = DataProto.from_single_dict(collated_s2_qa_gen_input_dict)
                                
                                s2_input_proto_for_qa_gen.meta_info = {
                                    'eos_token_id': self.tokenizer.eos_token_id, 'pad_token_id': self.tokenizer.pad_token_id,
                                    'max_new_tokens': self.config.actor_rollout_ref.rollout.get("max_new_tokens_qa", 10), # Short for QA letter
                                    'do_sample': self.config.actor_rollout_ref.rollout.get("do_sample_qa", True),
                                    'temperature': self.config.actor_rollout_ref.rollout.get("temperature_qa", 0.7),
                                }
                                # Pad for distributed generation
                                s2_qa_gen_input_padded_proto, s2_qa_gen_pad_size = pad_dataproto_to_divisor(s2_input_proto_for_qa_gen, self.actor_rollout_wg.world_size)
                                
                                s2_qa_gen_output_padded_proto = self.actor_rollout_wg.generate_sequences(s2_qa_gen_input_padded_proto)
                                s2_qa_gen_output_proto = unpad_dataproto(s2_qa_gen_output_padded_proto, pad_size=s2_qa_gen_pad_size)

                                if 'responses' not in s2_qa_gen_output_proto.batch:
                                    raise ValueError("Stage 2 QA generation failed (no 'responses' tensor in output).")
                        except Exception as e: 
                            print(f"{step_info_prefix} [ERROR] During Stage 2 (QA Generation): {e}. Skipping PPO for this batch."); 
                            import traceback; traceback.print_exc(); progress_bar.update(1); 
                            if self.global_steps > 0: self.global_steps -=1
                            continue
                    else:
                        print(f"{step_info_prefix} [INFO] No valid items prepped for Stage 2 QA generation. Skipping PPO for this batch.")
                        progress_bar.update(1); 
                        if self.global_steps > 0: self.global_steps -=1
                        continue # Skip PPO if no S2 items

                    # --- PPO Update Phase ---
                    # This `ppo_batch_for_actor_update` will be constructed for PPO steps.
                    # It's often just called `batch` inside typical PPO loops.
                    ppo_batch_for_actor_update = None 

                    # Only proceed if Stage 1 AND Stage 2 generation were successful AND we have valid items.
                    if s1_gen_output_proto and s2_input_proto_for_qa_gen and s2_qa_gen_output_proto and \
                       'responses' in s2_qa_gen_output_proto.batch and s2_valid_original_indices:
                        print(f"{step_info_prefix} -- PPO: Assembling Batch for Actor Update --")
                        try:
                            with _timer('PPO_Batch_Assembly', timing_raw):
                                # --- 1. Key components for PPO batch ---
                                #    From S2 Input (s2_input_proto_for_qa_gen):
                                #      - s2_input_proto_for_qa_gen.batch['input_ids'] (S2 prompt tokens)
                                #      - s2_input_proto_for_qa_gen.batch['attention_mask'] (S2 prompt mask)
                                #      - s2_input_proto_for_qa_gen.batch['position_ids'] (S2 prompt pos_ids, RoPEd for S2 video)
                                #      - s2_input_proto_for_qa_gen.non_tensor_batch['multi_modal_inputs'] (S2 visual features as np.array of dicts)
                                #    From S2 Output (s2_qa_gen_output_proto):
                                #      - s2_qa_gen_output_proto.batch['responses'] (S2 generated QA answer tokens - these are PPO "actions")
                                #      - Potentially, s2_qa_gen_output_proto.batch may contain 'input_ids', 'attention_mask', 'position_ids' for the *full* (S2_prompt + S2_action) sequence.
                                #    From S1 Output and Original Batch (full_batch_proto_from_loader):
                                #      - decoded_grounding_texts_s1 (S1 decoded grounding text)
                                #      - Ground truth start/end times and QA answers (for reward function)
                                #      - UIDs

                                print(f"{step_info_prefix} [PPO Assembly VERBOSE] Using s2_input_proto_for_qa_gen (Input to S2 Gen) and s2_qa_gen_output_proto (Output from S2 Gen).")
                                print(f"{step_info_prefix} [PPO Assembly VERBOSE]   s2_input_proto_for_qa_gen.batch keys: {list(s2_input_proto_for_qa_gen.batch.keys())}")
                                print(f"{step_info_prefix} [PPO Assembly VERBOSE]   s2_input_proto_for_qa_gen.non_tensor_batch keys: {list(s2_input_proto_for_qa_gen.non_tensor_batch.keys())}")
                                print(f"{step_info_prefix} [PPO Assembly VERBOSE]   s2_qa_gen_output_proto.batch keys: {list(s2_qa_gen_output_proto.batch.keys())}")

                                # Component: S2 Prompt Tokens (what the model saw before generating QA)
                                ppo_prompt_tokens = s2_input_proto_for_qa_gen.batch['input_ids']
                                ppo_prompt_mask = s2_input_proto_for_qa_gen.batch['attention_mask']
                                # ppo_prompt_pos_ids = s2_input_proto_for_qa_gen.batch['position_ids'] # We'll recalculate/get full later

                                # Component: S2 Visual Features (visual context for S2 prompt)
                                # This should be an np.array of dicts, each dict holding TENSOR features for one item.
                                ppo_prompt_visual_features_list_of_dicts = s2_input_proto_for_qa_gen.non_tensor_batch['multi_modal_inputs']
                                print(f"{step_info_prefix} [PPO Assembly VERBOSE]   ppo_prompt_visual_features_list_of_dicts type: {type(ppo_prompt_visual_features_list_of_dicts)}, length: {len(ppo_prompt_visual_features_list_of_dicts) if hasattr(ppo_prompt_visual_features_list_of_dicts, '__len__') else 'N/A'}")
                                if len(ppo_prompt_visual_features_list_of_dicts) > 0:
                                    print(f"{step_info_prefix} [PPO Assembly VERBOSE]     First item type: {type(ppo_prompt_visual_features_list_of_dicts[0])}, keys: {list(ppo_prompt_visual_features_list_of_dicts[0].keys()) if isinstance(ppo_prompt_visual_features_list_of_dicts[0], dict) else 'N/A'}")

                                # Component: S2 Actions (what the model generated for QA)
                                ppo_action_tokens = s2_qa_gen_output_proto.batch['responses']
                                print(f"{step_info_prefix} [PPO Assembly VERBOSE]   ppo_prompt_tokens shape: {ppo_prompt_tokens.shape}, ppo_action_tokens shape: {ppo_action_tokens.shape}")


                                # --- Construct FULL SEQUENCE tensors for PPO (S2_prompt + S2_actions) ---
                                # These need to be LEFT-PADDED for the PPO actor.
                                ppo_full_input_ids = s2_qa_gen_output_proto.batch.get('input_ids')
                                ppo_full_attention_mask = s2_qa_gen_output_proto.batch.get('attention_mask')
                                ppo_full_position_ids = s2_qa_gen_output_proto.batch.get('position_ids')

                                is_full_sequence_directly_available = (
                                    ppo_full_input_ids is not None and
                                    ppo_full_attention_mask is not None and
                                    ppo_full_position_ids is not None and
                                    ppo_full_input_ids.shape[1] == (ppo_prompt_tokens.shape[1] + ppo_action_tokens.shape[1])
                                )

                                if is_full_sequence_directly_available:
                                    print(f"{step_info_prefix} [PPO Assembly VERBOSE]   Using FULL sequence tensors directly from S2 generation output.")
                                else:
                                    print(f"{step_info_prefix} [PPO Assembly VERBOSE]   Manually constructing FULL sequence tensors for PPO.")
                                    # Mask for the S2 generated actions (handles potential internal padding in actions)
                                    ppo_action_internal_mask = (ppo_action_tokens != self.tokenizer.pad_token_id).long()

                                    temp_concat_tokens = torch.cat([ppo_prompt_tokens, ppo_action_tokens], dim=1)
                                    temp_concat_mask = torch.cat([ppo_prompt_mask, ppo_action_internal_mask], dim=1)
                                    
                                    # Define max length for the PPO actor's input sequence
                                    ppo_actor_max_seq_len = self.config.actor_rollout_ref.actor.get("max_sequence_length", 
                                                                                        self.config.data.get("max_prompt_length", 1024) + \
                                                                                        self.config.actor_rollout_ref.rollout.get("max_new_tokens_qa", 50) + \
                                                                                        50) # Buffer

                                    ppo_full_input_ids, ppo_full_attention_mask = verl_F.postprocess_data(
                                        input_ids=temp_concat_tokens,
                                        attention_mask=temp_concat_mask,
                                        max_length=ppo_actor_max_seq_len,
                                        pad_token_id=self.tokenizer.pad_token_id,
                                        left_pad=True,  # Critical: PPO actor expects left-padded combined sequence
                                        truncation='right' 
                                    )
                                    # Fallback for position_ids if not provided by generation output.
                                    # For RoPE, this needs to be handled carefully by the actor's forward pass or
                                    # by a more sophisticated re-calculation here.
                                    ppo_full_position_ids = compute_position_id_with_mask(ppo_full_attention_mask)
                                    print(f"{step_info_prefix} [PPO Assembly VERBOSE]     Manually created: ppo_full_input_ids shape: {ppo_full_input_ids.shape}, ppo_full_attention_mask shape: {ppo_full_attention_mask.shape}, ppo_full_position_ids shape: {ppo_full_position_ids.shape}")


                                # --- Mask for the "actions" part (S2 QA response) for PPO loss ---
                                # This mask corresponds to `ppo_action_tokens` and is used in PPO loss calculations.
                                ppo_mask_for_actions_in_loss = (ppo_action_tokens != self.tokenizer.pad_token_id).long()
                                print(f"{step_info_prefix} [PPO Assembly VERBOSE]   ppo_mask_for_actions_in_loss shape: {ppo_mask_for_actions_in_loss.shape} (should match ppo_action_tokens)")


                                # --- Assemble PPO Batch TENSOR Components ---
                                ppo_tensor_data_for_actor = {
                                    "input_ids": ppo_full_input_ids,
                                    "attention_mask": ppo_full_attention_mask,
                                    "position_ids": ppo_full_position_ids,
                                    "responses": ppo_action_tokens, # These are the PPO "actions"
                                    "response_mask": ppo_mask_for_actions_in_loss, # Mask for the "actions"
                                }

                                # --- Assemble PPO NON-TENSOR Components ---
                                ppo_non_tensor_data_for_actor = {}
                                # Assign the S2 visual features (list of dicts of tensors)
                                if isinstance(ppo_prompt_visual_features_list_of_dicts, np.ndarray) and \
                                   ppo_prompt_visual_features_list_of_dicts.dtype == object and \
                                   len(ppo_prompt_visual_features_list_of_dicts) > 0 and \
                                   isinstance(ppo_prompt_visual_features_list_of_dicts[0], dict):
                                    ppo_non_tensor_data_for_actor['multi_modal_inputs'] = ppo_prompt_visual_features_list_of_dicts
                                    print(f"{step_info_prefix} [PPO Assembly VERBOSE]   Assigned 'multi_modal_inputs' to PPO non_tensor_batch.")
                                else:
                                    print(f"{step_info_prefix} [PPO Assembly WARN]   `ppo_prompt_visual_features_list_of_dicts` has unexpected type/structure. PPO actor might miss visual input.")
                                    ppo_non_tensor_data_for_actor['multi_modal_inputs'] = np.array([], dtype=object)

                                # Add data needed by VQACombinedRewardManager
                                reward_fn_data_temp = defaultdict(list)
                                for original_idx_ppo in s2_valid_original_indices: # Iterate over indices that made it to S2
                                    reward_fn_data_temp['decoded_grounding_texts'].append(decoded_grounding_texts_s1[original_idx_ppo])
                                    reward_fn_data_temp['start_time'].append(full_batch_proto_from_loader.non_tensor_batch['start_time'][original_idx_ppo])
                                    reward_fn_data_temp['end_time'].append(full_batch_proto_from_loader.non_tensor_batch['end_time'][original_idx_ppo])
                                    reward_fn_data_temp['ground_truth_answer'].append(full_batch_proto_from_loader.non_tensor_batch['ground_truth_answer'][original_idx_ppo])
                                for key_val, list_val in reward_fn_data_temp.items():
                                    ppo_non_tensor_data_for_actor[key_val] = np.array(list_val, dtype=object)
                                
                                ppo_uids_for_valid_items = [full_batch_proto_from_loader.non_tensor_batch['uid'][idx_val_ppo] for idx_val_ppo in s2_valid_original_indices]
                                ppo_non_tensor_data_for_actor['uid'] = np.array(ppo_uids_for_valid_items, dtype=object)
                                print(f"{step_info_prefix} [PPO Assembly VERBOSE]   Added reward-specific non-tensor data (UIDs, GTs, S1 texts).")

                                # --- Create the final DataProto for PPO updates ---
                                # This is the 'batch' variable that PPO core algorithms expect.
                                ppo_batch_for_actor_update = DataProto( 
                                    batch=TensorDict(ppo_tensor_data_for_actor, batch_size=[len(s2_valid_original_indices)]),
                                    non_tensor_batch=ppo_non_tensor_data_for_actor
                                )
                                print(f"{step_info_prefix} [PPO Assembly INFO] Successfully assembled `ppo_batch_for_actor_update` with {len(s2_valid_original_indices)} items.")
                                print(f"{step_info_prefix} [PPO Assembly INFO]   Tensor keys: {list(ppo_batch_for_actor_update.batch.keys())}")
                                print(f"{step_info_prefix} [PPO Assembly INFO]   Non-tensor keys: {list(ppo_batch_for_actor_update.non_tensor_batch.keys())}")
                                if 'multi_modal_inputs' in ppo_batch_for_actor_update.non_tensor_batch and len(ppo_batch_for_actor_update.non_tensor_batch['multi_modal_inputs']) > 0:
                                    print(f"{step_info_prefix} [PPO Assembly INFO]     MM Inputs [0] keys: {list(ppo_batch_for_actor_update.non_tensor_batch['multi_modal_inputs'][0].keys()) if isinstance(ppo_batch_for_actor_update.non_tensor_batch['multi_modal_inputs'][0], dict) else 'Not a dict'}")

                        except Exception as ppo_assembly_err:
                            print(f"{step_info_prefix} [ERROR] PPO Batch Assembly Failed: {ppo_assembly_err}");
                            import traceback; traceback.print_exc(); ppo_batch_for_actor_update = None
                    
                    # --- PPO Update Steps (Reward, LogProbs, Values, Advantage, Actor/Critic Updates) ---
                    if ppo_batch_for_actor_update: # Only if assembly was successful
                        print(f"{step_info_prefix} -- PPO: Performing Core Update Steps --")
                        try:
                            # This is the `batch` that all subsequent PPO functions will use.
                            # For clarity in this section, let's use `ppo_batch_for_actor_update` directly.

                            # === Reward Calculation (uses VQACombinedRewardManager) ===
                            with _timer('PPO_Reward_Calc', timing_raw):
                                reward_result_dict = self.reward_fn(ppo_batch_for_actor_update, return_dict=True)
                                ppo_token_level_scores = reward_result_dict['reward_tensor'] # Shape: [N_valid_s2_items, S2_action_len]
                                ppo_reward_extra_info = reward_result_dict.get('reward_extra_info', {})
                            ppo_batch_for_actor_update.batch['token_level_scores'] = ppo_token_level_scores
                            ppo_batch_for_actor_update.non_tensor_batch.update({f"reward_{k}": v for k,v in ppo_reward_extra_info.items()}) # Prefix for distinct logging
                            print(f"{step_info_prefix} [PPO Update DEBUG]   `token_level_scores` shape: {ppo_batch_for_actor_update.batch['token_level_scores'].shape}")

                            # === Compute Log Probs (Current Actor and Reference Policy) ===
                            # Actor's `compute_log_prob` expects:
                            # - batch['input_ids'] (S2_prompt + S2_action)
                            # - batch['attention_mask'], batch['position_ids'] (for the full sequence)
                            # - batch['responses'] (S2_action tokens, used as labels for logprob calculation)
                            # - non_tensor_batch['multi_modal_inputs'] (S2 visual features for prompt)
                            with _timer('PPO_Compute_Old_LogProbs', timing_raw):
                                old_log_probs_proto = self.actor_rollout_wg.compute_log_prob(ppo_batch_for_actor_update) 
                            ppo_batch_for_actor_update.batch['old_log_probs'] = old_log_probs_proto.batch['old_log_probs'] # Shape: [N_valid, S2_action_len]
                            if 'entropys' in old_log_probs_proto.batch: # If PPO actor calculates entropy
                                ppo_batch_for_actor_update.batch['entropys'] = old_log_probs_proto.batch['entropys']
                            print(f"{step_info_prefix} [PPO Update DEBUG]   `old_log_probs` shape: {ppo_batch_for_actor_update.batch['old_log_probs'].shape}")

                            if self.use_reference_policy:
                                with _timer('PPO_Compute_Ref_LogProbs', timing_raw):
                                    ref_log_probs_proto = self.ref_policy_wg.compute_ref_log_prob(ppo_batch_for_actor_update)
                                ppo_batch_for_actor_update.batch['ref_log_prob'] = ref_log_probs_proto.batch['ref_log_prob']

                            # === Compute Values (Critic) ===
                            if self.use_critic:
                                with _timer('PPO_Compute_Values', timing_raw):
                                    values_proto = self.critic_wg.compute_values(ppo_batch_for_actor_update)
                                ppo_batch_for_actor_update.batch['values'] = values_proto.batch['values'] # Shape: [N_valid, S2_action_len] (or full seq)

                            # === Apply KL Penalty & Compute Advantage ===
                            with _timer('PPO_KL_Penalty_Advantage', timing_raw):
                                if self.config.algorithm.use_kl_in_reward:
                                    ppo_batch_for_actor_update, ppo_kl_metrics = core_algos.apply_kl_penalty(
                                        ppo_batch_for_actor_update, 
                                        kl_ctrl=self.kl_ctrl_in_reward, 
                                        kl_penalty=self.config.algorithm.kl_penalty
                                    )
                                    metrics.update(ppo_kl_metrics) # Log KL controller metrics
                                else: 
                                    ppo_batch_for_actor_update.batch['token_level_rewards'] = ppo_batch_for_actor_update.batch['token_level_scores']
                                
                                ppo_batch_for_actor_update.meta_info = ppo_batch_for_actor_update.meta_info or {} 
                                # `attention_mask` here is for the full (S2_prompt + S2_action) sequence
                                ppo_batch_for_actor_update.meta_info['global_token_num'] = torch.sum(ppo_batch_for_actor_update.batch['attention_mask'], dim=-1).tolist() 

                                if self.config.trainer.balance_batch: # Balance before advantage if enabled
                                    self._balance_batch(ppo_batch_for_actor_update, metrics=metrics, logging_prefix='ppo_seqlen_balance')
                                
                                ppo_batch_for_actor_update = compute_advantage(
                                    ppo_batch_for_actor_update, 
                                    adv_estimator=self.config.algorithm.adv_estimator,
                                    gamma=self.config.algorithm.gamma, 
                                    lam=self.config.algorithm.lam,
                                ) # Adds 'advantages' and 'returns' to batch

                            # === Update Critic ===
                            if self.use_critic:
                                with _timer('PPO_Update_Critic', timing_raw):
                                    critic_update_output_proto = self.critic_wg.update_critic(ppo_batch_for_actor_update)
                                metrics.update(reduce_metrics({f"critic/{k}": v for k, v in critic_update_output_proto.meta_info['metrics'].items()}))

                            # === Update Actor ===
                            if self.config.trainer.critic_warmup <= self.global_steps:
                                # PPO Actor's `update_policy` expects:
                                # - batch['input_ids'] (S2_prompt + S2_action)
                                # - batch['attention_mask'], batch['position_ids'] (for the full sequence)
                                # - batch['responses'] (S2_action tokens, used as labels in _forward_micro_batch during update)
                                # - batch['response_mask'] (mask for S2_action tokens, used for loss aggregation)
                                # - batch['old_log_probs'], batch['advantages'] (and 'returns' if critic is used for baseline in some algos)
                                # - non_tensor_batch['multi_modal_inputs'] (S2 visual features for the prompt)
                                if ppo_batch_for_actor_update.meta_info is None:
                                    ppo_batch_for_actor_update.meta_info = {}
                                
                                # Add temperature to meta_info for the PPO update step
                                # Use the temperature defined for rollouts/logprob computation for consistency,
                                # or a specific temperature for PPO updates if desired (e.g., 1.0).
                                # self.config.actor_rollout_ref.rollout.temperature is a common place.
                                ppo_update_temperature = self.config.actor_rollout_ref.rollout.get("temperature", 1.0) 
                                ppo_batch_for_actor_update.meta_info['temperature'] = ppo_update_temperature
                                print(f"{step_info_prefix} [PPO Update INFO] Setting temperature in meta_info for actor update: {ppo_update_temperature}")
                                with _timer('PPO_Update_Actor', timing_raw):
                                    actor_update_output_proto = self.actor_rollout_wg.update_actor(ppo_batch_for_actor_update)
                                metrics.update(reduce_metrics({f"actor/{k}": v for k, v in actor_update_output_proto.meta_info['metrics'].items()}))
                            else:
                                print(f"{step_info_prefix} [PPO Update INFO] Skipping actor update (Critic warmup: {self.global_steps}/{self.config.trainer.critic_warmup})")

                        except Exception as ppo_core_steps_err:
                            print(f"{step_info_prefix} [ERROR] During PPO Core Update Steps: {ppo_core_steps_err}");
                            import traceback; traceback.print_exc()
                    else:
                        print(f"{step_info_prefix} [WARN] Skipping PPO core update steps as `ppo_batch_for_actor_update` was not successfully assembled.")
                # ================== End of Core PPO Step ==================

                # --- Logging, Validation, Checkpointing (uses the main `metrics` dict) ---
                print(f"{step_info_prefix} -- Finalizing Step (Logging, Validation, Checkpoints) --")
                try:
                    if ppo_batch_for_actor_update: # Log PPO-related metrics only if updates happened
                        if 'token_level_rewards' in ppo_batch_for_actor_update.batch: # Check as it depends on KL logic
                            metrics.update(compute_data_metrics(batch=ppo_batch_for_actor_update, use_critic=self.use_critic))
                        if 'meta_info' in ppo_batch_for_actor_update and 'global_token_num' in ppo_batch_for_actor_update.meta_info:
                            metrics.update(compute_timing_metrics(batch=ppo_batch_for_actor_update, timing_raw=timing_raw))
                            n_gpus = self.resource_pool_manager.get_n_gpus()
                            metrics.update(compute_throughout_metrics(batch=ppo_batch_for_actor_update, timing_raw=timing_raw, n_gpus=n_gpus))
                        
                        # Log specific reward components (already prefixed with "reward_")
                        reward_metrics_from_ppo_batch = {}
                        for key_nt in ppo_batch_for_actor_update.non_tensor_batch:
                            if key_nt.startswith("reward_"): # e.g., "reward_grounding_accuracy"
                                clean_key_for_log = key_nt[len("reward_"):] # -> "grounding_accuracy"
                                val_array_nt = ppo_batch_for_actor_update.non_tensor_batch[key_nt]
                                if isinstance(val_array_nt, (np.ndarray, list)) and len(val_array_nt) > 0:
                                    numeric_vals_nt = [v_nt for v_nt in val_array_nt if isinstance(v_nt, (int, float)) and np.isfinite(v_nt)]
                                    reward_metrics_from_ppo_batch[f'reward/{clean_key_for_log}_mean'] = np.mean(numeric_vals_nt) if numeric_vals_nt else 0.0
                        metrics.update(reward_metrics_from_ppo_batch)
                    
                    metrics['timing/full_step_wall_time_s'] = timing_raw.get('full_step_wall_time', sum(timing_raw.values())) # Total wall time
                    logger_backend.log(data=metrics, step=self.global_steps)
                    print(f"{step_info_prefix} Metrics Logged. Final metrics snapshot: {pformat(metrics)}")
                except Exception as log_err: 
                    print(f"{step_info_prefix} [ERROR] During Metric Logging: {log_err}"); 
                    import traceback; traceback.print_exc()

                # --- Validation ---
                is_last_step_for_run = (self.global_steps >= self.total_training_steps)
                last_val_run_metrics = None # Store metrics from the validation run on the last step
                if self.val_reward_fn is not None and self.config.trainer.test_freq > 0 and \
                   (is_last_step_for_run or self.global_steps % self.config.trainer.test_freq == 0):
                   with _timer('Validation_Run', timing_raw): # Timer for the validation run
                       print(f"{step_info_prefix} -- Running VQA Validation --")
                       val_run_metrics: dict = self._validate_vqa() # Uses self.val_reward_fn
                       logger_backend.log(data=val_run_metrics, step=self.global_steps) # Log prefixed (e.g. val/vqa/final_score/mean)
                       print(f"{step_info_prefix} VQA Validation Metrics Logged: {pformat(val_run_metrics)}")
                       if is_last_step_for_run: last_val_run_metrics = val_run_metrics
                
                # --- Save Checkpoint ---
                if self.config.trainer.save_freq > 0 and \
                   (is_last_step_for_run or self.global_steps % self.config.trainer.save_freq == 0):
                    with _timer('Checkpoint_Save', timing_raw): # Timer for checkpointing
                        print(f"{step_info_prefix} -- Saving Checkpoint --")
                        self._save_checkpoint() # Saves actor, critic, dataloader state

                progress_bar.update(1) # Increment progress bar for this successfully processed step
                if is_last_step_for_run: 
                    is_last_step_reached = True; break # Break inner (batch) loop

            # --- End of Batch Loop ---
            if is_last_step_reached:
                print(f"--- Reached Target Global Step ({self.global_steps}/{self.total_training_steps}). Ending Training. ---")
                break # Break outer (epoch) loop

        progress_bar.close()
        print("--- Finished VQA Training Loop ---")
        if 'last_val_run_metrics' in locals() and last_val_run_metrics: # Check if defined
             print(f"Final validation metrics from the run at step {self.global_steps}: {pformat(last_val_run_metrics)}")
    