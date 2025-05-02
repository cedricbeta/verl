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
from qwen_vl_utils import fetch_video

from tensordict import TensorDict

WorkerType = Type[Worker]




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
        print(f"Parsing text: {text}") # Add print
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

def prepare_stage2_inputs_for_item(item_data, processor, tokenizer, config):
    # ... (implementation from previous response) ...
    """Prepares tokenized text and combines with video features for one item."""
    question_text = item_data["question_text"]
    clipped_video_frames_tensor = item_data["clipped_video"] # Assumes video is already processed

    qa_system_prompt = "Answer the following multiple-choice question based on the video clip by providing only the letter of the correct option." # Example
    stage2_messages = []
    if qa_system_prompt:
         stage2_messages.append({"role": "system", "content": qa_system_prompt})
    stage2_messages.append({"role": "user", "content": question_text})

    # Ensure processor is available (passed from trainer or accessed via self)
    stage2_raw_prompt = processor.apply_chat_template(stage2_messages, add_generation_prompt=True, tokenize=False)

    stage2_model_inputs = processor(
        text=[stage2_raw_prompt],
        images=None,
        videos=[clipped_video_frames_tensor],
        return_tensors="pt"
    )

    if "input_ids" not in stage2_model_inputs or "attention_mask" not in stage2_model_inputs:
         raise ValueError("Processor output missing keys for Stage 2")

    s2_input_ids_raw = stage2_model_inputs.pop("input_ids")
    s2_attn_mask_raw = stage2_model_inputs.pop("attention_mask")

    # Ensure tokenizer and config are available
    stage2_input_ids, stage2_attention_mask = verl_F.postprocess_data(
         input_ids=s2_input_ids_raw, attention_mask=s2_attn_mask_raw,
         max_length=config.data.get("max_prompt_length", 1024),
         pad_token_id=tokenizer.pad_token_id, left_pad=True,
         truncation=config.data.get("truncation", "error")
     )

    # Calculate Stage 2 Position IDs
    stage2_position_ids = compute_position_id_with_mask(stage2_attention_mask)[0]

    stage2_multi_modal_inputs = dict(stage2_model_inputs) # Other features like pixel_values_videos

    return {
        'input_ids': stage2_input_ids[0],
        'attention_mask': stage2_attention_mask[0],
        'position_ids': stage2_position_ids,
        'multi_modal_inputs': stage2_multi_modal_inputs,
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
            data_files="/home/chendong/video-rl/charades_sta/charades_vqa.jsonl",
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
        
        print("--- Starting Conceptual Two-Stage Training Loop ---")

        is_last_step = False # Define is_last_step outside loop scope

        for epoch in range(self.config.trainer.total_epochs):
            print(f"--- Starting Epoch {epoch+1}/{self.config.trainer.total_epochs} ---")
            # Use the dataloader connected to TwoStageVideoQADataset
            current_dataloader = self.two_stage_dataloader # As per user snippet

            for i, batch_dict in enumerate(current_dataloader):
                print(f"\n--- Starting Global Step {self.global_steps} ---")
                step_info_prefix = f"[Step {self.global_steps}/{self.total_training_steps}]"
                metrics = {}
                timing_raw = {}

                # --- Load and Validate Batch ---
                if not batch_dict: print("[DEBUG] Empty batch_dict. Skipping."); continue # Correct skip
                try:
                    full_batch_proto = DataProto.from_single_dict(batch_dict)
                    if not full_batch_proto or not len(full_batch_proto): print("[DEBUG] Empty DataProto. Skipping."); continue # Correct skip
                    current_batch_size = len(full_batch_proto)
                    print(f"[DEBUG] Loaded Batch - Size: {current_batch_size}")
                    print("[DEBUG] Initial batch keys: B={list(full_batch_proto.batch.keys())} N={list(full_batch_proto.non_tensor_batch.keys())}")
                except Exception as e: print(f"Error DataProto: {e}. Skip."); continue # Correct skip
                
                # print all keys and values in batch
                for key, value in full_batch_proto.batch.items():
                    print(f"Batch Key: {key}, Value: {value}")
                for key, value in full_batch_proto.non_tensor_batch.items():
                    print(f"Non-Tensor Key: {key}, Value: {value}")
                # Check if batch is empty after loading

                # --- Stage 1: Grounding ---
                print(f"[{self.global_steps}] -- Stage 1: Generating Grounding --")
                stage1_gen_output = None
                predicted_times = [(None, None)] * current_batch_size
                decoded_grounding_texts = ["<S1 Generation Failed>"] * current_batch_size

                # Using timer context manager assumed defined in the class or globally
                with _timer('S1_PrepGenParse', timing_raw):
                    try:
                        stage1_batch = full_batch_proto.pop(batch_keys=['input_ids', 'attention_mask', 'position_ids'], non_tensor_batch_keys=['multi_modal_inputs'])
                        # print("[DEBUG] Stage 1 batch prepared for generation (keys):", list(stage1_batch.keys(True,True)))

                        # ** CONCEPTUAL CALL: Generate grounding sequences **
                        stage1_gen_output = self.actor_rollout_wg.generate_sequences(stage1_batch)

                        # Parse Stage 1 Output
                        stage1_responses = stage1_gen_output.batch.get('responses')
                        if stage1_responses is None: raise ValueError("Stage 1 failed (no responses).")
                        print(f"[DEBUG] Stage 1 response tensor shape: {stage1_responses.shape}")
                        decoded_grounding_texts = self.tokenizer.batch_decode(stage1_responses, skip_special_tokens=True)
                        predicted_times = parse_grounding_times(decoded_grounding_texts)
                        print(f"[DEBUG] Stage 1 parsed times (first 5): {predicted_times[:5]}")
                        # self._log_stage1_sample(full_batch_proto, decoded_grounding_texts)

                    except Exception as e:
                        print(f"Error during Stage 1: {e}. Skipping step.")
                        import traceback; traceback.print_exc() # Keep traceback for debug
                        continue # Correctly skip to next iteration

                # --- Stage 2: QA ---
                print(f"[{self.global_steps}] -- Stage 2: Preparing Inputs & Generating QA --")
                stage2_batch_inputs = []  # List to collect inputs for valid items
                valid_original_indices = [] # Keep track of which original items were processed successfully

                with _timer('S2_PrepGen', timing_raw): # Combine prep and generation timing
                    # Loop through results from Stage 1 for the current batch
                    for idx in range(current_batch_size): # current_batch_size is length of original batch
                        try:
                            # 1. Get Inputs from the original full batch
                            start_t, end_t = predicted_times[idx]
                            video_path = full_batch_proto.non_tensor_batch["video_path"][idx]
                            # Retrieve the specific config dict for this item
                            video_config = full_batch_proto.non_tensor_batch["video_processing_config"][idx]
                            question_text = full_batch_proto.non_tensor_batch["question_text"][idx]
                            # Retrieve system prompt etc. if needed

                            # 2. Handle Fallback for Invalid Times
                            use_full_video_fallback = False
                            if start_t is None or end_t is None or not isinstance(start_t, (int, float)) or not isinstance(end_t, (int, float)) or start_t < 0 or start_t >= end_t:
                                print(f"[DEBUG S2 Prep] Item {idx}: Invalid grounding times ({start_t}, {end_t}). Falling back to full video.")
                                start_t = 0.0
                                end_t = None # Use None for end_pts to read till end
                                use_full_video_fallback = True # Flag for potential logging/analysis

                            # 3. Prepare `ele` Dictionary for fetch_video
                            base_config = video_config.copy() # Start with the item's config

                            # --- FIX: Ensure only 'fps' OR 'nframes' is in base_config ---
                            # Decide the priority. Let's prioritize 'nframes' if it's valid (not None).
                            if base_config.get("nframes") is not None:
                                # If nframes is set, remove fps to avoid the assertion error
                                base_config.pop("fps", None)
                                print(f"[DEBUG S2 Prep Fix] Item {idx}: Using 'nframes' ({base_config['nframes']}) for sampling.")
                            else:
                                # If nframes is None or not present, ensure 'fps' is used and remove 'nframes'.
                                base_config.pop("nframes", None)
                                # Make sure 'fps' actually exists if 'nframes' wasn't used
                                if "fps" not in base_config:
                                    print(f"[DEBUG S2 Prep Fix] Item {idx}: 'nframes' is None and 'fps' missing, adding default FPS.")
                                    # Add a default FPS value if it's missing; get it from your constants/config
                                    # Assuming FPS is imported or defined (e.g., from vqa_dataset.py)
                                    base_config["fps"] = 2
                                print(f"[DEBUG S2 Prep Fix] Item {idx}: Using 'fps' ({base_config.get('fps')}) for sampling.")
                            # --- END FIX ---

                            ele_segment = {
                                "video": video_path,
                                "video_start": start_t, # Parsed or 0.0
                                "video_end": end_t,     # Parsed or None
                                # Pass the individual item's video config dict
                                **base_config
                            }
                            print(f"[DEBUG S2 Prep] Item {idx}: Fetching video segment with config: {ele_segment}")

                            # 4. Call `Workspace_video` for the Segment
                            # Ensure fetch_video and dependencies are imported/available
                            # from vqa_dataset import fetch_video # Or wherever it's defined
                            
                            IMAGE_FACTOR = 28
                            clipped_video_frames_tensor = fetch_video(
                                ele_segment,
                                image_factor=video_config.get("image_factor", IMAGE_FACTOR) # Use config's factor or default
                            )

                            if clipped_video_frames_tensor is None or clipped_video_frames_tensor.nelement() == 0:
                                raise ValueError("Video segment processing resulted in empty tensor.")
                            
                            # 4. Prepare item data & Process with helper
                            item_data_for_processor = {
                                "question_text": question_text, "clipped_video": clipped_video_frames_tensor,
                                "original_index": idx # Store original index if needed later
                            }
                            # Pass self.processor, self.tokenizer, self.config to helper
                            processed_item = prepare_stage2_inputs_for_item(item_data_for_processor, self.processor, self.tokenizer, self.config)
                            stage2_batch_inputs.append(processed_item)
                            valid_original_indices.append(idx)

                        except Exception as e:
                            print(f"Error preparing Stage 2 input for item {idx}: {e}. Skipping item.")
                            # Optionally log more details or the error traceback
                            import traceback
                            traceback.print_exc()
                            continue # Skip this item and proceed to the next
                    # Collate and Generate QA for valid items
                    stage2_gen_output = None
                    decoded_qa_texts_map = {}
                    stage2_gen_batch = None # Define before try block
                    if stage2_batch_inputs:
                        try:
                            # Use Verl's collate_fn (assuming it's suitable)
                            from verl.utils.dataset.video_rl_dataset_mc import collate_fn
                            collated_s2_input_dict = collate_fn(stage2_batch_inputs)
                            stage2_gen_batch = DataProto.from_single_dict(collated_s2_input_dict)
                            stage2_gen_batch.meta_info = {'eos_token_id': self.tokenizer.eos_token_id, 'pad_token_id': self.tokenizer.pad_token_id}

                            with _timer('S2_Gen', timing_raw):
                                print(f" Generating QA for {len(stage2_gen_batch)} valid items.")
                                stage2_gen_output = self.actor_rollout_wg.generate_sequences(stage2_gen_batch)

                            with _timer('S2_Decode', timing_raw):
                                stage2_responses = stage2_gen_output.batch.get('responses')
                                if stage2_responses is None: raise ValueError("Stage 2 generation failed (no responses tensor).")
                                temp_decoded_qa_texts = self.tokenizer.batch_decode(stage2_responses, skip_special_tokens=True)
                                for i, original_idx in enumerate(valid_original_indices):
                                    if i < len(temp_decoded_qa_texts): decoded_qa_texts_map[original_idx] = temp_decoded_qa_texts[i]
                                print(f" S2 Decoded QA (first valid): {temp_decoded_qa_texts[0] if len(temp_decoded_qa_texts)>0 else 'N/A'}")

                        except Exception as e:
                            print(f" Error during Stage 2 generation/decoding: {e}.", exc_info=True)
                            stage2_gen_output = None # Ensure PPO update is skipped if S2 fails
                            
                # --- Assemble PPO Update Batch and Run PPO Steps ---
                ppo_update_batch = None
                if stage2_gen_output and valid_original_indices: # Check if Stage 2 ran and produced output
                    try:
                        print(f" -- Assembling PPO Update Batch --")
                        with _timer('PPO_Assemble', timing_raw):
                            # 1. Get Stage 2 Output Tensors from stage2_gen_output.batch
                            s2_responses_tensor = stage2_gen_output.batch.get('responses')
                            s2_attn_mask_full = stage2_gen_output.batch.get('attention_mask')
                            s2_response_len = s2_responses_tensor.shape[1]
                            if s2_attn_mask_full is not None and s2_attn_mask_full.shape[1] >= s2_response_len:
                                s2_response_masks_tensor = s2_attn_mask_full[:, -s2_response_len:]
                            else:
                                print("Assuming all S2 response tokens valid due to missing/short attention mask.")
                                s2_response_masks_tensor = torch.ones_like(s2_responses_tensor)

                            # 2. Get Stage 2 Input Tensors (already collated in stage2_gen_batch)
                            collated_s2_inputs_batch = stage2_gen_batch.batch # Contains input_ids, attention_mask, position_ids, maybe multi_modal_inputs

                            # 3. Gather Non-Tensor Data for Reward Function for the valid items
                            reward_non_tensor_data = defaultdict(list)
                            for original_idx in valid_original_indices: # Iterate using the index list
                                reward_non_tensor_data['decoded_grounding_texts'].append(decoded_grounding_texts[original_idx])
                                reward_non_tensor_data['start_time'].append(full_batch_proto.non_tensor_batch['start_time'][original_idx])
                                reward_non_tensor_data['end_time'].append(full_batch_proto.non_tensor_batch['end_time'][original_idx])
                                reward_non_tensor_data['ground_truth_answer'].append(full_batch_proto.non_tensor_batch['ground_truth_answer'][original_idx])
                            reward_non_tensor_data_np = {k: np.array(v, dtype=object) for k, v in reward_non_tensor_data.items()}

                            # Add UIDs
                            original_uids = full_batch_proto.non_tensor_batch.get('uid')
                            if original_uids is None: original_uids = np.array([str(uuid.uuid4()) for _ in range(current_batch_size)], dtype=object)
                            valid_uids = [original_uids[idx] for idx in valid_original_indices]
                            reward_non_tensor_data_np['uid'] = np.array(valid_uids, dtype=object)

                            # 4. Create the PPO Batch Dictionary (Tensor part)
                            ppo_batch_dict = {
                                "input_ids": collated_s2_inputs_batch['input_ids'],
                                "attention_mask": collated_s2_inputs_batch['attention_mask'], # S2 Input attention mask
                                "position_ids": collated_s2_inputs_batch['position_ids'],
                                "responses": s2_responses_tensor,        # S2 Output tensor
                                "response_mask": s2_response_masks_tensor, # S2 Output mask
                                **({'multi_modal_inputs': collated_s2_inputs_batch['multi_modal_inputs']}
                                   if 'multi_modal_inputs' in collated_s2_inputs_batch else {})
                            }

                            # 5. Create the initial PPO DataProto
                            ppo_update_batch = DataProto(
                                batch=TensorDict(ppo_batch_dict, batch_size=[len(valid_uids)]),
                                non_tensor_batch=reward_non_tensor_data_np
                            )

                    except Exception as ppo_prep_err:
                        print(f" Error assembling PPO update batch: {ppo_prep_err}", exc_info=True)
                        ppo_update_batch = None
                    
                    # --- PPO Update Steps ---
                    if ppo_update_batch:
                        print(f"{step_info_prefix} -- Performing PPO Updates --")
                        try:
                            # === Explicit Reward Calculation ===
                            # Call VQACombinedRewardManager instance (self.reward_fn)
                            with _timer('PPO_Reward', timing_raw):
                                reward_result = self.reward_fn(ppo_update_batch, return_dict=True)
                                reward_tensor = reward_result['reward_tensor']
                                reward_extra_infos_dict = reward_result.get('reward_extra_info', {})

                            # Add reward results to the batch
                            ppo_update_batch.batch['token_level_scores'] = reward_tensor
                            ppo_update_batch.non_tensor_batch.update(reward_extra_infos_dict)
                            # ==================================

                            # Compute Log Probs (Actor and Ref)
                            with _timer('PPO_old_log_prob', timing_raw):
                                old_log_prob_output = self.actor_rollout_wg.compute_log_prob(ppo_update_batch)
                            ppo_update_batch = ppo_update_batch.union(old_log_prob_output)

                            if self.use_reference_policy:
                                with _timer('PPO_ref_log_prob', timing_raw):
                                    ref_log_prob_output = self.ref_policy_wg.compute_ref_log_prob(ppo_update_batch)
                                ppo_update_batch = ppo_update_batch.union(ref_log_prob_output)

                            # Compute Values (Critic)
                            if self.use_critic:
                                with _timer('PPO_values', timing_raw):
                                    values_output = self.critic_wg.compute_values(ppo_update_batch)
                                ppo_update_batch = ppo_update_batch.union(values_output)

                            # Apply KL Penalty
                            with _timer('PPO_KL', timing_raw):
                                if self.config.algorithm.use_kl_in_reward:
                                    ppo_update_batch, kl_metrics = apply_kl_penalty(ppo_update_batch, kl_ctrl=self.kl_ctrl_in_reward, kl_penalty=self.config.algorithm.kl_penalty)
                                    metrics.update(kl_metrics)
                                else: # Ensure token_level_rewards exists for advantage calculation
                                    ppo_update_batch.batch['token_level_rewards'] = ppo_update_batch.batch['token_level_scores']


                            # Compute Advantage
                            with _timer('PPO_Adv', timing_raw):
                                if self.config.trainer.balance_batch: # Balance before advantage if enabled
                                    self._balance_batch(ppo_update_batch, metrics=metrics, logging_prefix='ppo_seqlen')
                                # Add global_token_num meta info if needed by advantage/logging
                                ppo_update_batch.meta_info['global_token_num'] = torch.sum(ppo_update_batch.batch['attention_mask'], dim=-1).tolist() # Use S2 input mask length

                                ppo_update_batch = compute_advantage(
                                    ppo_update_batch,
                                    adv_estimator=self.config.algorithm.adv_estimator,
                                    gamma=self.config.algorithm.gamma, lam=self.config.algorithm.lam,
                                )

                            # Update Critic
                            if self.use_critic:
                                with _timer('PPO_update_critic', timing_raw):
                                    critic_output = self.critic_wg.update_critic(ppo_update_batch)
                                # Ensure metrics are prefixed to avoid clashes if critic returns same keys
                                critic_output_metrics = reduce_metrics({f"critic/{k}": v for k, v in critic_output.meta_info['metrics'].items()})
                                metrics.update(critic_output_metrics)

                            # Update Actor
                            if self.config.trainer.critic_warmup <= self.global_steps:
                                with _timer('PPO_update_actor', timing_raw):
                                    actor_output = self.actor_rollout_wg.update_actor(ppo_update_batch)
                                # Ensure metrics are prefixed
                                actor_output_metrics = reduce_metrics({f"actor/{k}": v for k, v in actor_output.meta_info['metrics'].items()})
                                metrics.update(actor_output_metrics)
                            else:
                                print(f"{step_info_prefix} Skipping actor update (Critic warmup: {self.global_steps}/{self.config.trainer.critic_warmup})")

                        except Exception as ppo_update_err:
                            print(f"{step_info_prefix} Error during PPO update steps: {ppo_update_err}", exc_info=True)
                    else:
                        print(f"{step_info_prefix} Skipping PPO updates as no valid batch was prepared or S2 failed.")

                    # --- Logging ---
                    print(f"{step_info_prefix} -- Logging Metrics --")
                    try:
                        # Add data/timing metrics from PPO batch if it exists
                        if ppo_update_batch:
                            metrics.update(compute_data_metrics(batch=ppo_update_batch, use_critic=self.use_critic))
                            metrics.update(compute_timing_metrics(batch=ppo_update_batch, timing_raw=timing_raw))
                            n_gpus = self.resource_pool_manager.get_n_gpus()
                            metrics.update(compute_throughout_metrics(batch=ppo_update_batch, timing_raw=timing_raw, n_gpus=n_gpus))

                        # Add averaged reward metrics directly from the reward manager's output if available
                        reward_metrics_to_log = {}
                        if 'reward_extra_info' in locals() and reward_extra_infos_dict:
                            for key, val_array in reward_extra_infos_dict.items():
                                if isinstance(val_array, (np.ndarray, list)) and len(val_array) > 0:
                                    numeric_vals = [v for v in val_array if isinstance(v, (int, float)) and np.isfinite(v)]
                                    reward_metrics_to_log[f'reward/{key}_mean'] = np.mean(numeric_vals) if len(numeric_vals) > 0 else 0.0
                                else: reward_metrics_to_log[f'reward/{key}_mean'] = 0.0
                        metrics.update(reward_metrics_to_log)

                        # Log overall step time
                        metrics['timing/step_total_s'] = sum(timing_raw.values())
                        print(f"{step_info_prefix} Metrics Logged. Rewards: {reward_metrics_to_log}")
                    except Exception as log_err:
                        print(f"{step_info_prefix} Error during logging: {log_err}")

                    # --- Validate ---
                    # Note: Using standard _validate. Needs VQA-specific validation eventually.
                    is_last_step = self.global_steps >= self.total_training_steps # Check again after potential increment
                    if self.val_reward_fn is not None and self.config.trainer.test_freq > 0 and (is_last_step or self.global_steps % self.config.trainer.test_freq == 0):
                        with _timer('testing', timing_raw):
                            print(f"{step_info_prefix} -- Running Validation --")
                            val_metrics: dict = self._validate() # Uses val_dataloader + val_reward_fn (TVG?)
                            print(f"{step_info_prefix} Validation Metrics Logged.")
                            if is_last_step: last_val_metrics = val_metrics

                    # --- Save Checkpoint ---
                    if self.config.trainer.save_freq > 0 and (is_last_step or \
                            self.global_steps % self.config.trainer.save_freq == 0):
                        print(f"{step_info_prefix} -- Saving Checkpoint --")
                        with _timer('save_checkpoint', timing_raw): self._save_checkpoint()
                        # Saving training samples is omitted here for complexity, can be added back

                    progress_bar.update(1)
                    # Check completion condition again after all operations for the step
                    if self.global_steps >= self.total_training_steps:
                        is_last_step = True; break # Exit batch loop

                # --- End of Batch Loop ---
                if is_last_step:
                    print(f"--- Reached Last Step ({self.global_steps}), Breaking Epoch Loop ---")
                    break # Exit epoch loop

            progress_bar.close()
            print("--- Finished VQA Training Loop ---")
            if last_val_metrics: pprint(f'Final validation metrics: {last_val_metrics}')
                    # # --- Batch Generation for Stage 2 ---
                    # stage2_gen_output = None
                    # decoded_qa_texts = ["<S2 Generation Failed>"] * current_batch_size # Initialize with failure state

                    # if stage2_batch_inputs: # Only proceed if there are valid inputs
                    #     # Collate the list of dictionaries into a DataProto batch
                    #     # This requires a custom collation or careful stacking/padding
                    #     # Example using a basic collate function (assumes one exists):
                    #     from verl.utils.dataset.video_rl_dataset_mc import collate_fn # Use the existing one
                    #     collated_s2_input_dict = collate_fn(stage2_batch_inputs)
                    #     stage2_gen_batch = DataProto.from_single_dict(collated_s2_input_dict)

                    #     print(f"[DEBUG S2 Gen] Generating QA for {len(stage2_gen_batch)} valid items.")
                    #     # ** CONCEPTUAL CALL: Generate QA sequences **
                    #     stage2_gen_output = self.actor_rollout_wg.generate_sequences(stage2_gen_batch)

                    #     # --- Map results back to original indices ---
                    #     stage2_responses = stage2_gen_output.batch.get('responses')
                    #     if stage2_responses is not None:
                    #         temp_decoded_qa_texts = self.tokenizer.batch_decode(stage2_responses, skip_special_tokens=True)
                    #         for i, original_idx in enumerate(valid_original_indices):
                    #             if i < len(temp_decoded_qa_texts):
                    #                 decoded_qa_texts[original_idx] = temp_decoded_qa_texts[i] # Place result in correct slot
                    #     print(f"[DEBUG S2 Gen] Finished QA generation. Decoded texts (first 5 valid):")
                    #     count = 0
                    #     for idx in valid_original_indices:
                    #         print(f"  Item {idx}: {decoded_qa_texts[idx]}, groundtruth: {full_batch_proto.non_tensor_batch['ground_truth_answer'][idx]}")
                    #         count += 1
                    #         if count >= 5: break

                    # else:
                    #     print("[DEBUG S2 Gen] No valid items to generate QA for.")
                       
        # print("--- Starting Training Loop ---")
        # # Removed try...finally block as saving happens during the loop now
        

        # for epoch in range(self.config.trainer.total_epochs):
        #     print(f"--- Starting Epoch {epoch+1}/{self.config.trainer.total_epochs} ---")
        #     for i, batch_dict in enumerate(self.two_stage_dataloader):
        #         print(f"\n--- Starting Global Step {self.global_steps} (Epoch {epoch+1}, Batch {i+1}) ---")
        #         metrics = {}
        #         timing_raw = {}

        #          # --- Load and Validate Batch ---
        #         if not batch_dict: print("[DEBUG] Empty batch_dict. Skipping."); continue
        #         try:
        #             batch = DataProto.from_single_dict(batch_dict)
        #             print(f"[DEBUG] Batch Size: {len(batch.batch)}") 
                    
        #         except Exception as e: print(f"Error DataProto: {e}. Skip."); continue
                
        #         # --- Stage 1: Grounding ---
        #         print(f"[{self.global_steps}] -- Stage 1: Generating Grounding --")
        #         with _timer('stage1_prep_gen_parse', timing_raw): # Combine timing for conceptual stage
        #             # Prepare batch for Stage 1
        #             try:
        #                 # stage1_batch = DataProto(
        #                 #     batch=full_batch_proto.batch.select("stage1_input_ids", "stage1_attention_mask", "stage1_position_ids"),
        #                 #     non_tensor_batch=full_batch_proto.non_tensor_batch.select("stage1_multi_modal_inputs")
        #                 # )
        #                 stage1_batch = batch.pop(batch_keys=['input_ids', 'attention_mask', 'position_ids'], non_tensor_batch_keys=['multi_modal_inputs'])
        #                 # stage1_batch = full_batch_proto.pop(batch_keys=['stage1_input_ids', 'stage1_attention_mask', 'stage1_position_ids'], non_tensor_batch_keys=['stage1_multi_modal_inputs'])
        #             except KeyError as e: print(f"Missing Stage 1 key: {e}. Skip.");

        #             # Conceptual: Generate grounding sequences
        #             stage1_gen_output = self.actor_rollout_wg.generate_sequences(stage1_batch)

        #             # Parse Stage 1 Output
        #             batch = batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
        #             batch = batch.union(stage1_gen_output)
        #             response_tokens = batch.batch['responses'][0]
                    
        #             decoded_response = self.tokenizer.decode(response_tokens, skip_special_tokens=True)
        #             print("-" * 50); 
        #             print(f"[DEBUG SAMPLE] Output (Index 0): {decoded_response}"); 
        #             print("-" * 50)
        #             exit(0) # DEBUG
                    
        #         # Retrieve the start time and end time from response, if not found, fallback to whole video
                
        #         # construct the new prompt batch for stage 2, with the clipped video and the question and options, ask the model to give the letter answer.
                
        #         # get the response
                
        #         # get the reward using reward_fn 
                
        #         # update the GRPO
                
                


        #         is_last_step = self.global_steps >= self.total_training_steps

        #         with _timer('step', timing_raw):
        #             # generate sequences
        #             with _timer('gen', timing_raw): gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch)

        #             if 'uid' not in batch.non_tensor_batch: # Add UID if not present
        #                  batch.non_tensor_batch['uid'] = np.array([str(uuid.uuid4()) for _ in range(len(batch.batch))], dtype=object)
        #             print(f"[DEBUG] Repeating batch {self.config.actor_rollout_ref.rollout.n} times...") # DEBUG
        #             batch = batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
        #             print(f"[DEBUG] Batch size after repeat: {len(batch.batch)}") # DEBUG
        #             batch = batch.union(gen_batch_output)
        #             print("[DEBUG] Merged gen_batch_output. Batch keys:", list(batch.batch.keys())) # DEBUG

        #             # --- Store and Log Sample ---
        #             if len(batch.batch) > 0:
        #                 try:
        #                     prompt_text_to_print = original_prompts_text[0] if original_prompts_text is not None and len(original_prompts_text) > 0 else "N/A"
        #                     video_ref_to_save = original_videos[0][0] if original_videos is not None and len(original_videos) > 0 and len(original_videos[0]) > 0 else "N/A"
        #                     response_tokens = batch.batch['responses'][0]
        #                     decoded_response = self.tokenizer.decode(response_tokens, skip_special_tokens=True)
        #                     print("-" * 50); 
        #                     print(f"[DEBUG SAMPLE] Prompt (Index 0): {prompt_text_to_print}"); 
        #                     print(f"[DEBUG SAMPLE] Output (Index 0): {decoded_response}"); 
        #                     print("-" * 50)
        #                     # Append sample to list
        #                     sample_to_save = { "step": self.global_steps, "prompt": prompt_text_to_print, "response": decoded_response, "video_ref": video_ref_to_save}
        #                     self.all_train_samples.append(sample_to_save)
        #                 except Exception as e: print(f"[DEBUG SAMPLE] Error printing/saving sample: {e}")

        #             batch.batch['response_mask'] = compute_response_mask(batch)
        #             if self.config.trainer.balance_batch: self._balance_batch(batch, metrics=metrics)
        #             batch.meta_info['global_token_num'] = torch.sum(batch.batch['attention_mask'], dim=-1).tolist()

        #             # --- Log Probs ---
        #             with _timer('old_log_prob', timing_raw): old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
        #             # ... (entropy calc) ...
        #             batch = batch.union(old_log_prob)
        #             if self.use_reference_policy:
        #                 with _timer('ref', timing_raw): ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
        #                 batch = batch.union(ref_log_prob)

        #             # --- Compute Values ---
        #             if self.use_critic:
        #                 with _timer('values', timing_raw): values = self.critic_wg.compute_values(batch)
        #                 batch = batch.union(values)

        #             # --- Advantage Computation ---
        #             with _timer('adv', timing_raw):
        #                 # ... (RM logic) ...
        #                 try:
        #                     reward_result = self.reward_fn(batch, return_dict=True)
        #                     reward_tensor = reward_result['reward_tensor']
        #                     reward_extra_infos_dict = reward_result.get('reward_extra_info', {})
        #                 except Exception as e: print(f'Error in reward_fn: {e}'); import traceback; traceback.print_exc(); raise e
        #                 batch.batch['token_level_scores'] = reward_tensor
        #                 if reward_extra_infos_dict: batch.non_tensor_batch.update({k: np.array(v) for k, v in reward_extra_infos_dict.items()})

        #                 # --- Log Extra Reward Metrics ---
        #                 if reward_extra_infos_dict:
        #                     for key, val_array in reward_extra_infos_dict.items():
        #                         if isinstance(val_array, (list, np.ndarray)) and len(val_array) > 0:
        #                              if isinstance(val_array, np.ndarray): mean_val = np.mean([v for v in val_array if np.isfinite(v)]) if len(val_array)>0 else 0
        #                              else: mean_val = np.mean([v for v in val_array if isinstance(v, (int, float)) and np.isfinite(v)]) if len(val_array)>0 else 0
        #                              metrics[f'reward/{key}_mean'] = mean_val
        #                         else: metrics[f'reward/{key}_mean'] = 0
        #                 # --- End Log Extra Metrics ---

        #                 if self.config.algorithm.use_kl_in_reward: # ... (KL penalty) ...
        #                     batch, kl_metrics = apply_kl_penalty(batch, kl_ctrl=self.kl_ctrl_in_reward, kl_penalty=self.config.algorithm.kl_penalty); metrics.update(kl_metrics)
        #                 else: batch.batch['token_level_rewards'] = batch.batch['token_level_scores']

        #                 batch = compute_advantage(batch, adv_estimator=self.config.algorithm.adv_estimator, gamma=self.config.algorithm.gamma, lam=self.config.algorithm.lam, num_repeat=self.config.actor_rollout_ref.rollout.n)

        #             # --- Update Critic ---
        #             if self.use_critic:
        #                 with _timer('update_critic', timing_raw): critic_output = self.critic_wg.update_critic(batch)
        #                 critic_output_metrics = reduce_metrics(critic_output.meta_info['metrics']); metrics.update(critic_output_metrics)

        #             # --- Update Actor ---
        #             if self.config.trainer.critic_warmup <= self.global_steps:
        #                 # --- WORKAROUND START (Optional) ---
        #                 problematic_keys = ['tvg_accuracy', 'tvg_format', 'tvg_combined']
        #                 # print(f"[DEBUG] WORKAROUND: Removing problematic non-tensor keys before update_actor: {problematic_keys}")
        #                 for key in problematic_keys:
        #                     if key in batch.non_tensor_batch: del batch.non_tensor_batch[key]
        #                 # --- WORKAROUND END ---
        #                 with _timer('update_actor', timing_raw): actor_output = self.actor_rollout_wg.update_actor(batch)
        #                 actor_output_metrics = reduce_metrics(actor_output.meta_info['metrics']); metrics.update(actor_output_metrics)

        #             # --- Validate ---
        #             if self.val_reward_fn is not None and self.config.trainer.test_freq > 0 and \
        #                 (is_last_step or self.global_steps % self.config.trainer.test_freq == 0):
        #                 with _timer('testing', timing_raw): val_metrics: dict = self._validate()
        #                 if is_last_step: last_val_metrics = val_metrics
        #                 metrics.update(val_metrics)

        #             # --- Save Checkpoint & Training Samples ---
        #             if self.config.trainer.save_freq > 0 and ( is_last_step or \
        #                     self.global_steps % self.config.trainer.save_freq == 0):
        #                 print(f"[DEBUG] Saving checkpoint for step {self.global_steps}...")
        #                 with _timer('save_checkpoint', timing_raw): self._save_checkpoint()
        #                 print("[DEBUG] Checkpoint saved.")

        #                 # --- MODIFICATION: Save training samples periodically ---
        #                 try:
        #                     train_output_dir = os.path.join(self.config.trainer.default_local_dir, "training_samples")
        #                     os.makedirs(train_output_dir, exist_ok=True)
        #                     # Save samples collected *since last save*
        #                     train_samples_file_path = os.path.join(train_output_dir, f"train_samples_step_{self.global_steps}.jsonl")
        #                     print(f"[DEBUG] Saving {len(self.all_train_samples)} training samples to: {train_samples_file_path}")
        #                     with open(train_samples_file_path, 'w') as f:
        #                         for sample in self.all_train_samples:
        #                              f.write(json.dumps(sample) + '\n')
        #                     self.all_train_samples.clear() # <<< Clear list after saving
        #                     print("[DEBUG] Training samples saved and list cleared.")
        #                 except Exception as e:
        #                      print(f"Error saving training samples at step {self.global_steps}: {e}")
        #                 # --- END MODIFICATION ---


        #         # --- Logging ---
        #         # ... (logging metrics code as before) ...
        #         metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
        #         metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
        #         n_gpus = self.resource_pool_manager.get_n_gpus()
        #         metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, n_gpus=n_gpus))
        #         logger.log(data=metrics, step=self.global_steps)
        #         print(f"[DEBUG] Step {self.global_steps} Metrics Logged.")


        #         if is_last_step:
        #             pprint(f'Final validation metrics: {last_val_metrics}'); progress_bar.close()
        #             print("--- Reached Last Step, Exiting Fit ---"); break # Break inner loop
        #             # with _timer('save_checkpoint', timing_raw): self._save_checkpoint()

        #         progress_bar.update(1)
        #         self.global_steps += 1
        #     # End of batch loop

        # print("--- Finished Training Loop ---") # DEBUG