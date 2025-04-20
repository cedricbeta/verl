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
from verl.utils.dataset.video_rl_dataset import RLHFDataset, collate_fn
# from verl.utils.dataset.video_rl_dataset import VideoRLHFDataset, collate_fn
from verl.utils.tracking import ValidationGenerationsLogger
from torch.utils.data import Dataset, RandomSampler, SequentialSampler
from torchdata.stateful_dataloader import StatefulDataLoader
import json
from collections import defaultdict
from typing import List, Dict, Any

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

    # def _validate(self):
    #     data_source_lst = []
    #     reward_extra_infos_dict: dict[str, list] = defaultdict(list)

    #     # Lists to collect samples for the table
    #     sample_inputs = []
    #     sample_outputs = []
    #     sample_scores = []

    #     for test_data in self.val_dataloader:
    #         test_batch = DataProto.from_single_dict(test_data)

    #         # repeat test batch
    #         test_batch = test_batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.val_kwargs.n,
    #                                        interleave=True)

    #         # we only do validation on rule-based rm
    #         if self.config.reward_model.enable and test_batch[0].non_tensor_batch['reward_model']['style'] == 'model':
    #             return {}

    #         # Store original inputs
    #         input_ids = test_batch.batch['input_ids']
    #         # TODO: Can we keep special tokens except for padding tokens?
    #         input_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in input_ids]
    #         sample_inputs.extend(input_texts)

    #         if 'multi_modal_inputs' in test_batch.non_tensor_batch.keys():
    #             test_gen_batch = test_batch.pop(
    #                 batch_keys=['input_ids', 'attention_mask', 'position_ids'],
    #                 non_tensor_batch_keys=['raw_prompt_ids', 'multi_modal_data', 'multi_modal_inputs'],
    #             )
    #         else:
    #             test_gen_batch = test_batch.pop(
    #                 batch_keys=['input_ids', 'attention_mask', 'position_ids'],
    #                 non_tensor_batch_keys=['raw_prompt_ids'],
    #             )

    #         test_gen_batch.meta_info = {
    #             'eos_token_id': self.tokenizer.eos_token_id,
    #             'pad_token_id': self.tokenizer.pad_token_id,
    #             'recompute_log_prob': False,
    #             'do_sample': self.config.actor_rollout_ref.rollout.val_kwargs.do_sample,
    #             'validate': True,
    #         }
    #         print(f'test_gen_batch meta info: {test_gen_batch.meta_info}')

    #         # pad to be divisible by dp_size
    #         test_gen_batch_padded, pad_size = pad_dataproto_to_divisor(test_gen_batch, self.actor_rollout_wg.world_size)
    #         test_output_gen_batch_padded = self.actor_rollout_wg.generate_sequences(test_gen_batch_padded)

    #         # unpad
    #         test_output_gen_batch = unpad_dataproto(test_output_gen_batch_padded, pad_size=pad_size)
    #         print('validation generation end')

    #         # Store generated outputs
    #         output_ids = test_output_gen_batch.batch['responses']
    #         output_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in output_ids]
    #         sample_outputs.extend(output_texts)

    #         test_batch = test_batch.union(test_output_gen_batch)

    #         # evaluate using reward_function
    #         result = self.val_reward_fn(test_batch, return_dict=True)
    #         reward_tensor = result["reward_tensor"]
    #         scores = reward_tensor.sum(-1).cpu().tolist()
    #         sample_scores.extend(scores)

    #         reward_extra_infos_dict["reward"].extend(scores)
    #         if "reward_extra_info" in result:
    #             for key, lst in result["reward_extra_info"].items():
    #                 reward_extra_infos_dict[key].extend(lst)

    #         data_source_lst.append(test_batch.non_tensor_batch.get('data_source', ['unknown'] * reward_tensor.shape[0]))

    #     self._maybe_log_val_generations(inputs=sample_inputs, outputs=sample_outputs, scores=sample_scores)

    #     for key_info, lst in reward_extra_infos_dict.items():
    #         assert len(lst) == 0 or len(lst) == len(sample_scores), f"{key_info}: {len(lst)=}, {len(sample_scores)=}"

    #     data_sources = np.concatenate(data_source_lst, axis=0)

    #     data_src2var2metric2val = process_validation_metrics(data_sources, sample_inputs, reward_extra_infos_dict)
    #     metric_dict = {}
    #     for data_source, var2metric2val in data_src2var2metric2val.items():
    #         core_var = "acc" if "acc" in var2metric2val else "reward"
    #         for var_name, metric2val in var2metric2val.items():
    #             n_max = max([int(name.split("@")[-1].split("/")[0]) for name in metric2val.keys()])
    #             for metric_name, metric_val in metric2val.items():
    #                 if (var_name == core_var) and any(
    #                         metric_name.startswith(pfx) for pfx in ["mean", "maj", "best"]) and (f"@{n_max}"
    #                                                                                              in metric_name):
    #                     metric_sec = "val-core"
    #                 else:
    #                     metric_sec = "val-aux"
    #                 pfx = f"{metric_sec}/{data_source}/{var_name}/{metric_name}"
    #                 metric_dict[pfx] = metric_val

    #     return metric_dict
    
    

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
    
    # def _validate(self):
    #     print("[DEBUG] --- Running _validate ---") # DEBUG
    #     # --- MODIFICATION: Store ALL validation results ---
    #     all_val_results = []
    #     # --- END MODIFICATION ---

    #     data_source_lst = []
    #     reward_extra_infos_dict: dict[str, list] = defaultdict(list)
    #     sample_inputs = []
    #     sample_outputs = []
    #     sample_scores = []

    #     # Validation loop only runs once as val_dataloader has batch_size=len(dataset)
    #     for test_data in self.val_dataloader:
    #         test_batch = DataProto.from_single_dict(test_data)
    #         original_batch_size = len(test_batch) # Store original size before repeat

    #         # Store original inputs for logging ALL results later
    #         # Use non_tensor 'prompts' key if available, else decode input_ids
    #         if "prompts" in test_batch.non_tensor_batch:
    #              original_prompts = deepcopy(test_batch.non_tensor_batch["prompts"]) # Deepcopy if needed
    #         else:
    #              original_prompts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in test_batch.batch['input_ids']]

    #         # Keep original ground truth if needed
    #         original_ground_truth = deepcopy(test_batch.non_tensor_batch.get("ground_truth"))

    #         # repeat test batch
    #         repeat_n = self.config.actor_rollout_ref.rollout.val_kwargs.n
    #         test_batch = test_batch.repeat(repeat_times=repeat_n, interleave=True)

    #         # ... (rest of the popping logic for gen_batch) ...
    #         if 'multi_modal_inputs' in test_batch.non_tensor_batch.keys():
    #              test_gen_batch = test_batch.pop(
    #                  batch_keys=['input_ids', 'attention_mask', 'position_ids'],
    #                  non_tensor_batch_keys=['raw_prompt_ids', 'multi_modal_data', 'multi_modal_inputs'], # Pop more non-tensors
    #              )
    #         else:
    #              test_gen_batch = test_batch.pop(
    #                  batch_keys=['input_ids', 'attention_mask', 'position_ids'],
    #                  non_tensor_batch_keys=['raw_prompt_ids'], # Pop more non-tensors
    #              )

    #         test_gen_batch.meta_info = { # ... (meta_info setup as before) ...
    #             'eos_token_id': self.tokenizer.eos_token_id,
    #             'pad_token_id': self.tokenizer.pad_token_id,
    #             'recompute_log_prob': False,
    #             'do_sample': self.config.actor_rollout_ref.rollout.val_kwargs.do_sample,
    #             'validate': True,
    #         }
    #         print(f'[DEBUG VAL] test_gen_batch meta info: {test_gen_batch.meta_info}')


    #         # pad, generate, unpad
    #         test_gen_batch_padded, pad_size = pad_dataproto_to_divisor(test_gen_batch, self.actor_rollout_wg.world_size)
    #         test_output_gen_batch_padded = self.actor_rollout_wg.generate_sequences(test_gen_batch_padded)
    #         test_output_gen_batch = unpad_dataproto(test_output_gen_batch_padded, pad_size=pad_size)
    #         print('[DEBUG VAL] validation generation end')

    #         # Store generated outputs
    #         output_ids = test_output_gen_batch.batch['responses']
    #         output_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in output_ids]

    #         # Reconstruct full batch info for reward function
    #         test_batch = test_batch.union(test_output_gen_batch)
    #         # --- MODIFICATION: Add original ground_truth back for reward calc ---
    #         # Note: This assumes repeat preserves order or reward_fn handles repeated inputs
    #         if original_ground_truth is not None:
    #              repeated_gt = np.repeat(original_ground_truth, repeat_n, axis=0)
    #              test_batch.non_tensor_batch["ground_truth"] = repeated_gt
    #         # Similarly add back other needed non-tensor fields if popped earlier
    #         if "video_length" in test_gen_batch.non_tensor_batch:
    #              repeated_vl = np.repeat(test_gen_batch.non_tensor_batch["video_length"], repeat_n, axis=0)
    #              test_batch.non_tensor_batch["video_length"] = repeated_vl
    #         # --- END MODIFICATION ---

    #         # evaluate using reward_function
    #         print("[DEBUG VAL] Calculating rewards...") # DEBUG
    #         result = self.val_reward_fn(test_batch, return_dict=True)
    #         print("[DEBUG VAL] Rewards calculated.") # DEBUG
    #         reward_tensor = result["reward_tensor"]
    #         scores = reward_tensor.sum(-1).cpu().tolist() # Score per generated sequence

    #         # --- MODIFICATION: Collect ALL results ---
    #         # Reshape/group results per original prompt if n > 1
    #         current_reward_extra = result.get("reward_extra_info", {})
    #         for idx in range(original_batch_size):
    #              prompt = original_prompts[idx]
    #              results_for_prompt = []
    #              extra_info_for_prompt = defaultdict(list)
    #              for repeat_idx in range(repeat_n):
    #                   global_idx = idx * repeat_n + repeat_idx
    #                   results_for_prompt.append({
    #                       "response": output_texts[global_idx],
    #                       "score": scores[global_idx]
    #                   })
    #                   for key, values in current_reward_extra.items():
    #                       if len(values) > global_idx: # Check if reward_fn returned extras correctly
    #                          extra_info_for_prompt[key].append(values[global_idx])

    #              all_val_results.append({
    #                  "step": self.global_steps,
    #                  "prompt": prompt,
    #                  "ground_truth": original_ground_truth[idx] if original_ground_truth is not None else "N/A",
    #                  "outputs": results_for_prompt, # List of dicts for n>1 generations
    #                  "extra_rewards": dict(extra_info_for_prompt) # Dict of lists for n>1 generations
    #              })
    #         # --- END MODIFICATION ---

    #         # Keep populating for metrics processing
    #         sample_inputs.extend(original_prompts * repeat_n) # Repeat prompts to match repeated outputs
    #         sample_outputs.extend(output_texts)
    #         sample_scores.extend(scores)
    #         reward_extra_infos_dict["reward"].extend(scores)
    #         if "reward_extra_info" in result:
    #              for key, lst in result["reward_extra_info"].items(): reward_extra_infos_dict[key].extend(lst)
    #         data_source_lst.append(test_batch.non_tensor_batch.get('data_source', ['unknown'] * len(scores)))

    #     # Log a SUBSET to dashboard (existing logic)
    #     self._maybe_log_val_generations(inputs=sample_inputs, outputs=sample_outputs, scores=sample_scores)

    #     # --- MODIFICATION: Save ALL results to file ---
    #     try:
    #         # Define filename based on global step
    #         val_output_dir = os.path.join(self.config.trainer.default_local_dir, "validation_results")
    #         os.makedirs(val_output_dir, exist_ok=True)
    #         self.all_val_samples_file_path = os.path.join(val_output_dir, f"val_step_{self.global_steps}.json")
    #         print(f"[DEBUG VAL] Saving all validation results to: {self.all_val_samples_file_path}")
    #         with open(self.all_val_samples_file_path, 'w') as f:
    #              json.dump(all_val_results, f, indent=2)
    #     except Exception as e:
    #         print(f"Error saving validation results: {e}")
    #     # --- END MODIFICATION ---

    #     print("[DEBUG VAL] Processing metrics...") # DEBUG
    #     metric_dict = process_validation_metrics(
    #          data_sources=data_sources,
    #          sample_inputs=sample_inputs, # Passed but not used in simplified version
    #          infos_dict=reward_extra_infos_dict,
    #          target_metrics=["reward", "tvg_accuracy", "tvg_format", "tvg_combined"] # Specify desired keys
    #     )
    #     print("[DEBUG VAL] Metrics processing complete.") # DEBUG
        
    #     # Process metrics for logging (existing logic)
    #     # data_sources = np.concatenate(data_source_lst, axis=0)
    #     # data_src2var2metric2val = process_validation_metrics(data_sources, sample_inputs, reward_extra_infos_dict)
    #     # # ... (metric dictionary creation as before) ...
    #     # metric_dict = {}
    #     # for data_source, var2metric2val in data_src2var2metric2val.items():
    #     #      core_var = "acc" if "acc" in var2metric2val else "reward"
    #     #      for var_name, metric2val in var2metric2val.items():
    #     #           n_max = max([int(name.split("@")[-1].split("/")[0]) for name in metric2val.keys()])
    #     #           for metric_name, metric_val in metric2val.items():
    #     #                if (var_name == core_var) and any(
    #     #                        metric_name.startswith(pfx) for pfx in ["mean", "maj", "best"]) and (f"@{n_max}"
    #     #                                                                                             in metric_name):
    #     #                     metric_sec = "val-core"
    #     #                else:
    #     #                     metric_sec = "val-aux"
    #     #                pfx = f"{metric_sec}/{data_source}/{var_name}/{metric_name}"
    #     #                metric_dict[pfx] = metric_val


    #     return metric_dict

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

    # def fit(self):
    #     """
    #     The training loop of PPO.
    #     The driver process only need to call the compute functions of the worker group through RPC to construct the PPO dataflow.
    #     The light-weight advantage computation is done on the driver process.
    #     """
    #     from verl.utils.tracking import Tracking
    #     from omegaconf import OmegaConf

    #     logger = Tracking(project_name=self.config.trainer.project_name,
    #                       experiment_name=self.config.trainer.experiment_name,
    #                       default_backend=self.config.trainer.logger,
    #                       config=OmegaConf.to_container(self.config, resolve=True))

    #     self.global_steps = 0

    #     # load checkpoint before doing anything
    #     self._load_checkpoint()

    #     # perform validation before training
    #     # currently, we only support validation using the reward_function.
    #     if self.val_reward_fn is not None and self.config.trainer.get('val_before_train', True):
    #         val_metrics = self._validate()
    #         pprint(f'Initial validation metrics: {val_metrics}')
    #         logger.log(data=val_metrics, step=self.global_steps)
    #         if self.config.trainer.get('val_only', False):
    #             return

    #     # add tqdm
    #     progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="Training Progress")

    #     # we start from step 1
    #     self.global_steps += 1
    #     last_val_metrics = None

    #     for epoch in range(self.config.trainer.total_epochs):
    #         for batch_dict in self.train_dataloader:
    #             metrics = {}
    #             timing_raw = {}

    #             batch: DataProto = DataProto.from_single_dict(batch_dict)
    #             print("batch: ", batch.meta_info)

    #             # pop those keys for generation
    #             if 'multi_modal_inputs' in batch.non_tensor_batch.keys():
    #                 gen_batch = batch.pop(
    #                     batch_keys=['input_ids', 'attention_mask', 'position_ids'],
    #                     non_tensor_batch_keys=['raw_prompt_ids', 'multi_modal_data', 'multi_modal_inputs'],
    #                 )
    #             else:
    #                 gen_batch = batch.pop(
    #                     batch_keys=['input_ids', 'attention_mask', 'position_ids'],
    #                     non_tensor_batch_keys=['raw_prompt_ids'],
    #                 )

    #             is_last_step = self.global_steps >= self.total_training_steps

    #             with _timer('step', timing_raw):
    #                 # generate a batch
    #                 with _timer('gen', timing_raw):
    #                     gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch)

    #                 if self.config.algorithm.adv_estimator == AdvantageEstimator.REMAX:
    #                     with _timer('gen_max', timing_raw):
    #                         gen_baseline_batch = deepcopy(gen_batch)
    #                         gen_baseline_batch.meta_info['do_sample'] = False
    #                         gen_baseline_output = self.actor_rollout_wg.generate_sequences(gen_baseline_batch)

    #                         batch = batch.union(gen_baseline_output)
    #                         reward_baseline_tensor = self.reward_fn(batch)
    #                         reward_baseline_tensor = reward_baseline_tensor.sum(dim=-1)

    #                         batch.pop(batch_keys=list(gen_baseline_output.batch.keys()))

    #                         batch.batch['reward_baselines'] = reward_baseline_tensor

    #                         del gen_baseline_batch, gen_baseline_output

    #                 batch.non_tensor_batch['uid'] = np.array([str(uuid.uuid4()) for _ in range(len(batch.batch))],
    #                                                          dtype=object)
    #                 # repeat to align with repeated responses in rollout
    #                 batch = batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
    #                 batch = batch.union(gen_batch_output)

    #                 batch.batch['response_mask'] = compute_response_mask(batch)
    #                 # balance the number of valid tokens on each dp rank.
    #                 # Note that this breaks the order of data inside the batch.
    #                 # Please take care when you implement group based adv computation such as GRPO and rloo
    #                 if self.config.trainer.balance_batch:
    #                     self._balance_batch(batch, metrics=metrics)

    #                 # compute global_valid tokens
    #                 batch.meta_info['global_token_num'] = torch.sum(batch.batch['attention_mask'], dim=-1).tolist()

    #                 # recompute old_log_probs
    #                 with _timer('old_log_prob', timing_raw):
    #                     old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
    #                     entropys = old_log_prob.batch['entropys']
    #                     response_masks = batch.batch['response_mask']
    #                     loss_agg_mode = self.config.actor_rollout_ref.actor.loss_agg_mode
    #                     entropy_loss = agg_loss(loss_mat=entropys,
    #                                             loss_mask=response_masks,
    #                                             loss_agg_mode=loss_agg_mode)
    #                     old_log_prob_metrics = {"actor/entropy_loss": entropy_loss.detach().item()}
    #                     metrics.update(old_log_prob_metrics)
    #                     old_log_prob.batch.pop('entropys')
    #                     batch = batch.union(old_log_prob)

    #                 if self.use_reference_policy:
    #                     # compute reference log_prob
    #                     with _timer('ref', timing_raw):
    #                         ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
    #                         batch = batch.union(ref_log_prob)

    #                 # compute values
    #                 if self.use_critic:
    #                     with _timer('values', timing_raw):
    #                         values = self.critic_wg.compute_values(batch)
    #                         batch = batch.union(values)

    #                 with _timer('adv', timing_raw):
    #                     # compute scores. Support both model and function-based.
    #                     # We first compute the scores using reward model. Then, we call reward_fn to combine
    #                     # the results from reward model and rule-based results.
    #                     if self.use_rm:
    #                         # we first compute reward model score
    #                         reward_tensor = self.rm_wg.compute_rm_score(batch)
    #                         batch = batch.union(reward_tensor)

    #                     # we combine with rule-based rm
    #                     reward_extra_infos_dict: dict[str, list]
    #                     try:
    #                         reward_result = self.reward_fn(batch, return_dict=True)
    #                         reward_tensor = reward_result['reward_tensor']
    #                         reward_extra_infos_dict = reward_result['reward_extra_info']
    #                     except Exception as e:
    #                         print(f'Error in reward_fn: {e}')
    #                         reward_tensor = self.reward_fn(batch)
    #                         reward_extra_infos_dict = {}

    #                     batch.batch['token_level_scores'] = reward_tensor

    #                     print(f'{list(reward_extra_infos_dict.keys())=}')
    #                     if reward_extra_infos_dict:
    #                         batch.non_tensor_batch.update({k: np.array(v) for k, v in reward_extra_infos_dict.items()})

    #                     # compute rewards. apply_kl_penalty if available
    #                     if self.config.algorithm.use_kl_in_reward:
    #                         batch, kl_metrics = apply_kl_penalty(batch,
    #                                                              kl_ctrl=self.kl_ctrl_in_reward,
    #                                                              kl_penalty=self.config.algorithm.kl_penalty)
    #                         metrics.update(kl_metrics)
    #                     else:
    #                         batch.batch['token_level_rewards'] = batch.batch['token_level_scores']

    #                     # compute advantages, executed on the driver process
    #                     batch = compute_advantage(batch,
    #                                               adv_estimator=self.config.algorithm.adv_estimator,
    #                                               gamma=self.config.algorithm.gamma,
    #                                               lam=self.config.algorithm.lam,
    #                                               num_repeat=self.config.actor_rollout_ref.rollout.n)

    #                 # update critic
    #                 if self.use_critic:
    #                     with _timer('update_critic', timing_raw):
    #                         critic_output = self.critic_wg.update_critic(batch)
    #                     critic_output_metrics = reduce_metrics(critic_output.meta_info['metrics'])
    #                     metrics.update(critic_output_metrics)

    #                 # implement critic warmup
    #                 if self.config.trainer.critic_warmup <= self.global_steps:
    #                     # update actor
    #                     with _timer('update_actor', timing_raw):
    #                         actor_output = self.actor_rollout_wg.update_actor(batch)
    #                     actor_output_metrics = reduce_metrics(actor_output.meta_info['metrics'])
    #                     metrics.update(actor_output_metrics)

    #                 # validate
    #                 if self.val_reward_fn is not None and self.config.trainer.test_freq > 0 and \
    #                     (is_last_step or  self.global_steps % self.config.trainer.test_freq == 0):
    #                     with _timer('testing', timing_raw):
    #                         val_metrics: dict = self._validate()
    #                         if is_last_step:
    #                             last_val_metrics = val_metrics
    #                     metrics.update(val_metrics)

    #                 if self.config.trainer.save_freq > 0 and ( is_last_step or \
    #                         self.global_steps % self.config.trainer.save_freq == 0):
    #                     with _timer('save_checkpoint', timing_raw):
    #                         self._save_checkpoint()

    #             # collect metrics
    #             metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
    #             metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
    #             # TODO: implement actual tflpo and theoretical tflpo
    #             n_gpus = self.resource_pool_manager.get_n_gpus()
    #             metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, n_gpus=n_gpus))

    #             # TODO: make a canonical logger that supports various backend
    #             logger.log(data=metrics, step=self.global_steps)

    #             if is_last_step:
    #                 pprint(f'Final validation metrics: {last_val_metrics}')
    #                 progress_bar.close()
    #                 return

    #             progress_bar.update(1)
    #             self.global_steps += 1

    # def fit(self):
    #     """
    #     The training loop of PPO.
    #     The driver process only need to call the compute functions of the worker group through RPC to construct the PPO dataflow.
    #     The light-weight advantage computation is done on the driver process.
    #     """
    #     from verl.utils.tracking import Tracking
    #     from omegaconf import OmegaConf

    #     logger = Tracking(project_name=self.config.trainer.project_name,
    #                       experiment_name=self.config.trainer.experiment_name,
    #                       default_backend=self.config.trainer.logger,
    #                       config=OmegaConf.to_container(self.config, resolve=True))

    #     self.global_steps = 0
    #     print("--- Initializing Fit ---") # DEBUG

    #     # load checkpoint before doing anything
    #     print("--- Loading Checkpoint ---") # DEBUG
    #     self._load_checkpoint()
    #     print(f"--- Checkpoint Loaded, Global Step: {self.global_steps} ---") # DEBUG

    #     # perform validation before training
    #     if self.val_reward_fn is not None and self.config.trainer.get('val_before_train', True):
    #         print("--- Performing Initial Validation ---") # DEBUG
    #         val_metrics = self._validate()
    #         pprint(f'Initial validation metrics: {val_metrics}')
    #         logger.log(data=val_metrics, step=self.global_steps)
    #         if self.config.trainer.get('val_only', False):
    #             print("--- val_only=True, exiting fit ---") # DEBUG
    #             return

    #     # add tqdm
    #     progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="Training Progress")

    #     # we start from step 1
    #     self.global_steps += 1
    #     last_val_metrics = None

    #     print("--- Starting Training Loop ---") # DEBUG
    #     for epoch in range(self.config.trainer.total_epochs):
    #         print(f"--- Starting Epoch {epoch+1}/{self.config.trainer.total_epochs} ---") # DEBUG
    #         for i, batch_dict in enumerate(self.train_dataloader): # Added enumeration for step count
    #             print(f"\n--- Starting Global Step {self.global_steps} (Epoch {epoch+1}, Batch {i+1}) ---") # DEBUG
    #             metrics = {}
    #             timing_raw = {}

    #             batch: DataProto = DataProto.from_single_dict(batch_dict)
    #             # DEBUG: Print initial batch keys and shapes
    #             print("[DEBUG] Initial batch keys (tensor):", list(batch.batch.keys()))
    #             print("[DEBUG] Initial batch keys (non-tensor):", list(batch.non_tensor_batch.keys()))
    #             for key, val in batch.batch.items():
    #                 if isinstance(val, torch.Tensor): print(f"[DEBUG]   {key}: {val.shape}, {val.dtype}") # DEBUG
    #                 else: print(f"[DEBUG]   {key}: type {type(val)}") # DEBUG
    #             # DEBUG: Store original prompts if available before pop
    #             original_prompts_text = batch.non_tensor_batch.get("prompts", None)
    #             original_raw_ids = batch.non_tensor_batch.get("raw_prompt_ids", None)


    #             # pop those keys for generation
    #             print("[DEBUG] Popping keys for generation...") # DEBUG
    #             if 'multi_modal_inputs' in batch.non_tensor_batch.keys():
    #                 gen_batch = batch.pop(
    #                     batch_keys=['input_ids', 'attention_mask', 'position_ids'],
    #                     non_tensor_batch_keys=['raw_prompt_ids', 'multi_modal_data', 'multi_modal_inputs'],
    #                 )
    #             else:
    #                 gen_batch = batch.pop(
    #                     batch_keys=['input_ids', 'attention_mask', 'position_ids'],
    #                     non_tensor_batch_keys=['raw_prompt_ids'],
    #                 )
    #             print("[DEBUG] gen_batch keys (tensor):", list(gen_batch.batch.keys())) # DEBUG
    #             print("[DEBUG] Remaining batch keys (tensor):", list(batch.batch.keys())) # DEBUG


    #             is_last_step = self.global_steps >= self.total_training_steps

    #             with _timer('step', timing_raw):
    #                 # generate a batch
    #                 print("[DEBUG] Generating sequences...") # DEBUG
    #                 with _timer('gen', timing_raw):
    #                     gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch)
    #                 print("[DEBUG] Sequence generation complete.") # DEBUG
    #                 print("[DEBUG] gen_batch_output keys:", list(gen_batch_output.batch.keys())) # DEBUG
    #                 for key, val in gen_batch_output.batch.items(): print(f"[DEBUG]   {key}: {val.shape}") # DEBUG


    #                 # --- REMAX Logic (if applicable) ---
    #                 # if self.config.algorithm.adv_estimator == AdvantageEstimator.REMAX:
    #                 #     print("[DEBUG] Running REMAX baseline generation...") # DEBUG
    #                 #     with _timer('gen_max', timing_raw):
    #                 #         # ... (REMAX code as before) ...
    #                 #     print("[DEBUG] REMAX baseline generation complete.") # DEBUG


    #                 # --- Prepare batch for PPO steps ---
    #                 if 'uid' not in batch.non_tensor_batch: # Add UID if not present
    #                      batch.non_tensor_batch['uid'] = np.array([str(uuid.uuid4()) for _ in range(len(batch.batch))], dtype=object)
    #                 print(f"[DEBUG] Repeating batch {self.config.actor_rollout_ref.rollout.n} times...") # DEBUG
    #                 batch = batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
    #                 print(f"[DEBUG] Batch size after repeat: {len(batch.batch)}") # DEBUG
    #                 batch = batch.union(gen_batch_output)
    #                 print("[DEBUG] Merged gen_batch_output. Batch keys:", list(batch.batch.keys())) # DEBUG
                    
    #                  # --- !!! ADDED DEBUG PRINT FOR PROMPT/OUTPUT !!! ---
    #                 if len(batch.batch) > 0: # Check if batch is not empty
    #                     try:
    #                         # Try to get original prompt text stored before pop
    #                         prompt_text_to_print = "N/A"
    #                         if original_prompts_text is not None and len(original_prompts_text) > 0:
    #                             prompt_text_to_print = original_prompts_text[0] # Get first prompt text
    #                         elif original_raw_ids is not None and len(original_raw_ids) > 0:
    #                             # Fallback: decode raw_prompt_ids stored before pop
    #                             prompt_text_to_print = self.tokenizer.decode(original_raw_ids[0], skip_special_tokens=True)

    #                         # Get generated response tokens for the first item
    #                         response_tokens = batch.batch['responses'][0]
    #                         # Decode response
    #                         decoded_response = self.tokenizer.decode(response_tokens, skip_special_tokens=True)

    #                         print("-" * 50)
    #                         print(f"[DEBUG SAMPLE] Prompt (Index 0): {prompt_text_to_print}")
    #                         print(f"[DEBUG SAMPLE] Output (Index 0): {decoded_response}")
    #                         print("-" * 50)
    #                     except Exception as e:
    #                         print(f"[DEBUG SAMPLE] Error printing sample: {e}")
    #                 # --- !!! END OF ADDED DEBUG PRINT !!! ---

    #                 batch.batch['response_mask'] = compute_response_mask(batch)
    #                 print(f"[DEBUG] response_mask shape: {batch.batch['response_mask'].shape}") # DEBUG

    #                 if self.config.trainer.balance_batch:
    #                     print("[DEBUG] Balancing batch...") # DEBUG
    #                     self._balance_batch(batch, metrics=metrics)
    #                     print("[DEBUG] Balancing complete.") # DEBUG


    #                 batch.meta_info['global_token_num'] = torch.sum(batch.batch['attention_mask'], dim=-1).tolist()

    #                 # --- Log Probs ---
    #                 print("[DEBUG] Computing old log probs...") # DEBUG
    #                 with _timer('old_log_prob', timing_raw):
    #                     old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
    #                 print("[DEBUG] Old log probs computed.") # DEBUG
    #                 # ... (entropy calculation etc. as before) ...
    #                 entropys = old_log_prob.batch['entropys']
    #                 response_masks = batch.batch['response_mask']
    #                 loss_agg_mode = self.config.actor_rollout_ref.actor.loss_agg_mode
    #                 entropy_loss = agg_loss(loss_mat=entropys, loss_mask=response_masks, loss_agg_mode=loss_agg_mode)
    #                 old_log_prob_metrics = {"actor/entropy_loss": entropy_loss.detach().item()}
    #                 metrics.update(old_log_prob_metrics)
    #                 old_log_prob.batch.pop('entropys')
    #                 batch = batch.union(old_log_prob)
    #                 print("[DEBUG] Added old_log_probs. Batch keys:", list(batch.batch.keys())) # DEBUG
    #                 print(f"[DEBUG]   old_log_probs shape: {batch.batch['old_log_probs'].shape}") # DEBUG

    #                 if self.use_reference_policy:
    #                     print("[DEBUG] Computing reference log probs...") # DEBUG
    #                     with _timer('ref', timing_raw):
    #                         ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
    #                     print("[DEBUG] Reference log probs computed.") # DEBUG
    #                     batch = batch.union(ref_log_prob)
    #                     print("[DEBUG] Added ref_log_prob. Batch keys:", list(batch.batch.keys())) # DEBUG
    #                     print(f"[DEBUG]   ref_log_prob shape: {batch.batch['ref_log_prob'].shape}") # DEBUG


    #                 # --- Compute Values (CRITICAL STEP) ---
    #                 if self.use_critic:
    #                     print(f"[DEBUG] Computing values (Critic)...") # DEBUG
    #                     # DEBUG: Print shapes of inputs needed by critic before the call
    #                     print(f"[DEBUG]   Input - batch['input_ids']: {batch.batch.get('input_ids', 'N/A')}")
    #                     print(f"[DEBUG]   Input - batch['attention_mask']: {batch.batch.get('attention_mask', 'N/A')}")
    #                     print(f"[DEBUG]   Input - batch['responses']: {batch.batch.get('responses', 'N/A')}")
    #                     print(f"[DEBUG]   Input - batch['response_mask']: {batch.batch.get('response_mask', 'N/A')}")
    #                     if 'input_ids' in batch.batch: print(f"[DEBUG]   Input - batch['input_ids'] shape: {batch.batch['input_ids'].shape}") # DEBUG
    #                     if 'attention_mask' in batch.batch: print(f"[DEBUG]   Input - batch['attention_mask'] shape: {batch.batch['attention_mask'].shape}") # DEBUG
    #                     if 'responses' in batch.batch: print(f"[DEBUG]   Input - batch['responses'] shape: {batch.batch['responses'].shape}") # DEBUG

    #                     with _timer('values', timing_raw):
    #                         # >>> THIS IS WHERE THE ERROR OCCURS <<<
    #                         values = self.critic_wg.compute_values(batch)
    #                     print("[DEBUG] Values computed.") # DEBUG
    #                     batch = batch.union(values)
    #                     print("[DEBUG] Added values. Batch keys:", list(batch.batch.keys())) # DEBUG
    #                     print(f"[DEBUG]   values shape: {batch.batch['values'].shape}") # DEBUG
    #                 # --- End Compute Values ---


    #                 # --- Advantage Computation ---
    #                 print("[DEBUG] Computing advantages...") # DEBUG
    #                 with _timer('adv', timing_raw):
    #                     # ... (RM and Reward Function logic as before) ...
    #                     if self.use_rm:
    #                         reward_tensor = self.rm_wg.compute_rm_score(batch)
    #                         batch = batch.union(reward_tensor)
    #                     try:
    #                         reward_result = self.reward_fn(batch, return_dict=True)
    #                         reward_tensor = reward_result['reward_tensor']
    #                         reward_extra_infos_dict = reward_result['reward_extra_info']
    #                     except Exception as e:
    #                         print(f'Error in reward_fn: {e}')
    #                         reward_tensor = self.reward_fn(batch); reward_extra_infos_dict = {}
    #                     batch.batch['token_level_scores'] = reward_tensor
    #                     if reward_extra_infos_dict: batch.non_tensor_batch.update({k: np.array(v) for k, v in reward_extra_infos_dict.items()})

    #                     if self.config.algorithm.use_kl_in_reward:
    #                         batch, kl_metrics = apply_kl_penalty(batch, kl_ctrl=self.kl_ctrl_in_reward, kl_penalty=self.config.algorithm.kl_penalty)
    #                         metrics.update(kl_metrics)
    #                     else: batch.batch['token_level_rewards'] = batch.batch['token_level_scores']

    #                     # --- Advantage calculation ---
    #                     batch = compute_advantage(batch, adv_estimator=self.config.algorithm.adv_estimator, gamma=self.config.algorithm.gamma, lam=self.config.algorithm.lam, num_repeat=self.config.actor_rollout_ref.rollout.n)
    #                 print("[DEBUG] Advantages computed.") # DEBUG
    #                 print(f"[DEBUG]   advantages shape: {batch.batch['advantages'].shape}") # DEBUG
    #                 print(f"[DEBUG]   returns shape: {batch.batch['returns'].shape}") # DEBUG


    #                 # --- Update Critic ---
    #                 if self.use_critic:
    #                     print("[DEBUG] Updating critic...") # DEBUG
    #                     with _timer('update_critic', timing_raw):
    #                         critic_output = self.critic_wg.update_critic(batch)
    #                     print("[DEBUG] Critic update complete.") # DEBUG
    #                     critic_output_metrics = reduce_metrics(critic_output.meta_info['metrics']); metrics.update(critic_output_metrics)

    #                 # --- Update Actor ---
    #                 if self.config.trainer.critic_warmup <= self.global_steps:
    #                     print("[DEBUG] Updating actor...") # DEBUG
    #                     with _timer('update_actor', timing_raw):
    #                          actor_output = self.actor_rollout_wg.update_actor(batch)
    #                     print("[DEBUG] Actor update complete.") # DEBUG
    #                     actor_output_metrics = reduce_metrics(actor_output.meta_info['metrics']); metrics.update(actor_output_metrics)
    #                 else:
    #                      print("[DEBUG] Skipping actor update due to critic warmup.") # DEBUG


    #                 # --- Validate ---
    #                 if self.val_reward_fn is not None and self.config.trainer.test_freq > 0 and \
    #                     (is_last_step or self.global_steps % self.config.trainer.test_freq == 0):
    #                     print("[DEBUG] Validating...") # DEBUG
    #                     with _timer('testing', timing_raw):
    #                         val_metrics: dict = self._validate()
    #                         if is_last_step: last_val_metrics = val_metrics
    #                     metrics.update(val_metrics)
    #                     print("[DEBUG] Validation complete.") # DEBUG

    #                 # --- Save Checkpoint ---
    #                 if self.config.trainer.save_freq > 0 and ( is_last_step or \
    #                         self.global_steps % self.config.trainer.save_freq == 0):
    #                     print(f"[DEBUG] Saving checkpoint for step {self.global_steps}...") # DEBUG
    #                     with _timer('save_checkpoint', timing_raw): self._save_checkpoint()
    #                     print("[DEBUG] Checkpoint saved.") # DEBUG

    #             # --- Logging ---
    #             metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
    #             metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
    #             n_gpus = self.resource_pool_manager.get_n_gpus()
    #             metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, n_gpus=n_gpus))
    #             logger.log(data=metrics, step=self.global_steps)
    #             print(f"[DEBUG] Step {self.global_steps} Metrics Logged.") # DEBUG


    #             if is_last_step:
    #                 pprint(f'Final validation metrics: {last_val_metrics}')
    #                 progress_bar.close()
    #                 print("--- Reached Last Step, Exiting Fit ---") # DEBUG
    #                 return

    #             progress_bar.update(1)
    #             self.global_steps += 1

    #         print(f"--- Finished Epoch {epoch+1} ---") # DEBUG
    #     print("--- Finished All Epochs ---") # DEBUG
    
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
            for i, batch_dict in enumerate(self.train_dataloader):
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
                            print("-" * 50); print(f"[DEBUG SAMPLE] Prompt (Index 0): {prompt_text_to_print}"); print(f"[DEBUG SAMPLE] Output (Index 0): {decoded_response}"); print("-" * 50)
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
                        # print(f"[DEBUG] Saving checkpoint for step {self.global_steps}...")
                        # with _timer('save_checkpoint', timing_raw): self._save_checkpoint()
                        # print("[DEBUG] Checkpoint saved.")

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
                    with _timer('save_checkpoint', timing_raw): self._save_checkpoint()

                progress_bar.update(1)
                self.global_steps += 1
            # End of batch loop

            if is_last_step: break # Break outer loop if last step reached

        print("--- Finished Training Loop ---") # DEBUG