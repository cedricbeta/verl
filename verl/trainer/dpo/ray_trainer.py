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
Online DPO Trainer with Ray-based single controller.
This trainer adapts the PPO trainer logic for DPO (Direct Preference Optimization).
"""

import os
import uuid
from contextlib import contextmanager
from enum import Enum
from pprint import pprint
from typing import Type, Dict
from copy import deepcopy
from tqdm import tqdm

import ray
import numpy as np
import torch
from codetiming import Timer
from omegaconf import OmegaConf, open_dict
from verl import DataProto
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.single_controller.base import Worker
from verl.single_controller.ray import RayResourcePool, RayWorkerGroup, RayClassWithInitArgs
from verl.single_controller.ray.base import create_colocated_worker_cls
from verl.trainer.dpo import core_algos
from verl.trainer.ppo.metric_utils import compute_throughout_metrics, compute_timing_metrics
from verl.utils.seqlen_balancing import get_seqlen_balanced_partitions, log_seqlen_unbalance
from verl.utils.checkpoint.checkpoint_manager import find_latest_ckpt_path
from verl.utils.dataset.rl_dataset import RLHFDataset, collate_fn
from verl.utils.tracking import ValidationGenerationsLogger
from torch.utils.data import RandomSampler, SequentialSampler
from torchdata.stateful_dataloader import StatefulDataLoader
from verl.utils.torch_functional import masked_mean

WorkerType = Type[Worker]


class Role(Enum):
    """
    Roles for DPO training
    """
    Actor = 0
    Rollout = 1
    ActorRollout = 2
    RefPolicy = 3
    RewardModel = 4
    Judge = 5
    ActorRolloutRef = 6


class LossType(str, Enum):
    """
    Loss types for DPO
    """
    SIGMOID = 'sigmoid'
    IPO = 'ipo'


@contextmanager
def _timer(name: str, timing_raw: Dict[str, float]):
    with Timer(name=name, logger=None) as timer:
        yield
    timing_raw[name] = timer.last


def compute_response_mask(data: DataProto):
    responses = data.batch['responses']
    response_length = responses.size(1)
    attention_mask = data.batch['attention_mask']
    return attention_mask[:, -response_length:]


def compute_dpo_metrics(data: DataProto) -> Dict[str, float]:
    """
    Compute metrics specific to DPO training.
    """
    metrics = {}
    
    if 'dpo_metrics' in data.meta_info and data.meta_info['dpo_metrics']:
        metrics.update(data.meta_info['dpo_metrics'])
        
    # Add response length metrics
    response_mask = data.batch.get('response_mask', compute_response_mask(data))
    response_length = response_mask.sum(-1).float()
    
    metrics.update({
        'response_length/mean': response_length.mean().item(),
        'response_length/max': response_length.max().item(),
        'response_length/min': response_length.min().item(),
    })
    
    return metrics



class ResourcePoolManager:
    """
    Define a resource pool specification. Resource pool will be initialized first.
    """
    def __init__(self, resource_pool_spec, mapping):
        self.resource_pool_spec = resource_pool_spec
        self.mapping = mapping
        self.resource_pool_dict = {}

    def create_resource_pool(self):
        for resource_pool_name, process_on_nodes in self.resource_pool_spec.items():
            resource_pool = RayResourcePool(process_on_nodes=process_on_nodes,
                                            use_gpu=True,
                                            max_colocate_count=1,
                                            name_prefix=resource_pool_name)
            self.resource_pool_dict[resource_pool_name] = resource_pool

        self._check_resource_available()

    def get_resource_pool(self, role: Role) -> RayResourcePool:
        return self.resource_pool_dict[self.mapping[role]]

    def get_n_gpus(self) -> int:
        return sum([n_gpus for process_on_nodes in self.resource_pool_spec.values() for n_gpus in process_on_nodes])

    def _check_resource_available(self):
        node_available_resources = ray.state.available_resources_per_node()
        node_available_gpus = {node: node_info.get('GPU', 0) for node, node_info in node_available_resources.items()}

        total_available_gpus = sum(node_available_gpus.values())
        total_required_gpus = sum(
            [n_gpus for process_on_nodes in self.resource_pool_spec.values() for n_gpus in process_on_nodes])
        if total_available_gpus < total_required_gpus:
            raise ValueError(
                f"Total available GPUs {total_available_gpus} is less than total desired GPUs {total_required_gpus}")

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


class RayDPOTrainer(object):
    """
    DPO Trainer implementation that adapts the PPO trainer.
    """
    def __init__(self,
                 config,
                 tokenizer,
                 role_worker_mapping: dict[Role, WorkerType],
                 resource_pool_manager: ResourcePoolManager,
                 ray_worker_group_cls: RayWorkerGroup = RayWorkerGroup,
                 processor=None,
                 judge_fn=None,
                 val_reward_fn=None):
        
        self.tokenizer = tokenizer
        self.processor = processor
        self.config = config
        self.judge_fn = judge_fn
        self.val_reward_fn = val_reward_fn
        
        self.hybrid_engine = config.actor_rollout_ref.hybrid_engine
        assert self.hybrid_engine, 'Currently, only support hybrid engine'
        
        self.role_worker_mapping = role_worker_mapping
        self.resource_pool_manager = resource_pool_manager
        self.use_reference_policy = Role.RefPolicy in role_worker_mapping
        self.use_rm = Role.RewardModel in role_worker_mapping
        self.use_judge = Role.Judge in role_worker_mapping
        self.ray_worker_group_cls = ray_worker_group_cls
        self.validation_generations_logger = ValidationGenerationsLogger()
        
        # Ensure reference policy is enabled for DPO
        assert self.use_reference_policy, "Reference policy is required for DPO"
        
        self._validate_config()
        self._create_dataloader()
        
    def _validate_config(self):
        config = self.config
        n_gpus = config.trainer.n_gpus_per_node * config.trainer.nnodes
        
        real_train_batch_size = config.data.train_batch_size * config.actor_rollout_ref.rollout.n
        assert real_train_batch_size % n_gpus == 0, \
            f"real_train_batch_size ({real_train_batch_size}) must be divisible by total n_gpus ({n_gpus})."
        
        assert config.actor_rollout_ref.rollout.n >= 2, \
            f"Must generate at least 2 completions per prompt for DPO (n={config.actor_rollout_ref.rollout.n})"
            
        assert hasattr(config.algorithm, 'beta'), "beta parameter must be specified for DPO"
        
        assert hasattr(config.algorithm, 'loss_type'), "loss_type must be specified for DPO"
        assert config.algorithm.loss_type in [LossType.SIGMOID, LossType.IPO], \
            f"loss_type must be one of {[lt.value for lt in LossType]}, got {config.algorithm.loss_type}"
            
        print("[validate_config] All DPO configuration checks passed successfully!")

    def _create_dataloader(self):
        self.train_dataset = RLHFDataset(parquet_files=self.config.data.train_files,
                                         tokenizer=self.tokenizer,
                                         processor=self.processor,
                                         prompt_key=self.config.data.prompt_key,
                                         image_key=self.config.data.get('image_key', 'images'),
                                         max_prompt_length=self.config.data.max_prompt_length,
                                         return_raw_chat=self.config.data.get('return_raw_chat', False),
                                         truncation=self.config.data.get('truncation', 'error'),
                                         filter_overlong_prompts=self.config.data.filter_overlong_prompts)
                                         
        if self.config.data.shuffle:
            train_dataloader_generator = torch.Generator()
            train_dataloader_generator.manual_seed(self.config.data.get('seed', 1))
            sampler = RandomSampler(data_source=self.train_dataset, generator=train_dataloader_generator)
        else:
            sampler = SequentialSampler(data_source=self.train_dataset)
            
        self.train_dataloader = StatefulDataLoader(dataset=self.train_dataset,
                                                  batch_size=self.config.data.train_batch_size,
                                                  num_workers=8,
                                                  drop_last=True,
                                                  collate_fn=collate_fn,
                                                  sampler=sampler)
                                                  
        self.val_dataset = RLHFDataset(parquet_files=self.config.data.val_files,
                                      tokenizer=self.tokenizer,
                                      processor=self.processor,
                                      prompt_key=self.config.data.prompt_key,
                                      image_key=self.config.data.get('image_key', 'images'),
                                      max_prompt_length=self.config.data.max_prompt_length,
                                      return_raw_chat=self.config.data.get('return_raw_chat', False),
                                      truncation=self.config.data.get('truncation', 'error'),
                                      filter_overlong_prompts=self.config.data.filter_overlong_prompts)
                                      
        self.val_dataloader = StatefulDataLoader(
            dataset=self.val_dataset,
            batch_size=len(self.val_dataset),
            num_workers=8,
            shuffle=False,
            drop_last=False,
            collate_fn=collate_fn)
            
        assert len(self.train_dataloader) >= 1
        assert len(self.val_dataloader) == 1, "Validation dataloader must have a single batch."
        
        print(f'Size of train dataloader: {len(self.train_dataloader)}')
        
        total_training_steps = len(self.train_dataloader) * self.config.trainer.total_epochs
        if self.config.trainer.total_training_steps is not None:
            total_training_steps = self.config.trainer.total_training_steps
            
        self.total_training_steps = total_training_steps
        print(f'Total training steps: {self.total_training_steps}')
        
        OmegaConf.set_struct(self.config, True)
        with open_dict(self.config):
            self.config.actor_rollout_ref.actor.optim.total_training_steps = total_training_steps

    def init_workers(self):
        self.resource_pool_manager.create_resource_pool()
        
        self.resource_pool_to_cls = {pool: {} for pool in self.resource_pool_manager.resource_pool_dict.values()}
        
        if self.hybrid_engine:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.ActorRollout)
            actor_rollout_cls = RayClassWithInitArgs(cls=self.role_worker_mapping[Role.ActorRollout],
                                                    config=self.config.actor_rollout_ref,
                                                    role='actor_rollout')
            self.resource_pool_to_cls[resource_pool]['actor_rollout'] = actor_rollout_cls
        else:
            raise NotImplementedError
            
        resource_pool = self.resource_pool_manager.get_resource_pool(Role.RefPolicy)
        ref_policy_cls = RayClassWithInitArgs(self.role_worker_mapping[Role.RefPolicy],
                                             config=self.config.actor_rollout_ref,
                                             role='ref')
        self.resource_pool_to_cls[resource_pool]['ref'] = ref_policy_cls
        
        if self.use_rm:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RewardModel)
            rm_cls = RayClassWithInitArgs(self.role_worker_mapping[Role.RewardModel], config=self.config.reward_model)
            self.resource_pool_to_cls[resource_pool]['rm'] = rm_cls
            
        if self.use_judge:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.Judge)
            judge_cls = RayClassWithInitArgs(self.role_worker_mapping[Role.Judge], config=self.config.judge)
            self.resource_pool_to_cls[resource_pool]['judge'] = judge_cls
            
        all_wg = {}
        self.wg_dicts = []
        for resource_pool, class_dict in self.resource_pool_to_cls.items():
            worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)
            wg_dict = self.ray_worker_group_cls(resource_pool=resource_pool, ray_cls_with_init=worker_dict_cls)
            spawn_wg = wg_dict.spawn(prefix_set=class_dict.keys())
            all_wg.update(spawn_wg)
            self.wg_dicts.append(wg_dict)
            
        self.ref_policy_wg = all_wg['ref']
        self.ref_policy_wg.init_model()
        
        if self.use_rm:
            self.rm_wg = all_wg['rm']
            self.rm_wg.init_model()
            
        if self.use_judge:
            self.judge_wg = all_wg['judge']
            self.judge_wg.init_model()
            
        self.actor_rollout_wg = all_wg['actor_rollout']
        self.actor_rollout_wg.init_model()

    def _save_checkpoint(self):
        local_global_step_folder = os.path.join(self.config.trainer.default_local_dir,
                                              f'global_step_{self.global_steps}')
        print(f'local_global_step_folder: {local_global_step_folder}')
        actor_local_path = os.path.join(local_global_step_folder, 'actor')
        actor_remote_path = None if self.config.trainer.default_hdfs_dir is None else os.path.join(
            self.config.trainer.default_hdfs_dir, f'global_step_{self.global_steps}', 'actor')
        max_actor_ckpt_to_keep = self.config.trainer.get('max_actor_ckpt_to_keep', None)
        self.actor_rollout_wg.save_checkpoint(actor_local_path,
                                            actor_remote_path,
                                            self.global_steps,
                                            max_ckpt_to_keep=max_actor_ckpt_to_keep)
                                            
        dataloader_local_path = os.path.join(local_global_step_folder, 'data.pt')
        dataloader_state_dict = self.train_dataloader.state_dict()
        torch.save(dataloader_state_dict, dataloader_local_path)
        
        local_latest_checkpointed_iteration = os.path.join(self.config.trainer.default_local_dir,
                                                         'latest_checkpointed_iteration.txt')
        with open(local_latest_checkpointed_iteration, 'w') as f:
            f.write(str(self.global_steps))

    def _load_checkpoint(self):
        if self.config.trainer.resume_mode == 'disable':
            return 0
        if self.config.trainer.default_hdfs_dir is not None:
            raise NotImplementedError('load from hdfs is not implemented yet')
        else:
            checkpoint_folder = self.config.trainer.default_local_dir
            if not os.path.isabs(checkpoint_folder):
                working_dir = os.getcwd()
                checkpoint_folder = os.path.join(working_dir, checkpoint_folder)
            global_step_folder = find_latest_ckpt_path(checkpoint_folder)
            
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
        self.global_steps = int(global_step_folder.split('global_step_')[-1])
        print(f'Setting global step to {self.global_steps}')
        print(f'Resuming from {global_step_folder}')
        
        actor_path = os.path.join(global_step_folder, 'actor')
        self.actor_rollout_wg.load_checkpoint(actor_path,
                                            del_local_after_load=self.config.trainer.del_local_ckpt_after_load)
                                            
        dataloader_local_path = os.path.join(global_step_folder, 'data.pt')
        if os.path.exists(dataloader_local_path):
            dataloader_state_dict = torch.load(dataloader_local_path, weights_only=False)
            self.train_dataloader.load_state_dict(dataloader_state_dict)
        else:
            print(f"Warning: No dataloader state found at {dataloader_local_path}, will start from scratch")

    def _balance_batch(self, batch: DataProto, metrics, logging_prefix='global_seqlen'):
        attention_mask = batch.batch['attention_mask']
        batch_size = attention_mask.shape[0]
        global_seqlen_lst = batch.batch['attention_mask'].view(batch_size, -1).sum(-1).tolist()
        world_size = self.actor_rollout_wg.world_size
        global_partition_lst = get_seqlen_balanced_partitions(global_seqlen_lst,
                                                            k_partitions=world_size,
                                                            equal_size=True)
        global_idx = torch.tensor([j for partition in global_partition_lst for j in partition])
        batch.reorder(global_idx)
        global_balance_stats = log_seqlen_unbalance(seqlen_list=global_seqlen_lst,
                                                  partitions=global_partition_lst,
                                                  prefix=logging_prefix)
        metrics.update(global_balance_stats)

    # def _rank_responses(self, batch, gen_batch_output):
    #     batch_size = batch.batch['input_ids'].shape[0]
    #     rollout_n = self.config.actor_rollout_ref.rollout.n
        
    #     responses = gen_batch_output.batch['responses']
    #     responses_grouped = responses.view(batch_size, rollout_n, -1)
        
    #     if self.use_judge and self.judge_fn:
    #         prompts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in batch.batch['input_ids']]
    #         all_responses = []
    #         for i in range(batch_size):
    #             prompt_responses = []
    #             for j in range(rollout_n):
    #                 response_text = self.tokenizer.decode(responses_grouped[i, j], skip_special_tokens=True)
    #                 prompt_responses.append(response_text)
    #             all_responses.append(prompt_responses)
    #         rankings = self.judge_fn(prompts, all_responses)
    #         chosen_indices = []
    #         rejected_indices = []
    #         for i, ranking in enumerate(rankings):
    #             sorted_indices = sorted(range(rollout_n), key=lambda j: ranking[j])
    #             chosen_indices.append(i * rollout_n + sorted_indices[0])
    #             rejected_indices.append(i * rollout_n + sorted_indices[-1])
    #     elif self.use_rm:
    #         batch_with_responses = batch.repeat(repeat_times=rollout_n, interleave=True)
    #         batch_with_responses = batch_with_responses.union(gen_batch_output)
    #         scores = self.rm_wg.compute_rm_score(batch_with_responses)
    #         scores_sum = scores.sum(dim=-1)
    #         scores_reshaped = scores_sum.view(batch_size, rollout_n)
    #         chosen_indices = []
    #         rejected_indices = []
    #         for i in range(batch_size):
    #             best_idx = torch.argmax(scores_reshaped[i]).item()
    #             worst_idx = torch.argmin(scores_reshaped[i]).item()
    #             chosen_indices.append(i * rollout_n + best_idx)
    #             rejected_indices.append(i * rollout_n + worst_idx)
    #     else:
    #         response_lengths = torch.sum(compute_response_mask(gen_batch_output), dim=1)
    #         response_lengths = response_lengths.view(batch_size, rollout_n)
    #         chosen_indices = []
    #         rejected_indices = []
    #         for i in range(batch_size):
    #             best_idx = torch.argmax(response_lengths[i]).item()
    #             worst_idx = torch.argmin(response_lengths[i]).item()
    #             if best_idx == worst_idx:
    #                 candidates = list(range(rollout_n))
    #                 candidates.remove(best_idx)
    #                 if candidates:
    #                     worst_idx = np.random.choice(candidates)
    #             chosen_indices.append(i * rollout_n + best_idx)
    #             rejected_indices.append(i * rollout_n + worst_idx)
    #     return chosen_indices, rejected_indices
    
    def _rank_responses(self, gen_batch_output):
        batch_size = gen_batch_output.batch['input_ids'].shape[0]
        rollout_n = self.config.actor_rollout_ref.rollout.n

        responses = gen_batch_output.batch['responses']
        responses_grouped = responses.view(batch_size, rollout_n, -1)

        if self.use_judge and self.judge_fn:
            prompts = [self.tokenizer.decode(ids, skip_special_tokens=True) 
                    for ids in gen_batch_output.batch['input_ids']]
            # Continue with judge ranking using prompts and responses_grouped.
            # (Implement your ranking logic here)
        elif self.use_rm:
            # Use reward model based ranking.
            batch_with_responses = gen_batch_output.repeat(repeat_times=rollout_n, interleave=True)
            scores = self.rm_wg.compute_rm_score(batch_with_responses)
            scores_sum = scores.sum(dim=-1)
            scores_reshaped = scores_sum.view(batch_size, rollout_n)
            chosen_indices = []
            rejected_indices = []
            for i in range(batch_size):
                best_idx = torch.argmax(scores_reshaped[i]).item()
                worst_idx = torch.argmin(scores_reshaped[i]).item()
                chosen_indices.append(i * rollout_n + best_idx)
                rejected_indices.append(i * rollout_n + worst_idx)
        else:
            # Default heuristic ranking: use response lengths.
            response_lengths = torch.sum(compute_response_mask(gen_batch_output), dim=1)
            response_lengths = response_lengths.view(batch_size, rollout_n)
            chosen_indices = []
            rejected_indices = []
            for i in range(batch_size):
                best_idx = torch.argmax(response_lengths[i]).item()
                worst_idx = torch.argmin(response_lengths[i]).item()
                # If same, pick another candidate.
                if best_idx == worst_idx:
                    candidates = list(range(rollout_n))
                    candidates.remove(best_idx)
                    if candidates:
                        worst_idx = np.random.choice(candidates)
                chosen_indices.append(i * rollout_n + best_idx)
                rejected_indices.append(i * rollout_n + worst_idx)
        return chosen_indices, rejected_indices

    def _validate(self):
        reward_tensor_lst = []
        data_source_lst = []
        sample_inputs = []
        sample_outputs = []
        sample_scores = []
        
        for test_data in self.val_dataloader:
            test_batch = DataProto.from_single_dict(test_data)
            test_batch = test_batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.val_kwargs.n,
                                          interleave=True)
            if self.config.reward_model.enable and test_batch[0].non_tensor_batch['reward_model']['style'] == 'model':
                return {}
            input_ids = test_batch.batch['input_ids']
            input_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in input_ids]
            sample_inputs.extend(input_texts)
            
            if 'multi_modal_inputs' in test_batch.non_tensor_batch.keys():
                test_gen_batch = test_batch.pop(
                    batch_keys=['input_ids', 'attention_mask', 'position_ids'],
                    non_tensor_batch_keys=['raw_prompt_ids', 'multi_modal_data', 'multi_modal_inputs'],
                )
            else:
                test_gen_batch = test_batch.pop(
                    batch_keys=['input_ids', 'attention_mask', 'position_ids'],
                    non_tensor_batch_keys=['raw_prompt_ids'],
                )
                
            test_gen_batch.meta_info = {
                'eos_token_id': self.tokenizer.eos_token_id,
                'pad_token_id': self.tokenizer.pad_token_id,
                'recompute_log_prob': False,
                'do_sample': self.config.actor_rollout_ref.rollout.val_kwargs.do_sample,
                'validate': True,
            }
            print(f'test_gen_batch meta info: {test_gen_batch.meta_info}')
            
            test_gen_batch_padded, pad_size = pad_dataproto_to_divisor(test_gen_batch, self.actor_rollout_wg.world_size)
            test_output_gen_batch_padded = self.actor_rollout_wg.generate_sequences(test_gen_batch_padded)
            test_output_gen_batch = unpad_dataproto(test_output_gen_batch_padded, pad_size=pad_size)
            print('validation generation end')
            
            output_ids = test_output_gen_batch.batch['responses']
            output_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in output_ids]
            sample_outputs.extend(output_texts)
            
            test_batch = test_batch.union(test_output_gen_batch)
            
            if self.val_reward_fn:
                reward_tensor = self.val_reward_fn(test_batch)
                scores = reward_tensor.sum(-1).cpu().tolist()
                sample_scores.extend(scores)
                
                reward_tensor_lst.append(reward_tensor)
                data_source_lst.append(test_batch.non_tensor_batch.get('data_source', ['unknown'] * reward_tensor.shape[0]))
            
        if self.val_reward_fn:
            self._maybe_log_val_generations(inputs=sample_inputs, outputs=sample_outputs, scores=sample_scores)
            
            reward_tensor = torch.cat(reward_tensor_lst, dim=0).sum(-1).cpu()
            data_sources = np.concatenate(data_source_lst, axis=0)
            
            data_source_reward = {}
            for i in range(reward_tensor.shape[0]):
                data_source = data_sources[i]
                if data_source not in data_source_reward:
                    data_source_reward[data_source] = []
                data_source_reward[data_source].append(reward_tensor[i].item())
                
            metric_dict = {}
            for data_source, rewards in data_source_reward.items():
                metric_dict[f'val/test_score/{data_source}'] = np.mean(rewards)
                
            return metric_dict
        
        return {}
        
    def _maybe_log_val_generations(self, inputs, outputs, scores):
        generations_to_log = self.config.trainer.log_val_generations
        if generations_to_log == 0:
            return
        import numpy as np
        samples = list(zip(inputs, outputs, scores))
        samples.sort(key=lambda x: x[0])
        rng = np.random.RandomState(42)
        rng.shuffle(samples)
        samples = samples[:generations_to_log]
        self.validation_generations_logger.log(self.config.trainer.logger, samples, self.global_steps)
        
    def fit(self):
        """
        The training loop of DPO.
        The driver process coordinates the worker groups through RPC to implement the DPO workflow.
        """
        from verl.utils.tracking import Tracking
        from omegaconf import OmegaConf
        import torch
        import psutil

        logger = Tracking(project_name=self.config.trainer.project_name,
                         experiment_name=self.config.trainer.experiment_name,
                         default_backend=self.config.trainer.logger,
                         config=OmegaConf.to_container(self.config, resolve=True))
                         
        self.global_steps = 0
        
        self._load_checkpoint()
        
        if self.val_reward_fn is not None and self.config.trainer.get('val_before_train', True):
            val_metrics = self._validate()
            pprint(f'Initial validation metrics: {val_metrics}')
            logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get('val_only', False):
                return
                
        progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="Training Progress")
        self.global_steps += 1
        last_val_metrics = None
        
        for epoch in range(self.config.trainer.total_epochs):
            for batch_dict in self.train_dataloader:
                metrics = {}
                timing_raw = {}
                
                batch: DataProto = DataProto.from_single_dict(batch_dict)
                orig_batch = DataProto.from_dict({"input_ids": batch.batch["input_ids"]})
                
                is_last_step = self.global_steps >= self.total_training_steps
                
                with _timer('step', timing_raw):
                    if 'multi_modal_inputs' in batch.non_tensor_batch.keys():
                        gen_batch = batch.pop(
                            batch_keys=['input_ids', 'attention_mask', 'position_ids'],
                            non_tensor_batch_keys=['raw_prompt_ids', 'multi_modal_data', 'multi_modal_inputs'],
                        )
                    else:
                        gen_batch = batch.pop(
                            batch_keys=['input_ids', 'attention_mask', 'position_ids'],
                            non_tensor_batch_keys=['raw_prompt_ids'],
                        )
                        
                    with _timer('gen', timing_raw):
                        gen_batch.meta_info = {
                            'eos_token_id': self.tokenizer.eos_token_id,
                            'pad_token_id': self.tokenizer.pad_token_id,
                            'recompute_log_prob': False,
                            'do_sample': True,
                            'temperature': self.config.actor_rollout_ref.rollout.temperature,
                            'n': self.config.actor_rollout_ref.rollout.n,
                        }
                        gen_batch_padded, pad_size = pad_dataproto_to_divisor(gen_batch, self.actor_rollout_wg.world_size)
                        gen_batch_output_padded = self.actor_rollout_wg.generate_sequences(gen_batch_padded)
                        gen_batch_output = unpad_dataproto(gen_batch_output_padded, pad_size=pad_size)
                        print(gen_batch_output)
                        
                    with _timer('rank', timing_raw):
                        chosen_indices, rejected_indices = self._rank_responses(gen_batch_output)
                        
                    batch_size = batch.batch['input_ids'].shape[0]
                    
                    # Build chosen and rejected batches
                    chosen_batch = batch.select(list(range(batch_size)))
                    chosen_responses = gen_batch_output.batch['responses'].select(chosen_indices)
                    chosen_batch = chosen_batch.union(DataProto.from_dict({'responses': chosen_responses}))
                    chosen_batch.batch['response_mask'] = compute_response_mask(chosen_batch)
                    
                    rejected_batch = batch.select(list(range(batch_size)))
                    rejected_responses = gen_batch_output.batch['responses'].select(rejected_indices)
                    rejected_batch = rejected_batch.union(DataProto.from_dict({'responses': rejected_responses}))
                    rejected_batch.batch['response_mask'] = compute_response_mask(rejected_batch)
                    
                    with _timer('log_probs', timing_raw):
                        policy_chosen_logps = self.actor_rollout_wg.compute_log_prob(chosen_batch)
                        policy_rejected_logps = self.actor_rollout_wg.compute_log_prob(rejected_batch)
                        ref_chosen_logps = self.ref_policy_wg.compute_ref_log_prob(chosen_batch)
                        ref_rejected_logps = self.ref_policy_wg.compute_ref_log_prob(rejected_batch)
                    
                    # 5. Update Policy using DPO update
                    # Build update batch: concatenate chosen and rejected responses.
                    # The update function expects a batch of 2*N responses with meta_info["chosen_mask"]
                    with _timer('update_policy_dpo', timing_raw):
                        update_batch = chosen_batch.union(rejected_batch)
                        # Set chosen_mask to a tensor of ones (of length N) so that the first half is treated as winning.
                        chosen_mask = torch.ones(chosen_batch.batch['responses'].shape[0], dtype=torch.bool, device=torch.cuda.current_device())
                        update_batch.meta_info["chosen_mask"] = chosen_mask
                        update_metrics = self.actor_rollout_wg.update_actor_dpo(update_batch)
                    # Merge update metrics into our metrics dict.
                    metrics.update(update_metrics.meta_info['metrics'])
                    
                    # 6. Run validation if needed
                    if self.val_reward_fn is not None and self.config.trainer.test_freq > 0 and \
                        (is_last_step or self.global_steps % self.config.trainer.test_freq == 0):
                        with _timer('validation', timing_raw):
                            val_metrics = self._validate()
                            if is_last_step:
                                last_val_metrics = val_metrics
                        metrics.update(val_metrics)
                        
                    # 7. Save checkpoint if needed
                    if self.config.trainer.save_freq > 0 and (is_last_step or \
                            self.global_steps % self.config.trainer.save_freq == 0):
                        with _timer('save_checkpoint', timing_raw):
                            self._save_checkpoint()
                
                metrics.update(compute_dpo_metrics(batch))
                metrics.update(compute_timing_metrics(batch, timing_raw))
                
                n_gpus = self.resource_pool_manager.get_n_gpus()
                metrics.update(compute_throughout_metrics(batch, timing_raw, n_gpus))
                
                logger.log(data=metrics, step=self.global_steps)
                
                if is_last_step:
                    if last_val_metrics:
                        pprint(f'Final validation metrics: {last_val_metrics}')
                    progress_bar.close()
                    return
                    
                progress_bar.update(1)
                self.global_steps += 1
