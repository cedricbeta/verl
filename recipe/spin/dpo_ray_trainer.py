# Copyright 2024 DPO team and/or its affiliates
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
FSDP DPO Trainer with Ray-based single controller.
This trainer supports model-agonistic model initialization with huggingface
"""

import os
import statistics
import uuid
from copy import deepcopy
from pprint import pprint

import numpy as np
import torch
from omegaconf import OmegaConf, open_dict

from verl import DataProto
from verl.single_controller.ray import RayWorkerGroup
from verl.trainer.ppo.ray_trainer import RayPPOTrainer
from verl.trainer.ppo.ray_trainer import Role, WorkerType, ResourcePoolManager, reduce_metrics, _timer
from verl.trainer.ppo.metric_utils import _compute_response_info
from verl.utils.checkpoint.checkpoint_manager import find_latest_ckpt_path
from verl.utils.dataset.rl_dataset import RLHFDataset, collate_fn
from . import dpo_core_algos


def compute_pairwise_advantage(data: DataProto, config):
    """
    Compute pairwise advantages for DPO training.
    """
    if 'rm_scores' in data.batch.keys():
        # Extract response information
        responses = data.batch['responses']
        response_length = responses.size(-1)
        attention_mask = data.batch['attention_mask']
        prompt_length = data.batch['prompts'].shape[-1]
        response_mask = attention_mask[:, prompt_length:]

        # Convert response mask to group pairs of responses
        batch_size = response_mask.shape[0] // config.actor_rollout_ref.rollout.n
        preference_mask = torch.zeros(batch_size, dtype=torch.bool, device=response_mask.device)
        
        # Process each group of responses (for the same prompt)
        for i in range(batch_size):
            base_idx = i * config.actor_rollout_ref.rollout.n
            # Compare scores of response pairs and set preference
            scores = data.batch['rm_scores'][base_idx:base_idx + config.actor_rollout_ref.rollout.n].sum(dim=-1)
            if config.algorithm.get("use_ground_truth", False) and 'preferences' in data.batch:
                # Use ground truth preferences if available
                preference_mask[i] = data.batch['preferences'][i]
            else:
                # Determine best response by score
                preference_mask[i] = (scores.argmax() == 0)
        
        # Compute pairwise advantages
        data = dpo_core_algos.compute_pairwise_advantage(data, response_mask, config.actor_rollout_ref.rollout.n, config)
        
        # Add preference mask for logging and metrics
        data.batch['preference_mask'] = preference_mask
        
    else:
        raise ValueError("No reward scores found in batch data")
    
    return data


def compute_data_metrics(batch, use_critic=True):
    """
    Compute metrics from the batch data.
    """
    advantages = batch.batch['advantages']
    returns = batch.batch['returns']

    max_response_length = batch.batch['responses'].shape[-1]

    prompt_mask = batch.batch['attention_mask'][:, :-max_response_length].bool()
    response_mask = batch.batch['attention_mask'][:, -max_response_length:].bool()

    max_prompt_length = prompt_mask.size(-1)

    response_info = _compute_response_info(batch)
    prompt_length = response_info['prompt_length']
    response_length = response_info['response_length']

    valid_adv = torch.masked_select(advantages, response_mask)
    valid_returns = torch.masked_select(returns, response_mask)

    if use_critic:
        values = batch.batch['values']
        valid_values = torch.masked_select(values, response_mask)
        return_diff_var = torch.var(valid_returns - valid_values)
        return_var = torch.var(valid_returns)

    # Calculate metrics based on DPO results
    if 'preference_mask' in batch.batch:
        preference_mask = batch.batch['preference_mask']
        preference_accuracy = preference_mask.float().mean().item()
    else:
        preference_accuracy = 0.0

    metrics = {
        # DPO specific metrics
        'dpo/preference_accuracy': preference_accuracy,
        
        # advantage metrics
        'critic/advantages/mean':
            torch.mean(valid_adv).detach().item(),
        'critic/advantages/max':
            torch.max(valid_adv).detach().item(),
        'critic/advantages/min':
            torch.min(valid_adv).detach().item(),
        # returns
        'critic/returns/mean':
            torch.mean(valid_returns).detach().item(),
        'critic/returns/max':
            torch.max(valid_returns).detach().item(),
        'critic/returns/min':
            torch.min(valid_returns).detach().item(),
        **({
            # values
        # returns
        # values
            'critic/values/mean': torch.mean(valid_values).detach().item(),
            'critic/values/max': torch.max(valid_values).detach().item(),
            'critic/values/min': torch.min(valid_values).detach().item(),
            # vf explained var
            'critic/vf_explained_var': (1.0 - return_diff_var / (return_var + 1e-5)).detach().item(),
        } if use_critic else {}),

        # response length
        'response_length/mean':
            torch.mean(response_length).detach().item(),
        'response_length/max':
            torch.max(response_length).detach().item(),
        'response_length/min':
            torch.min(response_length).detach().item(),
        'response_length/clip_ratio':
            torch.mean(torch.eq(response_length, max_response_length).float()).detach().item(),
        # prompt length
        'prompt_length/mean':
            torch.mean(prompt_length).detach().item(),
        'prompt_length/max':
            torch.max(prompt_length).detach().item(),
        'prompt_length/min':
            torch.min(prompt_length).detach().item(),
        'prompt_length/clip_ratio':
            torch.mean(torch.eq(prompt_length, max_prompt_length).float()).detach().item(),
    }
    return metrics


def compute_timing_metrics(batch, timing_raw):
    response_info = _compute_response_info(batch)
    num_prompt_tokens = torch.sum(response_info['prompt_length']).item()
    num_response_tokens = torch.sum(response_info['response_length']).item()
    num_overall_tokens = num_prompt_tokens + num_response_tokens

    num_tokens_of_section = {
        'gen': num_response_tokens,
        **{
            name: num_overall_tokens for name in ['ref', 'values', 'adv', 'update_critic', 'update_actor']
        },
    }

    return {
        **{
            f'timing_s/{name}': value for name, value in timing_raw.items()
        },
        **{
            f'timing_per_token_ms/{name}': timing_raw[name] * 1000 / num_tokens_of_section[name] for name in set(num_tokens_of_section.keys(
            )) & set(timing_raw.keys())
        },
    }


class RayDPOTrainer(RayPPOTrainer):
    """
    DPO Trainer implementation that uses Ray for distributed processing.
    """

    def __init__(self,
                 config,
                 tokenizer,
                 role_worker_mapping: dict[Role, WorkerType],
                 resource_pool_manager: ResourcePoolManager,
                 ray_worker_group_cls: RayWorkerGroup = RayWorkerGroup,
                 reward_fn=None,
                 val_reward_fn=None):

        super().__init__(config, tokenizer, role_worker_mapping, resource_pool_manager, ray_worker_group_cls, reward_fn,
                         val_reward_fn)

        self.use_critic = False

    def _validate_config(self):
        super()._validate_config()
        # DPO-specific config validation
        config = self.config
        
        # Ensure necessary DPO parameters exist
        assert hasattr(config.algorithm, 'beta'), "DPO requires beta parameter in algorithm config"
        
        # Validate DPO specific parameters
        if hasattr(config.algorithm, 'loss_type'):
            assert config.algorithm.loss_type in ['sigmoid', 'ipo'], \
                f"Invalid loss_type: {config.algorithm.loss_type}. Must be 'sigmoid' or 'ipo'."

    def _create_dataloader(self):
        from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
        
        # Create train dataset
        self.train_dataset = RLHFDataset(parquet_files=self.config.data.train_files,
                                         tokenizer=self.tokenizer,
                                         prompt_key=self.config.data.prompt_key,
                                         max_prompt_length=self.config.data.max_prompt_length,
                                         return_raw_chat=self.config.data.get('return_raw_chat', False),
                                         truncation='error',
                                         filter_overlong_prompts=self.config.data.get('filter_overlong_prompts', False),
                                         num_workers=self.config.data.get('filter_overlong_prompts_workers', None))
        
        # Use sampler for better checkpoint resume
        if self.config.data.shuffle:
            train_dataloader_generator = torch.Generator()
            train_dataloader_generator.manual_seed(self.config.data.get('seed', 1))
            sampler = RandomSampler(data_source=self.train_dataset, generator=train_dataloader_generator)
        else:
            sampler = SequentialSampler(data_source=self.train_dataset)

        # Create train dataloader
        self.train_dataloader = DataLoader(dataset=self.train_dataset,
                                           batch_size=int(self.config.data.train_batch_size *
                                                          self.config.data.oversample_factor),
                                           drop_last=True,
                                           collate_fn=collate_fn,
                                           sampler=sampler)

        # Create validation dataset and dataloader
        self.val_dataset = RLHFDataset(parquet_files=self.config.data.val_files,
                                       tokenizer=self.tokenizer,
                                       prompt_key=self.config.data.prompt_key,
                                       max_prompt_length=self.config.data.max_prompt_length,
                                       return_raw_chat=self.config.data.get('return_raw_chat', False),
                                       truncation='error',
                                       filter_overlong_prompts=self.config.data.get('filter_overlong_prompts', False),
                                       num_workers=self.config.data.get('filter_overlong_prompts_workers', None))
        
        self.val_dataloader = DataLoader(dataset=self.val_dataset,
                                         batch_size=len(self.val_dataset),
                                         shuffle=True,
                                         drop_last=True,
                                         collate_fn=collate_fn)

        # Verify we have data
        assert len(self.train_dataloader) >= 1
        assert len(self.val_dataloader) >= 1

        print(f'Size of train dataloader: {len(self.train_dataloader)}')
        print(f'Size of val dataloader: {len(self.val_dataloader)}')

        # Set up total training steps
        total_training_steps = len(self.train_dataloader) * self.config.trainer.total_epochs

        if self.config.trainer.total_training_steps is not None:
            total_training_steps = self.config.trainer.total_training_steps

        self.total_training_steps = total_training_steps
        print(f'Total training steps: {self.total_training_steps}')

        # Update optimizer configurations with total steps
        OmegaConf.set_struct(self.config, True)
        with open_dict(self.config):
            self.config.actor_rollout_ref.actor.optim.total_training_steps = total_training_steps
            if hasattr(self.config, 'critic'):
                self.config.critic.optim.total_training_steps = total_training_steps

    def _save_checkpoint(self):
        """
        Save training checkpoint.
        """
        local_global_step_folder = os.path.join(self.config.trainer.default_local_dir,
                                                f'global_step_{self.global_steps}')
        print(f'local_global_step_folder: {local_global_step_folder}')
        actor_local_path = os.path.join(local_global_step_folder, 'actor')

        actor_remote_path = None if self.config.trainer.default_hdfs_dir is None else os.path.join(
            self.config.trainer.default_hdfs_dir, f'global_step_{self.global_steps}', 'actor')
        
        # Save actor model
        self.actor_rollout_wg.save_checkpoint(actor_local_path,
                                              actor_remote_path,
                                              self.global_steps,
                                              remove_previous_ckpt=self.config.trainer.remove_previous_ckpt_in_save)

        # Save reward model if enabled
        if self.use_rm:
            reward_local_path = os.path.join(local_global_step_folder, 'reward')
            reward_remote_path = None if self.config.trainer.default_hdfs_dir is None else os.path.join(
                self.config.trainer.default_hdfs_dir, f'global_step_{self.global_steps}', 'reward')
            
            self.rm_wg.save_checkpoint(reward_local_path,
                                       reward_remote_path,
                                       self.global_steps,
                                       remove_previous_ckpt=self.config.trainer.remove_previous_ckpt_in_save)

        # Save dataloader
        dataloader_local_path = os.path.join(local_global_step_folder, 'data.pt')
        import dill
        torch.save(self.train_dataloader, dataloader_local_path, pickle_module=dill)

        # Track latest checkpoint
        local_latest_checkpointed_iteration = os.path.join(self.config.trainer.default_local_dir,
                                                           'latest_checkpointed_iteration.txt')
        with open(local_latest_checkpointed_iteration, 'w') as f:
            f.write(str(self.global_steps))

    def _load_checkpoint(self):
        """
        Load training checkpoint.
        """
        if self.config.trainer.resume_mode == 'disable':
            return 0

        # Find checkpoint folder
        if self.config.trainer.default_hdfs_dir is not None:
            NotImplementedError('load from hdfs is not implemented yet')
        else:
            checkpoint_folder = self.config.trainer.default_local_dir
            if not os.path.isabs(checkpoint_folder):
                working_dir = os.getcwd()
                checkpoint_folder = os.path.join(working_dir, checkpoint_folder)
            global_step_folder = find_latest_ckpt_path(checkpoint_folder)  # None if no latest

        # Handle resume mode
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
        
        # Set global step
        self.global_steps = int(global_step_folder.split('global_step_')[-1])

        print(f'Setting global step to {self.global_steps}')
        print(f'Resuming from {global_step_folder}')

        # Load models
        actor_path = os.path.join(global_step_folder, 'actor')
        reward_path = os.path.join(global_step_folder, 'reward')
        
        # Load actor
        self.actor_rollout_wg.load_checkpoint(actor_path,
                                              del_local_after_load=self.config.trainer.del_local_ckpt_after_load)
        
        # Load reward model if used
        if self.use_rm:
            self.rm_wg.load_checkpoint(reward_path, del_local_after_load=self.config.trainer.del_local_ckpt_after_load)

        # Load dataloader state
        dataloader_local_path = os.path.join(global_step_folder, 'data.pt')
        self.train_dataloader = torch.load(dataloader_local_path)
        if isinstance(self.train_dataloader.dataset, RLHFDataset):
            self.train_dataloader.dataset.resume_dataset_state()

    def fit(self):
        """
        Main training loop for DPO.
        """
        from verl.utils.tracking import Tracking
        from omegaconf import OmegaConf

        # Setup logging
        logger = Tracking(project_name=self.config.trainer.project_name,
                          experiment_name=self.config.trainer.experiment_name,
                          default_backend=self.config.trainer.logger,
                          config=OmegaConf.to_container(self.config, resolve=True))

        self.global_steps = 0

        # Load checkpoint
        self._load_checkpoint()

        # Perform validation before training
        if self.val_reward_fn is not None and self.config.trainer.get('val_before_train', True):
            val_metrics = self._validate()
            pprint(f'Initial validation metrics: {val_metrics}')
            logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get('val_only', False):
                return

        # Start from step 1
        self.global_steps += 1

        # Main training loop
        for epoch in range(self.config.trainer.total_epochs):
            for batch_dict in self.train_dataloader:
                metrics = {}
                timing_raw = {}

                batch: DataProto = DataProto.from_single_dict(batch_dict)

                # Prepare for generation
                gen_batch = batch.pop(batch_keys=['input_ids', 'attention_mask', 'position_ids'])

                with _timer('step', timing_raw):
                    # Generate responses
                    with _timer('gen', timing_raw):
                        gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch)

                    # Process responses
                    batch.non_tensor_batch['uid'] = np.array([str(uuid.uuid4()) for _ in range(len(batch.batch))],
                                                             dtype=object)
                    
                    # Repeat to align with repeated responses in rollout
                    batch = batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
                    batch = batch.union(gen_batch_output)

                    # Track token counts
                    batch.meta_info['global_token_num'] = torch.sum(batch.batch['attention_mask'], dim=-1).tolist()

                    # Compute log probabilities
                    with _timer('old_log_prob', timing_raw):
                        old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
                        batch = batch.union(old_log_prob)

                    # Compute reference model log probabilities if needed
                    if self.use_reference_policy:
                        with _timer('ref', timing_raw):
                            ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
                            batch = batch.union(ref_log_prob)

                    # Compute reward scores
                    with _timer('reward', timing_raw):
                        if self.use_rm:
                            # Use reward model to score responses
                            reward_output = self.rm_wg.compute_rm_score(batch)
                            batch = batch.union(reward_output)
                            
                            if 'metrics' in reward_output.meta_info.keys():
                                reward_output_metrics = reduce_metrics(reward_output.meta_info['metrics'])
                                metrics.update(reward_output_metrics)

                    # Compute advantages for DPO
                    with _timer('adv', timing_raw):
                        # Compute pairwise advantages
                        batch = compute_pairwise_advantage(batch, config=self.config)

                    # Update actor
                    with _timer('update_actor', timing_raw):
                        actor_output = self.actor_rollout_wg.update_actor(batch)
                    
                    actor_output_metrics = reduce_metrics(actor_output.meta_info['metrics'])
                    metrics.update(actor_output_metrics)

                    # Update reward model if configured
                    if self.use_rm and self.config.reward_model.model.get('update', 'none') != 'none':
                        with _timer('update_rm', timing_raw):
                            rm_update_output = self.rm_wg.update_rm(batch)
                            
                            if 'metrics' in rm_update_output.meta_info.keys():
                                rm_update_metrics = reduce_metrics(rm_update_output.meta_info['metrics'])
                                metrics.update(rm_update_metrics)

                    # Perform validation if needed
                    if self.val_reward_fn is not None and self.config.trainer.test_freq > 0 and \
                        self.global_steps % self.config.trainer.test_freq == 0:
                        with _timer('testing', timing_raw):
                            val_metrics: dict = self._validate()
                        metrics.update(val_metrics)

                    # Save checkpoint if needed
                    if self.config.trainer.save_freq > 0 and \
                            self.global_steps % self.config.trainer.save_freq == 0:
                        with _timer('save_checkpoint', timing_raw):
                            self._save_checkpoint()

                # Collect and log metrics
                metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
                metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))

                logger.log(data=metrics, step=self.global_steps)

                self.global_steps += 1

                # Check if we've reached the maximum training steps
                if self.global_steps >= self.total_training_steps:
                    # Perform final validation
                    if self.val_reward_fn is not None:
                        val_metrics = self._validate()
                        pprint(f'Final validation metrics: {val_metrics}')
                        logger.log(data=val_metrics, step=self.global_steps)
                    
                    # Save final checkpoint if needed
                    if self.config.trainer.save_freq > 0 and \
                            (self.global_steps - 1) % self.config.trainer.save_freq != 0:
                        with _timer('save_checkpoint', timing_raw):
                            self._save_checkpoint()
                    return

    def filter_and_downsample(self, scores, batch: DataProto):
        """
        Downsample the batch according to oversample_factor.
        Samples passing the filters will be prioritized.
        """
        n_samples = int(self.config.actor_rollout_ref.rollout.n)
        reward_matrix = torch.tensor(scores).reshape(-1, n_samples)

        filter_mask = torch.ones((reward_matrix.shape[0]), dtype=torch.bool)

        # Apply quality filters if configured
        if self.config.data.filter_accuracy:
            acc_tensor = torch.mean(reward_matrix, dim=-1)
            filter_mask[(acc_tensor > self.config.data.accuracy_upper_bound) |
                        (acc_tensor < self.config.data.accuracy_lower_bound)] = False

        # Filter based on truncation
        if self.config.data.filter_truncate:
            length_matrix = batch.batch['attention_mask'][:, -batch.batch['responses'].shape[-1]:].sum(dim=-1).reshape(
                -1, n_samples)
            length_tensor = torch.max(length_matrix, dim=-1)[0]
            filter_mask[length_tensor >= self.config.data.max_response_length - 1] = False

        # Reorder to prioritize filtered samples
        reorder_index = torch.argsort(filter_mask, descending=True)
        reorder_index = (reorder_index.unsqueeze(-1) * n_samples + torch.arange(0, n_samples).unsqueeze(0)).view(-1)
        batch.reorder(reorder_index[:int(len(batch) //
                                         self.config.data.oversample_factor)])  # this operation is inplace

        return batch