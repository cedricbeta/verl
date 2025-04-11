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
Implement a multiprocess DPO reward and judge model
"""
import itertools
from typing import Iterable

import torch
import torch.distributed
from torch import nn, optim

from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

from .dpo_core_algos import compute_dpo_loss, compute_dpo_accuracy
from verl import DataProto
from verl.trainer.ppo import core_algos
from verl.workers.critic import BasePPOCritic
from verl.utils.py_functional import append_to_dict
from verl.utils.torch_functional import masked_mean
from verl.utils.ulysses import ulysses_pad_and_slice_inputs, gather_outpus_and_unpad
from verl.utils.seqlen_balancing import rearrange_micro_batches, get_reverse_idx
import verl.utils.torch_functional as verl_F

from flash_attn.bert_padding import pad_input, unpad_input, rearrange, index_first_axis

__all__ = ['DataParallelDPORewardModel']


class DataParallelDPORewardModel:
    """
    Data parallel implementation of the DPO reward model.
    """
    def __init__(self, config, reward_module: nn.Module, ref_module: nn.Module, reward_optimizer: optim.Optimizer):
        self.config = config
        self.reward_module = reward_module
        self.ref_module = ref_module
        self.reward_optimizer = reward_optimizer
        self.use_remove_padding = self.config.model.get('use_remove_padding', False)
        print(f'Reward model use_remove_padding={self.use_remove_padding}')

        self.ulysses_sequence_parallel_size = self.config.get('ulysses_sequence_parallel_size', 1)

    def _forward_micro_batch(self, micro_batch, prompt_length):
        """
        Forward pass for a micro-batch of data.
        """
        from flash_attn.bert_padding import pad_input, unpad_input, index_first_axis, rearrange
        from verl.utils.ulysses import ulysses_pad_and_slice_inputs, gather_outpus_and_unpad

        input_ids = micro_batch['input_ids']
        batch_size, seqlen = input_ids.shape
        attention_mask = micro_batch['attention_mask']
        position_ids = micro_batch['position_ids']

        num_responses = micro_batch['input_ids'].shape[-1] - prompt_length
        max_positions = micro_batch['attention_mask'][:, prompt_length:].sum(-1)

        if self.use_remove_padding:
            # Remove padding for efficient processing
            input_ids_rmpad, indices, *_ = unpad_input(input_ids.unsqueeze(-1), attention_mask)
            input_ids_rmpad = input_ids_rmpad.transpose(0, 1)  # (1, total_nnz)

            # Unpad position_ids to align the rotary
            position_ids_rmpad = index_first_axis(rearrange(position_ids.unsqueeze(-1), "b s ... -> (b s) ..."),
                                                indices).transpose(0, 1)

            # For computing log_prob
            input_ids_rmpad_rolled = torch.roll(input_ids_rmpad, shifts=-1, dims=1)  # (1, total_nnz)

            # Pad and slice the inputs if sequence parallel > 1
            if self.ulysses_sequence_parallel_size > 1:
                input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad_and_slice_inputs(
                    input_ids_rmpad, position_ids_rmpad, sp_size=self.ulysses_sequence_parallel_size)
                input_ids_rmpad_rolled, _, _ = ulysses_pad_and_slice_inputs(input_ids_rmpad_rolled, None,
                                                                        self.ulysses_sequence_parallel_size)
            input_ids_rmpad_rolled = input_ids_rmpad_rolled.squeeze(0)
            
            # Get reward model outputs
            rm_output_logits = self.reward_module(input_ids=input_ids_rmpad,
                                                attention_mask=None,
                                                position_ids=position_ids_rmpad,
                                                use_cache=False).logits.squeeze(0)
            
            # Calculate log probabilities
            rm_log_labels = verl_F.logprobs_from_logits(logits=rm_output_logits, labels=input_ids_rmpad_rolled)
            
            if self.ulysses_sequence_parallel_size > 1:
                rm_log_labels = gather_outpus_and_unpad(rm_log_labels, gather_dim=0, unpad_dim=0, padding_size=pad_size)
            
            rm_log_labels = pad_input(hidden_states=rm_log_labels.unsqueeze(-1),
                                    indices=indices,
                                    batch=batch_size,
                                    seqlen=seqlen).squeeze(-1)[:, -num_responses - 1:-1]
        else:
            # Standard forward pass without removing padding
            rm_output_logits = self.reward_module(input_ids=micro_batch['input_ids'],
                                                attention_mask=micro_batch['attention_mask'],
                                                position_ids=micro_batch['position_ids'],
                                                use_cache=False).logits
            
            rm_log_prob = torch.nn.functional.log_softmax(rm_output_logits[:, :-1, :], dim=-1)
            rm_log_labels = rm_log_prob.gather(dim=-1, index=micro_batch['input_ids'][:, 1:].unsqueeze(-1)).squeeze(-1)

        # Process reference model outputs similarly
        if self.ref_module is not None:
            with torch.no_grad(), torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                if self.ulysses_sequence_parallel_size > 1 and self.use_remove_padding:
                    ref_output_logits = self.ref_module(input_ids=input_ids_rmpad,
                                                      attention_mask=None,
                                                      position_ids=position_ids_rmpad,
                                                      use_cache=False).logits.squeeze(0)
                    ref_log_labels = verl_F.logprobs_from_logits(logits=ref_output_logits,
                                                               labels=input_ids_rmpad_rolled)
                    ref_log_labels = gather_outpus_and_unpad(ref_log_labels,
                                                           gather_dim=0,
                                                           unpad_dim=0,
                                                           padding_size=pad_size)
                    ref_log_labels = pad_input(hidden_states=ref_log_labels.unsqueeze(-1),
                                             indices=indices,
                                             batch=batch_size,
                                             seqlen=seqlen).squeeze(-1)[:, -num_responses - 1:-1]
                else:
                    ref_output_logits = self.ref_module(input_ids=micro_batch['input_ids'],
                                                      attention_mask=micro_batch['attention_mask'],
                                                      position_ids=micro_batch['position_ids'],
                                                      use_cache=False).logits
                    ref_log_prob = torch.nn.functional.log_softmax(ref_output_logits[:, :-1, :], dim=-1)
                    ref_log_labels = ref_log_prob.gather(dim=-1,
                                                       index=micro_batch['input_ids'][:, 1:].unsqueeze(-1)).squeeze(-1)
        else:
            # Use provided old_log_probs if no reference model
            ref_log_labels = micro_batch['old_log_probs']

        ref_log_labels.to(rm_log_labels.dtype)
        
        # Calculate policy-reference log probability differences
        q = rm_log_labels[:, -num_responses:] - ref_log_labels[:, -num_responses:]  # this is actually diff of q

        # Trim unnecessary logprobs
        for i in range(micro_batch['input_ids'].shape[0]):
            q[i, max_positions[i]:] = 0

        # Calculate preference scores based on configuration
        with torch.no_grad():
            beta = self.config.model.get('beta_train', 0.1)
            token_level_scores = q * beta
            
            # Apply different preference score strategies based on configuration
            if self.config.dpo_granularity == 'token':
                # Distribute preference scores across all tokens
                pass  # token_level_scores is already set correctly
            elif self.config.dpo_granularity == 'whole':
                # Place all preference at the last token
                new_scores = torch.zeros_like(q)
                for i in range(micro_batch['input_ids'].shape[0]):
                    if max_positions[i] > 0:
                        new_scores[i, max_positions[i] - 1] = token_level_scores[i, :max_positions[i]].sum()
                token_level_scores = new_scores
            else:
                raise NotImplementedError(f"Unknown DPO granularity: {self.config.dpo_granularity}")

        return token_level_scores, q

    def _optimizer_step(self):
        """
        Take an optimizer step with gradient clipping.
        """
        assert self.config.model.optim.grad_clip is not None

        if isinstance(self.reward_module, FSDP):
            grad_norm = self.reward_module.clip_grad_norm_(self.config.model.optim.grad_clip)
        else:
            grad_norm = torch.nn.utils.clip_grad_norm_(self.reward_module.parameters(),
                                                     max_norm=self.config.model.optim.grad_clip)
        self.reward_optimizer.step()
        return grad_norm

    def dpo_norm(self, token_level_scores):
        """
        Normalize preference scores based on configuration.
        """
        if self.config.dpo_norm == 'batch_norm':
            # Normalize by maximum absolute cumulative score
            reverse_cumsum = torch.cumsum(token_level_scores.flip(dims=[1]), dim=-1).flip(dims=[1])
            token_level_scores = token_level_scores / (reverse_cumsum.abs().max() + 1e-6)
        return token_level_scores

    def compute_rm_score(self, data: DataProto):
        """
        Compute preference scores using the reward model.
        """
        self.reward_module.eval()
        self.ref_module.eval()
        micro_batch_size = data.meta_info['micro_batch_size']
        select_keys = ['responses', 'input_ids', 'attention_mask', 'position_ids']
        
        # Include preference information if available
        if 'preferences' in data.batch:
            select_keys.append('preferences')
            
        batch = data.select(batch_keys=select_keys).batch
        use_dynamic_bsz = data.meta_info['use_dynamic_bsz']
        prompt_length = data.batch['input_ids'].shape[-1] - data.batch['responses'].shape[-1]

        if use_dynamic_bsz:
            # Split using dynamic batch size
            max_token_len = data.meta_info['max_token_len'] * self.ulysses_sequence_parallel_size
            micro_batches, indices = rearrange_micro_batches(batch=batch, max_token_len=max_token_len)
        else:
            micro_batches = batch.split(micro_batch_size)

        rm_scores_lst = []
        q_lst = []
        for micro_batch in micro_batches:
            with torch.no_grad():
                rm_score, q = self._forward_micro_batch(micro_batch, prompt_length)
            rm_scores_lst.append(rm_score)
            q_lst.append(q)
        rm_scores = torch.concat(rm_scores_lst, dim=0)
        q = torch.concat(q_lst, dim=0)

        rm_scores = self.dpo_norm(rm_scores)

        if use_dynamic_bsz:
            # Revert the order of samples
            indices = list(itertools.chain.from_iterable(indices))
            assert len(indices) == rm_scores.size(0), f"{len(indices)} vs. {rm_scores.size()}"
            revert_indices = torch.tensor(get_reverse_idx(indices), dtype=torch.long)
            rm_scores = rm_scores[revert_indices]

        # Calculate metrics
        metrics = {
            'reward_model/reward': rm_scores.sum(dim=-1).mean().item(),
            'reward_model/raw_reward': q.sum(dim=-1).mean().item()
        }
        
        return rm_scores, q.detach(), metrics

    def update_rm(self, data: DataProto):
        """
        Update the reward model using preference data.
        """
        self.reward_module.train()
        metrics = {}

        beta = self.config.model.get('beta_train', 0.1)

        # Select necessary keys from the data
        select_keys = ['input_ids', 'responses', 'attention_mask', 'position_ids', 'prompts']
        # Add preference data if available
        if 'preferences' in data.batch.keys():
            select_keys.append('preferences')
            
        batch = data.select(batch_keys=select_keys).batch
        
        # Split to make minibatch iterator for updating the model
        dataloader = batch.split(self.config.mini_batch_size)

        rm_scores_lst = []
        q_lst = []

        for batch_idx, data in enumerate(dataloader):
            # Split batch into micro_batches
            mini_batch = data
            if self.config.use_dynamic_bsz:
                max_token_len = self.config.dpo_max_token_len_per_gpu * self.ulysses_sequence_parallel_size
                micro_batches, _ = rearrange_micro_batches(batch=mini_batch, max_token_len=max_token_len)
            else:
                micro_batches = mini_batch.split(self.config.micro_batch_size_per_gpu)
                self.gradient_accumulation = self.config.mini_batch_size // self.config.micro_batch_size_per_gpu

            self.reward_optimizer.zero_grad()

            for data in micro_batches:
                data = data.cuda()
                attention_mask = data['attention_mask']

                prompt_ids = data['prompts']
                prompt_length = prompt_ids.shape[-1]

                response_mask = attention_mask[:, prompt_length:]

                # Forward pass
                rm_score, q = self._forward_micro_batch(data, prompt_length)

                rm_scores_lst.append(rm_score)
                q_lst.append(q.detach())

                # Calculate preference data - either from explicit preferences or from response pairs
                if 'preferences' in data:
                    preferences = data['preferences']
                else:
                    # Determine preferences based on reward score differences
                    # Assuming paired responses where every even and odd response are pairs
                    batch_size = data['input_ids'].shape[0] // 2
                    scores = q.sum(dim=-1)
                    preferences = (scores[:batch_size] >= scores[batch_size:]).float()

                # Compute DPO loss based on preferences
                loss_type = self.config.model.loss_type
                if loss_type == "sigmoid":
                    # Standard DPO loss with sigmoid
                    dpo_loss = compute_dpo_loss(
                        policy_logprobs=q, 
                        ref_logprobs=torch.zeros_like(q),  # Already subtracted in q
                        mask=response_mask, 
                        beta=beta,
                        loss_type="sigmoid"
                    )
                elif loss_type == "ipo":
                    # IPO loss (improved version of DPO)
                    dpo_loss = compute_dpo_loss(
                        policy_logprobs=q, 
                        ref_logprobs=torch.zeros_like(q),  # Already subtracted in q
                        mask=response_mask, 
                        beta=beta,
                        loss_type="ipo"
                    )
                else:
                    raise NotImplementedError(f"Unsupported loss type: {loss_type}")

                metrics_data = {'reward_model/dpo_loss': dpo_loss.detach().item()}

                # Scale loss based on batch size strategy
                if self.config.use_dynamic_bsz:
                    # Relative to the dynamic batch size
                    loss = dpo_loss * (len(data) / self.config.dpo_mini_batch_size)
                else:
                    loss = dpo_loss / self.gradient_accumulation

                loss.backward()

                append_to_dict(metrics, metrics_data)

            # Optimizer step
            grad_norm = self._optimizer_step()
            metrics_data = {'reward_model/grad_norm': grad_norm.detach().item()}
            append_to_dict(metrics, metrics_data)
            
        self.reward_optimizer.zero_grad()

        # Process results
        rm_scores = torch.cat(rm_scores_lst, dim=0)
        q = torch.concat(q_lst, dim=0)

        rm_scores = self.dpo_norm(rm_scores)

        metrics.update({
            'reward_model/reward': rm_scores.sum(dim=-1).mean().item(),
            'reward_model/raw_reward': q.sum(dim=-1).mean().item()
        })

        return rm_scores, metrics