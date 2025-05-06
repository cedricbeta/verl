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
Single Process Actor
"""

import itertools
from typing import Iterable, Tuple

import torch
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

from verl import DataProto
from verl.trainer.ppo.core_algos import compute_policy_loss, kl_penalty, agg_loss
from verl.workers.actor import BasePPOActor
from verl.utils.py_functional import append_to_dict
from verl.utils.torch_functional import logprobs_from_logits, masked_mean
from verl.utils.ulysses import ulysses_pad_and_slice_inputs, gather_outpus_and_unpad
from verl.utils.seqlen_balancing import rearrange_micro_batches, get_reverse_idx
import verl.utils.torch_functional as verl_F

from flash_attn.bert_padding import pad_input, unpad_input, rearrange, index_first_axis

__all__ = ['DataParallelPPOActor']


class DataParallelPPOActor(BasePPOActor):

    def __init__(self, config, actor_module: nn.Module, actor_optimizer: torch.optim.Optimizer = None):
        """When optimizer is None, it is Reference Policy"""
        super().__init__(config)
        self.actor_module = actor_module
        self.actor_optimizer = actor_optimizer
        self.use_remove_padding = self.config.get('use_remove_padding', False)
        print(f'Actor use_remove_padding={self.use_remove_padding}')
        self.ulysses_sequence_parallel_size = self.config.ulysses_sequence_parallel_size
        self.use_ulysses_sp = self.ulysses_sequence_parallel_size > 1

        self.compute_entropy_from_logits = (
            torch.compile(verl_F.entropy_from_logits, dynamic=True)
            if self.config.get('use_torch_compile', True)  #  use torch compile by default
            else verl_F.entropy_from_logits)
        
        if self.config.ppo_mini_batch_size is None:
            raise ValueError("ppo_mini_batch_size must be set in the config.")
        else: 
            print(f"DEBUG CONFIG Actor Init: self.config.ppo_mini_batch_size = {self.config.ppo_mini_batch_size}")
        


    def _forward_micro_batch(self,
                             micro_batch,
                             temperature,
                             calculate_entropy=False) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns: 
            entropy: # (bs, response_len)
            log_probs: # (bs, response_len)
        """
        response_length = micro_batch['responses'].size(-1)
        # multi_modal_inputs = {}
        # if 'multi_modal_inputs' in micro_batch:
        #     for key in micro_batch['multi_modal_inputs'][0].keys():
        #         multi_modal_inputs[key] = torch.cat([inputs[key] for inputs in micro_batch['multi_modal_inputs']],
        #                                             dim=0)
        multi_modal_inputs_processed = {} # Renamed to avoid confusion
        if 'multi_modal_inputs' in micro_batch:
            mm_inputs_in_micro_batch = micro_batch['multi_modal_inputs']
            print(f"\nDEBUG ACTOR FORWARD: Type of micro_batch['multi_modal_inputs']: {type(mm_inputs_in_micro_batch)}")
            if isinstance(mm_inputs_in_micro_batch, list) and len(mm_inputs_in_micro_batch) > 0:
                print(f"DEBUG ACTOR FORWARD: Length of list: {len(mm_inputs_in_micro_batch)}")
                first_element = mm_inputs_in_micro_batch[0]
                print(f"DEBUG ACTOR FORWARD: Type of first element: {type(first_element)}")
                if isinstance(first_element, dict):
                    print(f"DEBUG ACTOR FORWARD: Keys in first element dict: {list(first_element.keys())}")
                    for key, value in first_element.items():
                        print(f"DEBUG ACTOR FORWARD: Type of value for key '{key}' in first element: {type(value)}")
                        if isinstance(value, torch.Tensor):
                            print(f"DEBUG ACTOR FORWARD: Shape of tensor for key '{key}' in first element: {value.shape}")
                else:
                    # Print content if not a dict and not too large
                    try: print(f"DEBUG ACTOR FORWARD: First element content: {str(first_element)[:200]}...")
                    except: print("DEBUG ACTOR FORWARD: First element content: <Cannot print>")
            elif isinstance(mm_inputs_in_micro_batch, dict):
                print(f"DEBUG ACTOR FORWARD: It's a dict! Keys: {list(mm_inputs_in_micro_batch.keys())}")
                # Print info about the lists inside the dict
                for key, value_list in mm_inputs_in_micro_batch.items():
                    print(f"DEBUG ACTOR FORWARD: Key '{key}' contains type: {type(value_list)}")
                    if isinstance(value_list, list) and len(value_list) > 0:
                        print(f"DEBUG ACTOR FORWARD:   List length: {len(value_list)}, Type of first item: {type(value_list[0])}")
                        if isinstance(value_list[0], torch.Tensor):
                                print(f"DEBUG ACTOR FORWARD:     Tensor shape: {value_list[0].shape}")
                    elif isinstance(value_list, torch.Tensor):
                        print(f"DEBUG ACTOR FORWARD:   It's a Tensor! Shape: {value_list.shape}")

            # ----> The problematic loop starts below <----
            # It expects mm_inputs_in_micro_batch to be a list of dicts [{key: tensor}, ...]
            try:
                if isinstance(mm_inputs_in_micro_batch, list) and len(mm_inputs_in_micro_batch) > 0 and isinstance(mm_inputs_in_micro_batch[0], dict):
                    print("DEBUG ACTOR FORWARD: Structure appears to be list-of-dicts. Proceeding with torch.cat loop.")
                    for key in mm_inputs_in_micro_batch[0].keys():
                        # Check if the items are actually tensors before cat
                        tensors_to_cat = []
                        valid_cat = True
                        for inputs in mm_inputs_in_micro_batch:
                            item = inputs.get(key)
                            if isinstance(item, torch.Tensor):
                                tensors_to_cat.append(item)
                            else:
                                print(f"ERROR ACTOR FORWARD: Expected Tensor for key '{key}' but got {type(item)}. Cannot torch.cat.")
                                valid_cat = False
                                break # Stop trying to cat this key

                        if valid_cat and tensors_to_cat:
                            try:
                                multi_modal_inputs_processed[key] = torch.cat(tensors_to_cat, dim=0)
                                print(f"DEBUG ACTOR FORWARD: Successfully concatenated key '{key}'. Result shape: {multi_modal_inputs_processed[key].shape}")
                            except Exception as cat_err:
                                print(f"ERROR ACTOR FORWARD: torch.cat failed for key '{key}': {cat_err}")
                        elif not tensors_to_cat and valid_cat:
                            print(f"WARN ACTOR FORWARD: No tensors found to concatenate for key '{key}'")

                else:
                    print(f"ERROR ACTOR FORWARD: micro_batch['multi_modal_inputs'] is not the expected list-of-dicts. Skipping processing.")

            except Exception as loop_err:
                print(f"ERROR ACTOR FORWARD: Error during mm_inputs processing loop: {loop_err}")
                import traceback; traceback.print_exc()

        # Make sure multi_modal_inputs used later is the processed one
        multi_modal_inputs_arg = multi_modal_inputs_processed
        

        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            input_ids = micro_batch['input_ids']
            batch_size, seqlen = input_ids.shape
            attention_mask = micro_batch['attention_mask']
            position_ids = micro_batch['position_ids']
            entropy = None
            if position_ids.dim() == 3:  # qwen2vl mrope
                position_ids = position_ids.transpose(0, 1)  # (bsz, 3, seqlen) -> (3, bsz, seqlen)

            if self.use_remove_padding:
                input_ids_rmpad, indices, *_ = unpad_input(input_ids.unsqueeze(-1),
                                                           attention_mask)  # input_ids_rmpad (total_nnz, ...)
                input_ids_rmpad = input_ids_rmpad.transpose(0, 1)  # (1, total_nnz)

                # unpad the position_ids to align the rotary
                if position_ids.dim() == 3:
                    position_ids_rmpad = index_first_axis(rearrange(position_ids, "c b s ... -> (b s) c ..."),
                                                          indices).transpose(0, 1).unsqueeze(
                                                              1)  # (3, bsz, seqlen) -> (3, 1, bsz * seqlen)
                else:
                    position_ids_rmpad = index_first_axis(rearrange(position_ids.unsqueeze(-1), "b s ... -> (b s) ..."),
                                                          indices).transpose(0, 1)

                # for compute the log_prob
                input_ids_rmpad_rolled = torch.roll(input_ids_rmpad, shifts=-1, dims=1)  # (1, total_nnz)

                # pad and slice the inputs if sp > 1
                if self.use_ulysses_sp:
                    input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad_and_slice_inputs(input_ids_rmpad, \
                                                                                                position_ids_rmpad, \
                                                                                                sp_size=self.ulysses_sequence_parallel_size)
                    input_ids_rmpad_rolled, _, _ = ulysses_pad_and_slice_inputs(input_ids_rmpad_rolled, None,
                                                                                self.ulysses_sequence_parallel_size)

                input_ids_rmpad_rolled = input_ids_rmpad_rolled.squeeze(0)  # ((total_nnz / sp) + pad)

                # only pass input_ids and position_ids to enable flash_attn_varlen
                output = self.actor_module(input_ids=input_ids_rmpad,
                                           attention_mask=None,
                                           position_ids=position_ids_rmpad,
                                           **multi_modal_inputs_arg,
                                           use_cache=False)  # prevent model thinks we are generating
                logits_rmpad = output.logits.squeeze(0)  # (total_nnz, vocab_size)

                logits_rmpad.div_(temperature)

                # if use_sp: ((total_nnz / sp) + pad) ; if not use_sp: (batch, seqlen)
                inplace_backward = True
                if calculate_entropy:
                    inplace_backward = False
                log_probs = logprobs_from_logits(logits=logits_rmpad,
                                                 labels=input_ids_rmpad_rolled,
                                                 inplace_backward=inplace_backward)

                # compute entropy
                if calculate_entropy:
                    entropy_rmpad = self.compute_entropy_from_logits(logits_rmpad)  # ((total_nnz / sp) + pad)

                # gather log_prob if sp > 1
                if self.use_ulysses_sp:
                    # gather and unpad for the ulysses sp
                    log_probs = gather_outpus_and_unpad(log_probs, gather_dim=0, unpad_dim=0, padding_size=pad_size)
                    if calculate_entropy:
                        entropy_rmpad = gather_outpus_and_unpad(entropy_rmpad,
                                                                gather_dim=0,
                                                                unpad_dim=0,
                                                                padding_size=pad_size)
                # pad back to (bsz, seqlen)
                if calculate_entropy:
                    full_entropy = pad_input(hidden_states=entropy_rmpad.unsqueeze(-1),
                                             indices=indices,
                                             batch=batch_size,
                                             seqlen=seqlen)
                full_log_probs = pad_input(hidden_states=log_probs.unsqueeze(-1),
                                           indices=indices,
                                           batch=batch_size,
                                           seqlen=seqlen)

                # only return response part:
                if calculate_entropy:
                    entropy = full_entropy.squeeze(-1)[:, -response_length - 1:-1]  # (bsz, response_length)
                log_probs = full_log_probs.squeeze(-1)[:, -response_length - 1:-1]  # (bsz, response_length)

            else:  # not using rmpad and no ulysses sp
                output = self.actor_module(input_ids=input_ids,
                                           attention_mask=attention_mask,
                                           position_ids=position_ids,
                                           **multi_modal_inputs_arg,
                                           use_cache=False)  # prevent model thinks we are generating
                logits = output.logits
                logits.div_(temperature)
                logits = logits[:, -response_length - 1:-1, :]  # (bsz, response_length, vocab_size)
                log_probs = logprobs_from_logits(logits, micro_batch['responses'])
                if calculate_entropy:
                    entropy = verl_F.entropy_from_logits(logits)  # (bsz, response_length)
                    

            return entropy, log_probs

    def _optimizer_step(self):
        assert self.config.grad_clip is not None

        if isinstance(self.actor_module, FSDP):
            grad_norm = self.actor_module.clip_grad_norm_(max_norm=self.config.grad_clip)
        else:
            grad_norm = torch.nn.utils.clip_grad_norm_(self.actor_module.parameters(), max_norm=self.config.grad_clip)

        # if grad_norm is not finite, skip the update
        if not torch.isfinite(grad_norm):
            print(f"WARN: grad_norm is not finite: {grad_norm}")
            self.actor_optimizer.zero_grad()
        else:
            self.actor_optimizer.step()
        return grad_norm

    def compute_log_prob(self, data: DataProto, calculate_entropy=False) -> torch.Tensor:
        """Compute the log probability of the responses given input_ids, attention_mask and position_ids

        Args:
            data (DataProto): a DataProto containing keys

                ``input_ids``: tensor of shape [batch_size, sequence_length]. torch.int64. Note that input_ids is the
                concatenation of prompt and response. Note that ``sequence_length = prompt_length + response_length``.

                ``attention_mask``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``position_ids``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``responses``:  tensor of shape [batch_size, response_length]. torch.int64.

        Returns:
            torch.Tensor: the log_prob tensor
        """
        # set to eval
        self.actor_module.eval()

        micro_batch_size = data.meta_info['micro_batch_size']
        temperature = data.meta_info['temperature']  # temperature must be in the data.meta_info to avoid slient error
        use_dynamic_bsz = data.meta_info['use_dynamic_bsz']

        select_keys = ['responses', 'input_ids', 'attention_mask', 'position_ids']
        batch = data.select(batch_keys=select_keys).batch
        has_multi_modal_inputs = 'multi_modal_inputs' in data.non_tensor_batch.keys()
    
        
        print(f"DEBUG DP_ACTOR (compute_log_prob): data.batch.batch_size[0] = {data.batch.batch_size[0]}")
        print(f"DEBUG DP_ACTOR (compute_log_prob): micro_batch_size from meta_info = {data.meta_info.get('micro_batch_size')}")
        # If not using dynamic_bsz, num_micro_batches is calculated before chunking
        if not use_dynamic_bsz and micro_batch_size > 0 : # Add check for micro_batch_size > 0
            num_micro_batches = data.batch.batch_size[0] // micro_batch_size
            print(f"DEBUG DP_ACTOR (compute_log_prob): Calculated num_micro_batches = {num_micro_batches}")
        elif not use_dynamic_bsz and micro_batch_size == 0:
            print(f"ERROR DP_ACTOR (compute_log_prob): micro_batch_size is 0, will cause ZeroDivisionError!")

        if has_multi_modal_inputs:
            num_micro_batches = data.batch.batch_size[0] // micro_batch_size
            non_tensor_select_keys = ['multi_modal_inputs']
            micro_batches = data.select(select_keys, non_tensor_select_keys).chunk(num_micro_batches)
        elif use_dynamic_bsz:
            # split using dynamic bsz
            max_token_len = data.meta_info['max_token_len'] * self.ulysses_sequence_parallel_size
            micro_batches, indices = rearrange_micro_batches(batch=batch, max_token_len=max_token_len)
        else:
            micro_batches = batch.split(micro_batch_size)

        log_probs_lst = []
        entropy_lst = []
        for micro_batch in micro_batches:
            if isinstance(micro_batch, DataProto):
                micro_batch = {**micro_batch.batch, **micro_batch.non_tensor_batch}

            response_mask = micro_batch['attention_mask'][:, -micro_batch['responses'].size(-1):]
            with torch.no_grad():
                entropy, log_probs = self._forward_micro_batch(micro_batch,
                                                               temperature=temperature,
                                                               calculate_entropy=calculate_entropy)
            log_probs_lst.append(log_probs)
            if calculate_entropy:
                entropy_lst.append(entropy)

        log_probs = torch.concat(log_probs_lst, dim=0)
        entropys = None
        if calculate_entropy:
            entropys = torch.concat(entropy_lst, dim=0)
        if use_dynamic_bsz:
            indices = list(itertools.chain.from_iterable(indices))
            assert len(indices) == log_probs.size(0), f"{len(indices)} vs. {log_probs.size()}"
            revert_indices = torch.tensor(get_reverse_idx(indices), dtype=torch.long)
            log_probs = log_probs[revert_indices]

        return log_probs, entropys

    def update_policy(self, data: DataProto):
        # make sure we are in training mode
        self.actor_module.train()


        temperature = data.meta_info['temperature']  # temperature must be in the data.meta_info to avoid slient error

        select_keys = ['responses', 'input_ids', 'attention_mask', 'position_ids', 'old_log_probs', 'advantages']
        if self.config.use_kl_loss:
            select_keys.append('ref_log_prob')
        batch = data.select(batch_keys=select_keys).batch
        has_multi_modal_inputs = 'multi_modal_inputs' in data.non_tensor_batch.keys()
        
        current_mini_batch_size = 0
        if isinstance(data, DataProto): # 'data' here is a mini_batch
            current_mini_batch_size = data.batch.batch_size[0]
        else: # if data is a TensorDict
            current_mini_batch_size = data.batch_size[0]

        print(f"DEBUG DP_ACTOR (update_policy): current_mini_batch_size = {current_mini_batch_size}")
        print(f"DEBUG DP_ACTOR (update_policy): self.config.ppo_micro_batch_size_per_gpu = {self.config.ppo_micro_batch_size_per_gpu}")

        if self.config.ppo_micro_batch_size_per_gpu == 0:
            print(f"ERROR DP_ACTOR (update_policy): self.config.ppo_micro_batch_size_per_gpu is 0, will cause ZeroDivisionError!")
        else:
            # This is for the gradient accumulation calculation
            grad_accum_calc_num_micro_batches = current_mini_batch_size // self.config.ppo_micro_batch_size_per_gpu
            print(f"DEBUG DP_ACTOR (update_policy): Calculated num_micro_batches for grad_accum = {grad_accum_calc_num_micro_batches}")

        
        # print data keys
        # print(f"DEBUG update: data keys: {data.keys()}")
        # print(f"DEBUG update: data non_tensor_batch keys: {data.non_tensor_batch.keys()}")

        # Split to make minibatch iterator for updating the actor
        # See PPO paper for details. https://arxiv.org/abs/1707.06347
        if has_multi_modal_inputs:
            num_mini_batches = data.batch.batch_size[0] // self.config.ppo_mini_batch_size
            non_tensor_select_keys = ['multi_modal_inputs']
            dataloader = data.select(select_keys, non_tensor_select_keys).chunk(num_mini_batches)
        else:
            dataloader = batch.split(self.config.ppo_mini_batch_size)

        metrics = {}
        for epoch in range(self.config.ppo_epochs):
            for batch_idx, data in enumerate(dataloader):
                # split batch into micro_batches
                mini_batch = data
                if has_multi_modal_inputs:
                    self.gradient_accumulation = self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size_per_gpu
                    print(f"DEBUG scaling: GradAccum - gradient_accumulation = {self.gradient_accumulation}")
                    print(f"DEBUG scaling: GradAccum - mini_batch_size = {self.config.ppo_mini_batch_size}")
                    print(f"DEBUG scaling: GradAccum - micro_batch_size = {self.config.ppo_micro_batch_size_per_gpu}")
                    
                    num_micro_batches = mini_batch.batch.batch_size[0] // self.config.ppo_micro_batch_size_per_gpu
                    micro_batches = data.select(select_keys, non_tensor_select_keys).chunk(num_micro_batches)
                elif self.config.use_dynamic_bsz:
                    max_token_len = self.config.ppo_max_token_len_per_gpu * self.ulysses_sequence_parallel_size
                    micro_batches, _ = rearrange_micro_batches(batch=mini_batch, max_token_len=max_token_len)
                else:
                    self.gradient_accumulation = self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size_per_gpu
                    # split batch into micro_batches
                    micro_batches = mini_batch.split(self.config.ppo_micro_batch_size_per_gpu)

                self.actor_optimizer.zero_grad()

                for data in micro_batches:
                    # Support all hardwares
                    if isinstance(data, DataProto):
                        data = {**data.batch.to(torch.cuda.current_device()), **data.non_tensor_batch}
                    else:
                        data = data.to(torch.cuda.current_device())  # actor device is cpu when using offload
                    # responses = data['responses']
                    # response_length = responses.size(1)
                    # attention_mask = data['attention_mask']
                    # response_mask = attention_mask[:, -response_length:]
                    # old_log_prob = data['old_log_probs']
                    # advantages = data['advantages']
                    responses = data['responses']
                    # response_length = responses.size(1) # No longer needed to slice from input mask
                    # attention_mask = data['attention_mask'] # No longer used to derive response_mask here
                    response_mask = data['response_mask'] # <<< USE THE CORRECT MASK
                    old_log_prob = data['old_log_probs']
                    advantages = data['advantages']

                    clip_ratio = self.config.clip_ratio
                    clip_ratio_low = self.config.clip_ratio_low if self.config.clip_ratio_low is not None else clip_ratio
                    clip_ratio_high = self.config.clip_ratio_high if self.config.clip_ratio_high is not None else clip_ratio
                    clip_ratio_c = self.config.get('clip_ratio_c', 3.0)
                    entropy_coeff = self.config.entropy_coeff
                    loss_agg_mode = self.config.loss_agg_mode

                    # all return: (bsz, response_length)
                    calculate_entropy = False
                    if entropy_coeff != 0:
                        calculate_entropy = True
                    entropy, log_prob = self._forward_micro_batch(micro_batch=data,
                                                                  temperature=temperature,
                                                                  calculate_entropy=calculate_entropy)

                    pg_loss, pg_clipfrac, ppo_kl, pg_clipfrac_lower = compute_policy_loss(
                        old_log_prob=old_log_prob,
                        log_prob=log_prob,
                        advantages=advantages,
                        response_mask=response_mask,
                        cliprange=clip_ratio,
                        cliprange_low=clip_ratio_low,
                        cliprange_high=clip_ratio_high,
                        clip_ratio_c=clip_ratio_c,
                        loss_agg_mode=loss_agg_mode)

                    if entropy_coeff != 0:
                        entropy_loss = agg_loss(loss_mat=entropy, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)

                        # compute policy loss
                        policy_loss = pg_loss - entropy_loss * entropy_coeff
                    else:
                        policy_loss = pg_loss

                    if self.config.use_kl_loss:
                        ref_log_prob = data['ref_log_prob']
                        # compute kl loss
                        kld = kl_penalty(logprob=log_prob,
                                         ref_logprob=ref_log_prob,
                                         kl_penalty=self.config.kl_loss_type)
                        kl_loss = agg_loss(loss_mat=kld,
                                           loss_mask=response_mask,
                                           loss_agg_mode=self.config.loss_agg_mode)

                        policy_loss = policy_loss + kl_loss * self.config.kl_loss_coef
                        metrics['actor/kl_loss'] = kl_loss.detach().item()
                        metrics['actor/kl_coef'] = self.config.kl_loss_coef

                    
                    old_log_prob = data['old_log_probs']
                    advantages = data['advantages']
                    print(f"DEBUG update: old_log_prob has NaN/Inf: {torch.isnan(old_log_prob).any().item()}/{torch.isinf(old_log_prob).any().item()}")
                    print(f"DEBUG update: advantages mean/min/max: {advantages.mean().item()}/{advantages.min().item()}/{advantages.max().item()}")
                    print(f"DEBUG update: advantages has NaN/Inf: {torch.isnan(advantages).any().item()}/{torch.isinf(advantages).any().item()}")
                    if self.config.use_kl_loss:
                        ref_log_prob = data['ref_log_prob']
                        print(f"DEBUG update: ref_log_prob has NaN/Inf: {torch.isnan(ref_log_prob).any().item()}/{torch.isinf(ref_log_prob).any().item()}")

                    # After calling _forward_micro_batch
                    print(f"DEBUG update: Current log_prob has NaN/Inf: {torch.isnan(log_prob).any().item()}/{torch.isinf(log_prob).any().item()}")
                    if calculate_entropy:
                        print(f"DEBUG update: Current entropy has NaN/Inf: {torch.isnan(entropy).any().item()}/{torch.isinf(entropy).any().item()}")

                    # After calculating pg_loss, kl_loss, entropy_loss
                    print(f"DEBUG update: pg_loss: {pg_loss.item()}, pg_clipfrac: {pg_clipfrac.item()}, ppo_kl: {ppo_kl.item()}")
                    if self.config.use_kl_loss:
                        print(f"DEBUG update: kl_loss: {kl_loss.item()}")
                    if entropy_coeff != 0:
                        print(f"DEBUG update: entropy_loss: {entropy_loss.item()}")

                    print(f"DEBUG update: Final policy_loss before backward: {policy_loss.item()}")
                    if torch.isnan(policy_loss).any() or torch.isinf(policy_loss).any():
                        print("ERROR: policy_loss is NaN/Inf before backward!")
                        import pdb; pdb.set_trace() # Optional breakpoint
                        
                    if self.config.use_dynamic_bsz:
                        # relative to the dynamic bsz
                        loss = policy_loss * (len(data) / self.config.ppo_mini_batch_size)
                    else:
                        grad_accum_steps = self.gradient_accumulation
                        print(f"DEBUG scaling: GradAccum - gradient_accumulation = {grad_accum_steps}")
                        if grad_accum_steps == 0:
                            print("ERROR: gradient_accumulation is ZERO!")
                            # Handle error: maybe skip update or raise an exception
                            loss = torch.tensor(0.0, device=policy_loss.device, requires_grad=True) # Assign a dummy loss to avoid crash? Or raise error.
                        else:
                            loss = policy_loss / grad_accum_steps
                            
                    print(f"DEBUG update: Loss before backward: {loss.item()}")
                    loss.backward()
                    print("DEBUG update: Backward pass completed.")

                    data = {
                        'actor/pg_loss': pg_loss.detach().item(),
                        'actor/pg_clipfrac': pg_clipfrac.detach().item(),
                        'actor/ppo_kl': ppo_kl.detach().item(),
                        'actor/pg_clipfrac_lower': pg_clipfrac_lower.detach().item(),
                    }
                    append_to_dict(metrics, data)
                
                # ----> ADD GRADIENT CHECK HERE <----
                found_nan_grad = False
                # Note: Accessing grads with FSDP might require iterating differently
                # or using FSDP-specific methods if parameters aren't locally available.
                # Try this first, it might work depending on FSDP state.
                for name, param in self.actor_module.named_parameters():
                    if param.grad is not None:
                        if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                            print(f"ERROR: Found NaN/Inf gradient in parameter: {name}")
                            found_nan_grad = True
                    else:
                        print(f"DEBUG: Grad is None for parameter: {name}") # Less critical now

                if found_nan_grad:
                    print("ERROR: NaN/Inf gradients detected immediately after backward pass!")
                    # import pdb; pdb.set_trace() # Optional breakpoint
                else:
                    # This might be printed even if grad_norm becomes NaN later during clipping
                    print("DEBUG: Gradients seem finite immediately after backward pass.")
                # ----> END GRADIENT CHECK <----
                
                grad_norm = self._optimizer_step()
                data = {'actor/grad_norm': grad_norm.detach().item()}
            append_to_dict(metrics, data)
        self.actor_optimizer.zero_grad()
        return metrics
