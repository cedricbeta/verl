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

import torch
import verl
import verl.utils.torch_functional as verl_F


def compute_dpo_loss(policy_logprobs, ref_logprobs, mask, beta, loss_type="sigmoid"):
    """
    Compute the DPO loss given the policy and reference model log probabilities for chosen and rejected responses.
    
    Args:
        policy_logprobs: Policy model log probabilities, shape [batch_size * 2, seq_len]
        ref_logprobs: Reference model log probabilities, shape [batch_size * 2, seq_len]
        mask: Attention mask for completions, shape [batch_size * 2, seq_len]
        beta: Temperature parameter for DPO
        loss_type: Type of loss function ("sigmoid" or "ipo")
        
    Returns:
        Loss tensor and metrics dictionary
    """
    batch_size = policy_logprobs.shape[0] // 2
    
    # Sum log probs over sequence length, masking out padding tokens
    policy_logprobs_sum = (policy_logprobs * mask).sum(dim=1)
    ref_logprobs_sum = (ref_logprobs * mask).sum(dim=1)
    
    # Split into chosen and rejected
    chosen_policy_logprobs = policy_logprobs_sum[:batch_size]
    rejected_policy_logprobs = policy_logprobs_sum[batch_size:]
    chosen_ref_logprobs = ref_logprobs_sum[:batch_size]
    rejected_ref_logprobs = ref_logprobs_sum[batch_size:]
    
    # Compute log ratios
    policy_logratios = chosen_policy_logprobs - rejected_policy_logprobs
    ref_logratios = chosen_ref_logprobs - rejected_ref_logprobs
    
    # Final logits used for loss calculation
    logits = policy_logratios - ref_logratios
    
    # Apply the specified loss function
    if loss_type == "sigmoid":
        loss = -torch.nn.functional.logsigmoid(beta * logits).mean()
    elif loss_type == "ipo":
        loss = ((logits - 1/(2*beta))**2).mean()
    else:
        raise ValueError(f"Unsupported loss type: {loss_type}")
    
    return loss


def compute_dpo_accuracy(token_level_scores, preference_mask, response_mask, n_samples):
    """
    Compute accuracy of DPO predictions compared to preference mask.
    
    Args:
        token_level_scores: Token-level scores from policy model
        preference_mask: Boolean tensor indicating preferred responses
        response_mask: Mask for valid response tokens
        n_samples: Number of response samples per prompt
        
    Returns:
        Accuracy metric
    """
    dpo_acc = []
    for start_id in range(0, token_level_scores.shape[0], n_samples):
        cur_scores = (token_level_scores[start_id:start_id + n_samples] *
                      response_mask[start_id:start_id + n_samples]).sum(dim=1)

        def get_upper_triangle(tensor_x):
            diff_matrix = tensor_x.unsqueeze(1) - tensor_x.unsqueeze(0)
            upper_tri_indices = torch.triu(torch.ones_like(diff_matrix).bool(), diagonal=1)
            return diff_matrix[upper_tri_indices]

        cur_pref_diff = get_upper_triangle(preference_mask[start_id:start_id + n_samples].float())  # in range [0,1]
        cur_score_diff = get_upper_triangle(cur_scores)  # in R
        cur_score_prediction = (cur_score_diff > 0).float()  # in [0,1]
        
        if cur_pref_diff.abs().sum() == 0:
            cur_acc = torch.zeros_like(cur_score_prediction[0]) + 0.5
        else:
            cur_acc = (((cur_score_diff > 0) == (cur_pref_diff > 0)).float() *
                       cur_pref_diff.abs()).sum() / cur_pref_diff.abs().sum()

        dpo_acc.append(cur_acc.unsqueeze(0))

    return torch.cat(dpo_acc, dim=0).mean()


def compute_pairwise_advantage(data: verl.DataProto, response_mask: torch.Tensor, n_samples, config):
    """
    Compute pairwise advantages for DPO training.
    
    Args:
        data: Data proto containing batch data
        response_mask: Mask for valid response tokens
        n_samples: Number of response samples per prompt
        config: Configuration with algorithm parameters
        
    Returns:
        Updated data proto with advantages and returns
    """
    batch_size = data.batch['responses'].shape[0] // n_samples

    # Extract preference scores (typically from reward model or judge)
    if 'rm_scores' in data.batch.keys():
        preference_scores = data.batch['rm_scores']
    elif 'judge_preferences' in data.batch.keys():
        preference_scores = data.batch['judge_preferences']
    else:
        raise ValueError("No preference scores found in batch")

    # Reshape for pairwise comparisons
    preference_scores = preference_scores.view(batch_size, n_samples)
    response_mask_reshaped = response_mask.view(batch_size, n_samples, -1)
    
    # Compute advantages as difference from average score
    advantages = torch.zeros_like(response_mask, dtype=torch.float32)
    returns = torch.zeros_like(response_mask, dtype=torch.float32)
    
    for i in range(batch_size):
        # For each group of responses to the same prompt
        for j in range(n_samples):
            # Get valid response length
            valid_length = response_mask_reshaped[i, j].sum().item()
            if valid_length > 0:
                # Set advantage at final position based on preference score
                # This implements "whole" sequence level reward
                advantages[i * n_samples + j, valid_length - 1] = preference_scores[i, j]
                returns[i * n_samples + j, valid_length - 1] = preference_scores[i, j]
    
    # Apply token-level advantage if configured
    if config.algorithm.get("token_level_advantage", False):
        # Spread the advantage across all tokens
        for i in range(batch_size * n_samples):
            valid_length = response_mask[i].sum().item()
            if valid_length > 0:
                final_adv = advantages[i, valid_length - 1].item()
                advantages[i, :valid_length] = final_adv / valid_length
    
    data.batch['advantages'] = advantages
    data.batch['returns'] = returns
    
    return data


def compute_direct_preference_loss(policy_logprobs, ref_logprobs, preference_mask, response_mask, beta, config):
    """
    Compute direct preference optimization loss.
    
    Args:
        policy_logprobs: Policy model log probabilities
        ref_logprobs: Reference model log probabilities
        preference_mask: Boolean tensor indicating which responses are preferred
        response_mask: Mask for valid response tokens
        beta: Temperature parameter for DPO
        config: Configuration with algorithm parameters
        
    Returns:
        Loss value and metrics dictionary
    """
    batch_size = policy_logprobs.shape[0] // 2
    
    # Compute log probabilities for each response
    policy_logprobs_sum = (policy_logprobs * response_mask).sum(dim=1)
    ref_logprobs_sum = (ref_logprobs * response_mask).sum(dim=1)
    
    # Reshape to get chosen/rejected pairs
    policy_logprobs_sum = policy_logprobs_sum.view(batch_size, 2)
    ref_logprobs_sum = ref_logprobs_sum.view(batch_size, 2)
    
    # Get chosen and rejected indices based on preference mask
    chosen_idx = preference_mask.long()
    rejected_idx = 1 - chosen_idx
    
    # Extract chosen and rejected log probs
    batch_indices = torch.arange(batch_size, device=policy_logprobs.device)
    chosen_policy_logprobs = policy_logprobs_sum[batch_indices, chosen_idx]
    rejected_policy_logprobs = policy_logprobs_sum[batch_indices, rejected_idx]
    chosen_ref_logprobs = ref_logprobs_sum[batch_indices, chosen_idx]
    rejected_ref_logprobs = ref_logprobs_sum[batch_indices, rejected_idx]
    
    # Compute log ratios
    policy_logratios = chosen_policy_logprobs - rejected_policy_logprobs
    ref_logratios = chosen_ref_logprobs - rejected_ref_logprobs
    
    # Final logits
    logits = policy_logratios - ref_logratios
    
    # Apply the specified loss function
    loss_type = config.algorithm.get("loss_type", "sigmoid")
    if loss_type == "sigmoid":
        loss = -torch.nn.functional.logsigmoid(beta * logits).mean()
    elif loss_type == "ipo":
        loss = ((logits - 1/(2*beta))**2).mean()
    else:
        raise ValueError(f"Unsupported loss type: {loss_type}")
    
    # Compute metrics
    accuracy = (logits > 0).float().mean()
    
    metrics = {
        "dpo/loss": loss.item(),
        "dpo/accuracy": accuracy.item(),
        "dpo/logits_mean": logits.mean().item(),
        "dpo/logits_std": logits.std().item(),
    }
    
    return loss, metrics