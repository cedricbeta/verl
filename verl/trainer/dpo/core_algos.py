
"""
Core functions to implement DPO algorithm.
The function implemented in this file should be used by trainer with different distributed strategies to
implement Online DPO
"""

import torch
import torch.nn.functional as F
from verl.utils.torch_functional import masked_mean


def compute_dpo_sigmoid_loss(policy_chosen_logps, policy_rejected_logps, 
                            ref_chosen_logps, ref_rejected_logps, 
                            beta, response_mask=None):
    """
    Compute DPO loss with sigmoid formulation.
    
    Args:
        policy_chosen_logps: log probabilities of chosen responses from policy model
        policy_rejected_logps: log probabilities of rejected responses from policy model
        ref_chosen_logps: log probabilities of chosen responses from reference model
        ref_rejected_logps: log probabilities of rejected responses from reference model
        beta: temperature parameter for DPO
        response_mask: mask for valid tokens in responses

    Returns:
        loss: DPO loss value
        metrics: dictionary of metrics
    """
    # Compute log ratios between policy and reference models
    logits = (policy_chosen_logps - ref_chosen_logps) - (policy_rejected_logps - ref_rejected_logps)
    
    if response_mask is not None:
        # Apply the response mask to consider only valid tokens
        # Sum the masked log probabilities
        policy_chosen_logps_sum = masked_mean(policy_chosen_logps, response_mask, sum_mask=True)
        policy_rejected_logps_sum = masked_mean(policy_rejected_logps, response_mask, sum_mask=True)
        ref_chosen_logps_sum = masked_mean(ref_chosen_logps, response_mask, sum_mask=True)
        ref_rejected_logps_sum = masked_mean(ref_rejected_logps, response_mask, sum_mask=True)
        
        # Compute log ratios for sequence-level probabilities
        logits = (policy_chosen_logps_sum - ref_chosen_logps_sum) - (policy_rejected_logps_sum - ref_rejected_logps_sum)
    
    # Compute sigmoid loss
    losses = -F.logsigmoid(beta * logits)
    loss = losses.mean()

    # Calculate accuracy as the percentage of pairs where chosen has higher likelihood than rejected
    accuracy = (logits > 0).float().mean().item()
    
    # Compute policy KL divergence from reference model
    chosen_kl = (policy_chosen_logps - ref_chosen_logps).mean().item()
    rejected_kl = (policy_rejected_logps - ref_rejected_logps).mean().item()
    
    metrics = {
        "dpo/loss": loss.item(),
        "dpo/accuracy": accuracy,
        "dpo/chosen_kl": chosen_kl,
        "dpo/rejected_kl": rejected_kl,
        "dpo/margin": logits.mean().item(),
    }
    
    return loss, metrics


def compute_dpo_ipo_loss(policy_chosen_logps, policy_rejected_logps, 
                         ref_chosen_logps, ref_rejected_logps, 
                         beta, response_mask=None):
    """
    Compute IPO (Regularized DPO) loss.
    
    Args:
        policy_chosen_logps: log probabilities of chosen responses from policy model
        policy_rejected_logps: log probabilities of rejected responses from policy model
        ref_chosen_logps: log probabilities of chosen responses from reference model
        ref_rejected_logps: log probabilities of rejected responses from reference model
        beta: temperature parameter for DPO
        response_mask: mask for valid tokens in responses

    Returns:
        loss: IPO loss value
        metrics: dictionary of metrics
    """
    # Compute log ratios between policy and reference models
    if response_mask is not None:
        # Apply the response mask to consider only valid tokens
        # Sum the masked log probabilities
        policy_chosen_logps_sum = masked_mean(policy_chosen_logps, response_mask, sum_mask=True)
        policy_rejected_logps_sum = masked_mean(policy_rejected_logps, response_mask, sum_mask=True)
        ref_chosen_logps_sum = masked_mean(ref_chosen_logps, response_mask, sum_mask=True)
        ref_rejected_logps_sum = masked_mean(ref_rejected_logps, response_mask, sum_mask=True)
        
        # Compute log ratios for sequence-level probabilities
        logits = (policy_chosen_logps_sum - ref_chosen_logps_sum) - (policy_rejected_logps_sum - ref_rejected_logps_sum)
    else:
        logits = (policy_chosen_logps - ref_chosen_logps) - (policy_rejected_logps - ref_rejected_logps)
    
    # Compute IPO loss: (log_ratio - target)^2
    target = 1.0 / (2.0 * beta)
    losses = (logits - target)**2
    loss = losses.mean()

    # Calculate accuracy as the percentage of pairs where chosen has higher likelihood than rejected
    accuracy = (logits > 0).float().mean().item()
    
    # Compute policy KL divergence from reference model
    chosen_kl = (policy_chosen_logps - ref_chosen_logps).mean().item()
    rejected_kl = (policy_rejected_logps - ref_rejected_logps).mean().item()
    
    metrics = {
        "dpo/loss": loss.item(),
        "dpo/accuracy": accuracy,
        "dpo/chosen_kl": chosen_kl,
        "dpo/rejected_kl": rejected_kl,
        "dpo/margin": logits.mean().item(),
    }
    
    return loss, metrics