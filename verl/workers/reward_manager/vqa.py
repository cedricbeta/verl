# Add this class definition to video_verl/video.py or a new reward manager file

import torch
import re
import json
import numpy as np
from collections import defaultdict
from transformers import PreTrainedTokenizer
from typing import Union, Dict, Any, Optional

from verl import DataProto
from .video import temporal_iou, tvg_format_reward, parse_time_string, tvg_accuracy_reward, tvg_compute_score # Assuming helper functions are in video.py

import logging # Add logging

logger = logging.getLogger(__name__)


class VQACombinedRewardManager:
    """
    A reward manager for two-stage Video QA tasks.

    Calculates a combined reward based on:
    1. Grounding Quality: Format correctness and temporal IoU of the Stage 1 response.
    2. QA Accuracy: Correctness of the final answer letter from the Stage 2 response.
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        grounding_weight: float = 0.5,
        qa_weight: float = 0.5,
        num_examine: int = 1
    ) -> None:
        """
        Initializes the VQACombinedRewardManager.

        Args:
            tokenizer: The tokenizer used to decode model responses.
            grounding_weight: The weight assigned to the grounding score component.
            qa_weight: The weight assigned to the QA accuracy component.
            num_examine: The number of decoded responses per batch to print for debugging.
        """
        self.tokenizer = tokenizer
        self.grounding_weight = grounding_weight
        self.qa_weight = qa_weight
        # Ensure num_examine is non-negative
        self.num_examine = max(0, num_examine)

        # Normalize weights if they don't sum to 1 (optional, but good practice)
        total_weight = self.grounding_weight + self.qa_weight
        if total_weight <= 1e-6:
             logger.warning("Grounding and QA weights sum to zero or less. Rewards will be zero.")
             self.grounding_weight = 0.0
             self.qa_weight = 0.0
        elif abs(total_weight - 1.0) > 1e-6:
             logger.warning(f"Normalizing reward weights (Grounding: {self.grounding_weight}, QA: {self.qa_weight}) to sum to 1.")
             self.grounding_weight /= total_weight
             self.qa_weight /= total_weight

        print(f"VQACombinedRewardManager initialized. Weights: Grounding={self.grounding_weight:.2f}, QA={self.qa_weight:.2f}. Will print {self.num_examine} examples.")


    def __call__(self, data: DataProto, return_dict=False) -> Union[torch.Tensor, Dict[str, Any]]:
        """
        Calculates combined VQA rewards for a batch of data.

        Expects the following in data.non_tensor_batch for each item 'i':
            - 'decoded_grounding_texts'[i]: The decoded text output from Stage 1.
            - 'start_time'[i]: Ground truth start time for grounding.
            - 'end_time'[i]: Ground truth end time for grounding.
            - 'ground_truth_answer'[i]: Ground truth answer letter for QA.

        Expects the following in data.batch:
            - 'responses': The response tensor from Stage 2 (QA output).
            - 'response_mask': Mask corresponding to the Stage 2 response tensor.
              (If not present, it assumes all tokens in 'responses' are valid).
        """
        # --- Validate Input Keys ---
        required_batch_keys = ['responses']
        required_non_tensor_keys = ['decoded_grounding_texts', 'start_time', 'end_time', 'ground_truth_answer']

        missing_batch_keys = [k for k in required_batch_keys if k not in data.batch]
        missing_non_tensor_keys = [k for k in required_non_tensor_keys if k not in data.non_tensor_batch]

        if missing_batch_keys or missing_non_tensor_keys:
             raise KeyError(f"DataProto missing required keys. Batch needs: {required_batch_keys} (Got: {list(data.batch.keys())}). Non-tensor needs: {required_non_tensor_keys} (Got: {list(data.non_tensor_batch.keys())})")

        # Check for response mask, fallback if missing
        if 'response_mask' in data.batch:
             response_masks = data.batch['response_mask'] # Shape [batch_size, stage2_response_length]
        else:
             logger.warning("VQA Reward: 'response_mask' not found in batch. Assuming all tokens in 'responses' are valid.")
             response_masks = torch.ones_like(data.batch['responses'])

        # Ensure non-tensor lists have the same length as the batch dimension
        batch_size = len(data.batch['responses'])
        for key in required_non_tensor_keys:
            if len(data.non_tensor_batch[key]) != batch_size:
                raise ValueError(f"Length mismatch: Batch size is {batch_size}, but non_tensor_batch['{key}'] has length {len(data.non_tensor_batch[key])}")


        # --- Initialize Output Structures ---
        reward_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)
        reward_extra_info = defaultdict(list)
        already_print_count = 0

        # --- Process Each Item in the Batch ---
        for i in range(batch_size):
            try:
                # Stage 2 Response (QA Answer Prediction)
                s2_response_ids = data.batch['responses'][i]
                s2_res_mask = response_masks[i]

                # Decode Stage 2 response
                valid_indices_s2 = torch.where(s2_res_mask != 0)[0]
                if len(valid_indices_s2) == 0:
                    logger.warning(f"VQA Reward: Sample {i} has an empty Stage 2 response mask. Assigning zero reward.")
                    pred_qa_answer_letter = "" # Cannot decode
                    last_token_index = 0 # Cannot determine last token index
                else:
                    valid_s2_response_ids = s2_response_ids[valid_indices_s2]
                    decoded_s2_response = self.tokenizer.decode(valid_s2_response_ids, skip_special_tokens=True)
                    # Extract predicted answer letter (simple extraction, might need refinement)
                    pred_qa_answer_letter = decoded_s2_response.strip().upper()
                    pred_qa_answer_letter = pred_qa_answer_letter[0] if pred_qa_answer_letter else ""
                    last_token_index = valid_indices_s2[-1].item()

                # Stage 1 Response (Grounding Prediction)
                s1_response_text = data.non_tensor_batch['decoded_grounding_texts'][i]

                # Ground Truth Data
                gt_start = data.non_tensor_batch['start_time'][i]
                gt_end = data.non_tensor_batch['end_time'][i]
                gt_qa_answer_letter = data.non_tensor_batch['ground_truth_answer'][i]


                # --- Calculate Grounding Reward ---
                grounding_reward = 0.0
                tvg_accuracy = 0.0
                tvg_format = 0.0
                # Only calculate if GT times are valid
                if gt_start is not None and gt_end is not None and isinstance(gt_start, (int, float)) and isinstance(gt_end, (int, float)) and gt_start <= gt_end:
                     try:
                         # Use tvg_compute_score for consistency
                         grounding_score_dict = tvg_compute_score(s1_response_text, [gt_start, gt_end])
                         grounding_reward = grounding_score_dict["score"]
                         tvg_accuracy = grounding_score_dict["tvg_accuracy"]
                         tvg_format = grounding_score_dict["tvg_format"]
                     except Exception as e:
                         logger.warning(f"VQA Reward: Error calculating grounding score for item {i}: {e}")
                else:
                    logger.warning(f"VQA Reward: Invalid ground truth times for item {i} ({gt_start}, {gt_end}). Grounding reward is 0.")


                # --- Calculate QA Reward ---
                qa_reward = 1.0 if pred_qa_answer_letter == gt_qa_answer_letter.upper() else 0.0


                # --- Combine Rewards ---
                final_score = self.grounding_weight * grounding_reward + self.qa_weight * qa_reward


                # --- Store Rewards and Metrics ---
                reward_extra_info['grounding_accuracy'].append(tvg_accuracy)
                reward_extra_info['grounding_format'].append(tvg_format)
                reward_extra_info['grounding_score'].append(grounding_reward)
                reward_extra_info['qa_accuracy'].append(qa_reward)
                reward_extra_info['final_score'].append(final_score)

                # Place the final combined score at the last token position of Stage 2 response
                if len(valid_indices_s2) > 0: # Check again in case of prior warning
                    reward_tensor[i, last_token_index] = final_score


                # --- Debug Printing ---
                if already_print_count < self.num_examine:
                    print("-" * 50)
                    print(f"VQA Reward Example {i+1}/{batch_size} (Debug Print {already_print_count + 1}/{self.num_examine})")
                    print(f"[S1 Response]: {s1_response_text}")
                    print(f"[S2 Response]: {decoded_s2_response if len(valid_indices_s2) > 0 else '<Empty Response>'}")
                    print(f"[GT Times]: [{gt_start}, {gt_end}]")
                    print(f"[GT Answer]: {gt_qa_answer_letter}")
                    print(f"[Predicted Answer]: {pred_qa_answer_letter}")
                    print(f"[Grounding Acc (IoU)]: {tvg_accuracy:.4f}")
                    print(f"[Grounding Format]: {tvg_format:.1f}")
                    print(f"[Grounding Score (Weighted {self.grounding_weight:.2f})]: {self.grounding_weight * grounding_reward:.4f}")
                    print(f"[QA Accuracy]: {qa_reward:.1f}")
                    print(f"[QA Score (Weighted {self.qa_weight:.2f})]: {self.qa_weight * qa_reward:.4f}")
                    print(f"[Final Combined Reward]: {final_score:.4f}")
                    print("-" * 50)
                    already_print_count += 1

            except KeyError as e:
                logger.error(f"VQA Reward: Error processing sample {i} due to missing key: {e}. Assigning zero reward.")
                # Append default values to maintain structure
                reward_extra_info['grounding_accuracy'].append(0.0); reward_extra_info['grounding_format'].append(0.0)
                reward_extra_info['grounding_score'].append(0.0); reward_extra_info['qa_accuracy'].append(0.0)
                reward_extra_info['final_score'].append(0.0)
            except Exception as e:
                logger.error(f"VQA Reward: Unexpected error processing sample {i}: {e}")
                import traceback; traceback.print_exc()
                # Append default values
                reward_extra_info['grounding_accuracy'].append(0.0); reward_extra_info['grounding_format'].append(0.0)
                reward_extra_info['grounding_score'].append(0.0); reward_extra_info['qa_accuracy'].append(0.0)
                reward_extra_info['final_score'].append(0.0)


        # --- Finalize Output ---
        # Convert lists in reward_extra_info to numpy arrays
        final_reward_extra_info = {k: np.array(v) for k, v in reward_extra_info.items()}

        if return_dict:
            return {
                "reward_tensor": reward_tensor,
                "reward_extra_info": final_reward_extra_info,
            }
        else:
            return reward_tensor