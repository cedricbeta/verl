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

import torch
import re
import json
import numpy as np
from collections import defaultdict
from transformers import PreTrainedTokenizer
from typing import Union, Dict, Any

from verl import DataProto

# Assuming DataProto is available (imported where this class is used)
# from verl import DataProto

# --- TVG Helper Functions ---

def temporal_iou(pred_interval: list, gt_interval: list) -> float:
    """
    Calculates the Temporal Intersection over Union (IoU) between two intervals.

    Args:
        pred_interval: A list or tuple representing the predicted interval [start, end].
        gt_interval: A list or tuple representing the ground truth interval [start, end].

    Returns:
        The Temporal IoU score, between 0.0 and 1.0.
    """
    pred_start, pred_end = pred_interval
    gt_start, gt_end = gt_interval

    # Ensure intervals are valid
    if pred_start > pred_end or gt_start > gt_end:
        # Or raise an error, depending on desired behavior
        print(f"Warning: Invalid interval detected. Pred: {pred_interval}, GT: {gt_interval}")
        return 0.0

    # Calculate intersection
    intersection_start = max(pred_start, gt_start)
    intersection_end = min(pred_end, gt_end)
    intersection_duration = max(0.0, intersection_end - intersection_start)

    # Calculate union
    union_duration = (pred_end - pred_start) + (gt_end - gt_start) - intersection_duration

    # Prevent division by zero
    if union_duration <= 0:
        # If union is zero, IoU is 1 if intersection is also zero (identical empty intervals?), else 0
        return 1.0 if intersection_duration == 0 else 0.0

    iou = intersection_duration / union_duration
    return max(0.0, iou) # Ensure IoU is not negative due to float precision issues


def tvg_format_reward(predict_str: str) -> float:
    """
    Checks if the prediction string strictly matches the expected TVG format:
    <think>...</think><answer>{"start_time": "...", "end_time": "..."}</answer>

    Args:
        predict_str: The complete response string from the model.

    Returns:
        1.0 if the format matches exactly, 0.0 otherwise.
    """
    # Regex to match the exact structure including optional whitespace
    # Uses re.DOTALL to make '.' match newlines within <think> tags
    # Requires start_time and end_time values to be digits possibly with a decimal
    pattern = r'<think>.*?</think>\s*<answer>\s*{\s*"start_time"\s*:\s*"([\d.]+)"\s*,\s*"end_time"\s*:\s*"([\d.]+)"\s*}\s*</answer>'
    # Use re.fullmatch to ensure the entire string matches the pattern
    match = re.fullmatch(pattern, predict_str, re.DOTALL | re.IGNORECASE) # Added IGNORECASE for robustness
    return 1.0 if match else 0.0


def parse_time_string(time_str: str) -> float:
    """Converts MM:SS.f or SS.f string format to total seconds."""
    if ':' in time_str:
        try:
            parts = time_str.split(':')
            if len(parts) == 2: # MM:SS.f format
                minutes = float(parts[0])
                seconds = float(parts[1])
                return minutes * 60.0 + seconds
            else:
                 # Handle unexpected colon format (e.g., HH:MM:SS) if necessary,
                 # or raise error/return 0. For now, assume MM:SS.f or SS.f
                 raise ValueError(f"Unexpected time format with colons: {time_str}")
        except ValueError:
            # Reraise if conversion within parts fails
            raise ValueError(f"Could not parse MM:SS.f format: {time_str}")
    else:
        # Assume SS.f format
        return float(time_str) # This might still raise ValueError if not a valid float

def tvg_accuracy_reward(predict_str: str, ground_truth: list) -> float:
    """
    Calculates the accuracy reward based on temporal IoU using ABSOLUTE times.
    It parses the predicted timestamp from the <answer> tag and compares it
    directly with the absolute ground truth timestamp interval.

    Args:
        predict_str: The complete response string from the model.
        ground_truth: A list or tuple containing the ground truth ABSOLUTE
                      timestamp interval [start_absolute, end_absolute].
        video_length: The total length of the video in seconds (used for validation).

    Returns:
        The temporal IoU score between the predicted and ground truth intervals,
        or 0.0 if parsing fails or intervals are invalid.
    """
    

    try:
        content_answer_match = re.search(r"<answer>(.*?)</answer>", predict_str, re.DOTALL | re.IGNORECASE)
        if not content_answer_match: return 0.0
        content_answer = content_answer_match.group(1).strip()
        if not (content_answer.startswith('{') and content_answer.endswith('}')): return 0.0

        answer_data = json.loads(content_answer)
        if not isinstance(answer_data, dict): return 0.0
        if "start_time" not in answer_data or "end_time" not in answer_data: return 0.0

        start_time_str = answer_data["start_time"]
        end_time_str = answer_data["end_time"]

        # Parse time strings (handles MM:SS.f and SS.f)
        start_time_pred = parse_time_string(start_time_str)
        end_time_pred = parse_time_string(end_time_str)

        # --- Get Absolute Ground Truth Times ---
        # ground_truth argument now directly contains absolute times
        start_time_gt, end_time_gt = ground_truth

        # --- Validation ---
        # # Check predicted times are valid and within video length
        # if start_time_pred < 0 or end_time_pred < 0 or start_time_pred > end_time_pred or end_time_pred > video_length:
        #     # print(f"Warning: Invalid predicted timestamp range. Pred: [{start_time_pred}, {end_time_pred}], Len: {video_length}")
        #     return 0.0
        # # Check ground truth times are valid (relative to video_length if available)
        # if start_time_gt < 0 or end_time_gt < 0 or start_time_gt > end_time_gt or end_time_gt > video_length:
        #     # print(f"Warning: Invalid ground_truth timestamp range provided. GT: [{start_time_gt}, {end_time_gt}], Len: {video_length}")
        #     # Decide if this should be an error or just return 0
        #     return 0.0
        
        reward = temporal_iou([start_time_pred, end_time_pred], [start_time_gt, end_time_gt])

        return reward

    except json.JSONDecodeError: return 0.0
    except ValueError as e: # Catch errors from float() or parse_time_string()
        # print(f"Debug: Could not convert/parse timestamp string: {e}")
        return 0.0
    except Exception as e:
        print(f"Error calculating TVG accuracy reward: {e}")
        return 0.0
    
def tvg_compute_score(predict_str: str, ground_truth: list) -> dict:
    """
    Computes the combined TVG score by averaging accuracy and format rewards.

    Args:
        predict_str: The complete response string from the model.
        ground_truth: A list or tuple containing the ground truth normalized
                      timestamp interval [start_normalized, end_normalized].
        video_length: The total length of the video in seconds.

    Returns:
        A dictionary containing:
            'score': The combined reward (0.5 * accuracy + 0.5 * format).
            'tvg_accuracy': The raw temporal IoU score.
            'tvg_format': The format matching score (0.0 or 1.0).
    """
    acc_reward = tvg_accuracy_reward(predict_str, ground_truth)
    format_reward = tvg_format_reward(predict_str)

    # Combine the rewards (e.g., simple average)
    combined_reward = 0.5 * acc_reward + 0.5 * format_reward

    return {
        "score": combined_reward,
        "tvg_accuracy": acc_reward,
        "tvg_format": format_reward,
    }


# --- Reward Manager Class ---

class TVGRewardManager:
    """
    A reward manager specifically for Temporal Video Grounding (TVG) tasks.

    Calculates rewards based on the format correctness and temporal IoU accuracy
    of the model's predicted timestamp compared to a ground truth interval.
    """

    def __init__(self, tokenizer: PreTrainedTokenizer, num_examine: int = 1) -> None:
        """
        Initializes the TVGRewardManager.

        Args:
            tokenizer: The tokenizer used to decode model responses.
            num_examine: The number of decoded responses per batch to print for debugging.
        """
        self.tokenizer = tokenizer
        # Ensure num_examine is non-negative
        self.num_examine = max(0, num_examine)
        print(f"TVGRewardManager initialized. Will print {self.num_examine} examples per batch.")


    def __call__(self, data: DataProto, return_dict=False) -> Union[torch.Tensor, Dict[str, Any]]:
        """
        Calculates TVG rewards for a batch of data.
        MODIFIED: Correctly uses response_mask for indexing response_ids.
        """
        if 'responses' not in data.batch or 'attention_mask' not in data.batch:
             raise KeyError("DataProto batch must contain 'responses' and 'attention_mask'.")
        # --- Check if response_mask is provided, otherwise compute it ---
        if 'response_mask' in data.batch:
             response_masks = data.batch['response_mask'] # Shape [batch_size, response_length]
        else:
             # Fallback: Recompute response_mask if not present
             print("Warning: 'response_mask' not found in batch, recomputing.")
             responses = data.batch['responses']
             response_length = responses.shape[1]
             full_attention_mask = data.batch['attention_mask']
             # This assumes right padding for prompt, left padding for response within the full mask
             # Adjust if padding scheme is different
             response_masks = full_attention_mask[:, -response_length:]

        batch_size = len(data.batch['responses'])
        reward_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)
        reward_extra_info = defaultdict(list)
        already_print_count = 0

        for i in range(batch_size):
            try:
                response_ids = data.batch['responses'][i] # Shape [response_length]
                res_mask = response_masks[i]            # Shape [response_length]

                # --- MODIFICATION START: Index using response_mask ---
                # Find indices where the RESPONSE mask is non-zero
                valid_indices_in_response = torch.where(res_mask != 0)[0]

                if len(valid_indices_in_response) == 0:
                    print(f"Warning: Sample {i} has an empty response mask. Skipping.")
                    reward_extra_info['tvg_accuracy'].append(0.0)
                    reward_extra_info['tvg_format'].append(0.0)
                    reward_extra_info['tvg_combined'].append(0.0)
                    continue

                # Use these valid indices (relative to response) to get response tokens
                valid_response_ids = response_ids[valid_indices_in_response]
                # --- MODIFICATION END ---

                # Decode the valid response tokens
                # skip_special_tokens=True removes EOS, BOS, PAD etc.
                response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)

                # Retrieve ground truth and video length from non-tensor batch data
                # Use .get() for safer access, though VideoDataset should provide them
                ground_truth = data.non_tensor_batch['ground_truth'][i]
                # video_length = data.non_tensor_batch['video_length'][i]

                 # --- Calculate TVG Score ---
                score_dict = tvg_compute_score(response_str, ground_truth)
                final_reward = score_dict["score"]
                acc_reward = score_dict["tvg_accuracy"]
                format_reward = score_dict["tvg_format"]

                # Store individual components for potential analysis
                reward_extra_info['tvg_accuracy'].append(acc_reward)
                reward_extra_info['tvg_format'].append(format_reward)
                reward_extra_info['tvg_combined'].append(final_reward) # Store the combined score too

                # --- Place reward in the tensor ---
                # Place the reward at the position of the *last actual token* of the sequence
                last_token_index = valid_indices_in_response[-1].item() # Get the index of the last valid token
                reward_tensor[i, last_token_index] = final_reward

                # --- Debug Printing ---
                if already_print_count < self.num_examine:
                    print("-" * 50)
                    print(f"Example {i+1}/{batch_size} (Debug Print {already_print_count + 1}/{self.num_examine})")
                    # Optionally decode prompt too if needed for context
                    # prompt_ids = data.batch['prompts'][i] # Requires 'prompts' key
                    # prompt_str = self.tokenizer.decode(prompt_ids, skip_special_tokens=True)
                    # print(f"[Prompt]: {prompt_str}")
                    print(f"[Response]: {response_str}")
                    print(f"[Ground Truth]: {ground_truth}")
                    # print(f"[Video Length]: {video_length:.2f}s")
                    print(f"[TVG Accuracy (IoU)]: {acc_reward:.4f}")
                    print(f"[TVG Format]: {format_reward:.1f}")
                    print(f"[TVG Combined Reward]: {final_reward:.4f}")
                    print("-" * 50)
                    already_print_count += 1

            except KeyError as e:
                print(f"Error processing sample {i}: Missing expected key {e}. Ensure 'ground_truth' and 'video_length' are in non_tensor_batch.")
                # Append default values to maintain structure if possible, or re-raise
                reward_extra_info['tvg_accuracy'].append(0.0)
                reward_extra_info['tvg_format'].append(0.0)
                reward_extra_info['tvg_combined'].append(0.0)
                # Continue to next item or raise error? Continuing for now.
            except Exception as e:
                print(f"Unexpected error processing sample {i}: {e}")
                import traceback
                traceback.print_exc()
                # Append default values or re-raise
                reward_extra_info['tvg_accuracy'].append(0.0)
                reward_extra_info['tvg_format'].append(0.0)
                reward_extra_info['tvg_combined'].append(0.0)
                # Continue to next item or raise error? Continuing for now.


        # Convert lists in reward_extra_info to numpy arrays for consistency with collate_fn/DataProto expectations
        # Though the trainer might handle lists directly, converting is safer.
        final_reward_extra_info = {k: np.array(v) for k, v in reward_extra_info.items()}


        if return_dict:
            return {
                "reward_tensor": reward_tensor,
                "reward_extra_info": final_reward_extra_info,
            }
        else:
            return reward_tensor