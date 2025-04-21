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
from typing import Union, Dict, Any, List, Tuple, Optional
import logging
import math # Added for interval union calculation

from verl import DataProto # Assuming DataProto is available

logger = logging.getLogger(__name__)

# --- TVG Helper Functions ---

def temporal_iou(pred_interval: list, gt_interval: list) -> float:
    """
    Calculates the Temporal Intersection over Union (IoU) between two intervals.
    Returns 0.0 for invalid inputs.
    (Unchanged)
    """
    try:
        if not (isinstance(pred_interval, (list, tuple)) and len(pred_interval) == 2 and
                isinstance(gt_interval, (list, tuple)) and len(gt_interval) == 2):
            return 0.0
        pred_start, pred_end = float(pred_interval[0]), float(pred_interval[1])
        gt_start, gt_end = float(gt_interval[0]), float(gt_interval[1])
        if pred_start > pred_end or gt_start > gt_end or pred_start < 0 or gt_start < 0: return 0.0
        intersection_start = max(pred_start, gt_start)
        intersection_end = min(pred_end, gt_end)
        intersection_duration = max(0.0, intersection_end - intersection_start)
        pred_duration = pred_end - pred_start
        gt_duration = gt_end - gt_start
        union_duration = pred_duration + gt_duration - intersection_duration
        if union_duration <= 1e-6: return 1.0 if intersection_duration <= 1e-6 else 0.0
        iou = intersection_duration / union_duration
        return max(0.0, min(iou, 1.0))
    except (ValueError, TypeError, IndexError): return 0.0

def parse_time_string(time_str: str) -> Optional[float]:
    """Converts MM:SS.f or SS.f string format to total seconds. Returns None on failure."""
    # (Unchanged)
    try:
        if not isinstance(time_str, str): return None
        if ':' in time_str:
            parts = time_str.split(':')
            if len(parts) == 2:
                minutes = float(parts[0])
                seconds = float(parts[1])
                if minutes < 0 or seconds < 0 or seconds >= 60: return None
                return minutes * 60.0 + seconds
            else: return None
        else:
            val = float(time_str)
            if val < 0: return None
            return val
    except (ValueError, TypeError): return None

def parse_multi_grounding_output(predict_str: str) -> Tuple[Optional[List[List[float]]], float]:
    """
    Parses the multi-candidate grounding part <answer>[{...}, ...]</answer>
    from the model's prediction string. Returns list of valid [start, end] pairs and format score.
    (Unchanged)
    """
    answer_match = re.search(r"<answer>(.*?)</answer>", predict_str, re.DOTALL | re.IGNORECASE)
    if not answer_match: return None, 0.0
    content_answer = answer_match.group(1).strip()
    try:
        if not (content_answer.startswith('[') and content_answer.endswith(']')): return None, 0.0
        answer_list = json.loads(content_answer)
        if not isinstance(answer_list, list): return None, 1.0
        valid_intervals = []
        for answer_data in answer_list:
            if not isinstance(answer_data, dict) or "start_time" not in answer_data or "end_time" not in answer_data: continue
            start_time_str = answer_data["start_time"]
            end_time_str = answer_data["end_time"]
            start_time_pred = parse_time_string(start_time_str)
            end_time_pred = parse_time_string(end_time_str)
            if start_time_pred is not None and end_time_pred is not None and start_time_pred <= end_time_pred:
                valid_intervals.append([start_time_pred, end_time_pred])
        return valid_intervals, 1.0
    except json.JSONDecodeError: return None, 0.0
    except Exception as e:
        logger.warning(f"Unexpected error parsing multi-grounding answer: {e}")
        return None, 0.0

# --- NEW Helper Function for Interval Union ---
def calculate_interval_union(intervals: Optional[List[List[float]]]) -> List[List[float]]:
    """
    Merges overlapping intervals to find the union.

    Args:
        intervals: A list of [start, end] pairs.

    Returns:
        A list of disjoint [start, end] pairs representing the union.
    """
    if not intervals:
        return []

    # Sort intervals based on start time
    sorted_intervals = sorted(intervals, key=lambda x: x[0])

    merged = []
    if not sorted_intervals: # Handle case after potential filtering if input was empty list
        return []

    current_start, current_end = sorted_intervals[0]

    for next_start, next_end in sorted_intervals[1:]:
        # Check for overlap or adjacency
        if next_start <= current_end:
            # Merge overlapping intervals by extending the current end
            current_end = max(current_end, next_end)
        else:
            # No overlap, finalize the current merged interval and start a new one
            merged.append([current_start, current_end])
            current_start, current_end = next_start, next_end

    # Add the last merged interval
    merged.append([current_start, current_end])

    return merged

# --- MODIFIED Compute Score Function ---
def tvg_multi_compute_score(
    predict_str: str,
    ground_truth_grounding: Optional[List[float]],
    weights: Optional[Dict[str, float]] = None
    ) -> Dict[str, Any]:
    """
    Computes the combined TVG score for MULTI-CANDIDATE grounding ONLY.
    Calculates IoU based on the UNION of predicted intervals.

    Args:
        predict_str: The complete response string from the model.
        ground_truth_grounding: List [start_gt, end_gt] for grounding.
        weights: Optional dictionary to weight reward components, e.g.,
                 {'grounding_iou_union': 0.7, 'grounding_format': 0.3}.
                 Defaults if None.

    Returns:
        A dictionary containing individual scores and the final combined score.
    """
    # Default weights for grounding only (using union IoU)
    if weights is None:
        weights = {'grounding_iou_union': 0.7, 'grounding_format': 0.3} # Key changed
        total_weight = sum(weights.values())
        if total_weight > 0: weights = {k: v / total_weight for k, v in weights.items()}
        else: weights = {k: 0.0 for k in weights}

    # --- Parse Model Output ---
    predicted_grounding_list, grounding_format_score = parse_multi_grounding_output(predict_str)

    # --- Calculate IoU based on Union ---
    iou_union_score = 0.0
    predicted_union_intervals = [] # Store the calculated union for logging
    if predicted_grounding_list is not None and ground_truth_grounding is not None:
        # 1. Calculate the union of predicted intervals
        predicted_union_intervals = calculate_interval_union(predicted_grounding_list)

        # 2. Calculate total duration of the predicted union
        total_duration_predicted_union = sum(end - start for start, end in predicted_union_intervals)

        # 3. Calculate duration of ground truth
        gt_start, gt_end = ground_truth_grounding
        duration_ground_truth = gt_end - gt_start

        # 4. Calculate total intersection between predicted union and ground truth
        total_intersection = 0.0
        for pred_start, pred_end in predicted_union_intervals:
            intersection_start = max(pred_start, gt_start)
            intersection_end = min(pred_end, gt_end)
            intersection_duration = max(0.0, intersection_end - intersection_start)
            total_intersection += intersection_duration

        # 5. Calculate IoU
        total_union_measure = total_duration_predicted_union + duration_ground_truth - total_intersection
        if total_union_measure > 1e-6:
            iou_union_score = total_intersection / total_union_measure
        elif total_intersection <= 1e-6: # If both union and intersection are zero
             iou_union_score = 1.0
        # else: iou_union_score remains 0.0

        iou_union_score = max(0.0, min(iou_union_score, 1.0)) # Clamp

    # --- Combine Scores (Grounding Only) ---
    combined_score = (
        weights.get('grounding_iou_union', 0.0) * iou_union_score + # Use union IoU score
        weights.get('grounding_format', 0.0) * grounding_format_score
    )

    combined_score = max(0.0, min(combined_score, 1.0)) # Clamp score

    return {
        "score": combined_score,
        "grounding_iou_union": iou_union_score, # Changed key
        "grounding_format": grounding_format_score,
        "predicted_grounding_candidates": predicted_grounding_list, # Original parsed list
        "predicted_grounding_union": predicted_union_intervals, # Calculated union
        # "best_predicted_grounding": best_predicted_grounding, # No longer relevant
    }


# --- Reward Manager Class ---
class TVGRewardManager:
    """
    Reward manager for multi-candidate Temporal Video Grounding (TVG) ONLY.
    Uses IoU based on the union of predicted intervals.
    """
    def __init__(self, tokenizer: PreTrainedTokenizer, num_examine: int = 1, reward_weights: Optional[Dict[str, float]] = None) -> None:
        """
        Initializes the TVGRewardManager.
        Args:
            tokenizer: The tokenizer used to decode model responses.
            num_examine: Number of decoded responses per batch to print for debugging.
            reward_weights: Optional dictionary to weight reward components ('grounding_iou_union', 'grounding_format').
        """
        self.tokenizer = tokenizer
        self.num_examine = max(0, num_examine)
        # Use weights passed from config, or None to use defaults in compute_score
        self.reward_weights = reward_weights
        logger.info(f"TVGRewardManager (multi-candidate, union IoU) initialized. num_examine={self.num_examine}, reward_weights={self.reward_weights}")

    def __call__(self, data: DataProto, return_dict=False) -> Union[torch.Tensor, Dict[str, Any]]:
        """
        Calculates multi-candidate TVG rewards (using union IoU) for a batch of data.
        """
        # --- Input Validation ---
        required_batch_keys = ['responses', 'attention_mask']
        required_nontensor_keys = ['ground_truth_grounding']
        if not all(key in data.batch for key in required_batch_keys):
             missing = [key for key in required_batch_keys if key not in data.batch]; raise KeyError(f"Batch missing keys: {missing}")
        if not all(key in data.non_tensor_batch for key in required_nontensor_keys):
             missing = [key for key in required_nontensor_keys if key not in data.non_tensor_batch]; logger.warning(f"Non_tensor_batch missing keys: {missing}.")

        # --- Determine Response Mask ---
        if 'response_mask' in data.batch: response_masks = data.batch['response_mask']
        else:
             logger.warning("'response_mask' not found, recomputing."); responses = data.batch['responses']
             response_length = responses.shape[1]; full_attention_mask = data.batch['attention_mask']
             response_masks = full_attention_mask[:, -response_length:]

        batch_size = len(data.batch['responses'])
        reward_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)
        reward_extra_info = defaultdict(lambda: [None] * batch_size)
        already_print_count = 0

        # --- Process Each Sample ---
        for i in range(batch_size):
            try:
                response_ids = data.batch['responses'][i]; res_mask = response_masks[i]
                valid_indices_in_response = torch.where(res_mask != 0)[0]

                if len(valid_indices_in_response) == 0: score_dict = tvg_multi_compute_score("", None, self.reward_weights)
                else:
                    valid_response_ids = response_ids[valid_indices_in_response]
                    response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)
                    gt_grounding = data.non_tensor_batch.get('ground_truth_grounding')[i] if 'ground_truth_grounding' in data.non_tensor_batch else None
                    score_dict = tvg_multi_compute_score(response_str, gt_grounding, self.reward_weights)

                # --- Store Rewards and Extra Info ---
                final_reward = score_dict["score"]
                if len(valid_indices_in_response) > 0:
                    last_token_index = valid_indices_in_response[-1].item()
                    reward_tensor[i, last_token_index] = final_reward
                for key, value in score_dict.items(): reward_extra_info[key][i] = value

                # --- Debug Printing (Updated) ---
                if already_print_count < self.num_examine and len(valid_indices_in_response) > 0:
                    print("-" * 50)
                    print(f"Example {i+1}/{batch_size} (Debug Print {already_print_count + 1}/{self.num_examine})")
                    print(f"[Response]: {response_str}")
                    print(f"[GT Grounding]: {gt_grounding}")
                    print(f"[Pred Grounding Cands]: {score_dict['predicted_grounding_candidates']}")
                    print(f"[Pred Grounding Union]: {score_dict['predicted_grounding_union']}") # Added Union
                    print(f"[Union Grounding IoU]: {score_dict['grounding_iou_union']:.4f}") # Changed Label
                    print(f"[Grounding Format]: {score_dict['grounding_format']:.1f}")
                    print(f"[Combined Reward]: {final_reward:.4f}")
                    print("-" * 50)
                    already_print_count += 1

            except Exception as e:
                logger.error(f"Unexpected error processing sample {i}: {e}", exc_info=True)
                default_score = tvg_multi_compute_score("", None, self.reward_weights)
                for key, value in default_score.items(): reward_extra_info[key][i] = value

        # Convert lists in reward_extra_info to numpy arrays
        final_reward_extra_info = {}
        for key, val_list in reward_extra_info.items():
            try: final_reward_extra_info[key] = np.array(val_list, dtype=object)
            except Exception as e:
                logger.warning(f"Could not convert reward_extra_info key '{key}' to numpy array: {e}. Keeping as list.")
                final_reward_extra_info[key] = val_list

        if return_dict:
            return { "reward_tensor": reward_tensor, "reward_extra_info": final_reward_extra_info }
        else: return reward_tensor
