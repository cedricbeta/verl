# Add this class definition to video_verl/video.py or a new reward manager file

import torch
import re
import json
import numpy as np
from collections import defaultdict
from transformers import PreTrainedTokenizer
from typing import Union, Dict, Any, Optional

from verl import DataProto

import logging
logger = logging.getLogger(__name__)

# Updated tvg_compute_score to expect <thinking> and <grounding>
def tvg_compute_score(
    response_text: str,
    gt_times: list[Optional[float]],
    grounding_tag_regex: str = r"<answer>(.*?)</answer>", # Regex for grounding tag
    thinking_tag_present_bonus: float = 0.1 # Small bonus if <thinking> is found
    ) -> dict:
    """
    Computes TVG score expecting <thinking>...</thinking><grounding>...</grounding> format.
    The <grounding> tag should contain a JSON with "start_time" and "end_time".

    Returns:
        dict: {"score": float, "tvg_accuracy": float (IoU), "tvg_format": float (0 to 1)}
              tvg_format:
                - 1.0 if <grounding> found and JSON inside is valid (keys present)
                - + thinking_tag_present_bonus if <thinking> found
                - 0.0 otherwise for base format
              score: Combines format and accuracy.
    """
    parsed_start, parsed_end = None, None
    base_tvg_format_score = 0.0 # For the grounding tag and its content
    thinking_bonus = 0.0
    tvg_accuracy = 0.0 # IoU
    final_score = 0.0

    # Check for thinking tag
    if re.search(r"<thinking>(.*?)</thinking>", response_text, re.IGNORECASE | re.DOTALL):
        thinking_bonus = thinking_tag_present_bonus

    try:
        # Extract content within <grounding>...</grounding>
        grounding_match = re.search(grounding_tag_regex, response_text, re.IGNORECASE | re.DOTALL)
        if grounding_match and grounding_match.group(1):
            grounding_content = grounding_match.group(1).strip()
            if grounding_content.startswith('{') and grounding_content.endswith('}'):
                answer_data = json.loads(grounding_content)
                if isinstance(answer_data, dict) and "start_time" in answer_data and "end_time" in answer_data:
                    base_tvg_format_score = 1.0 # Grounding tag and JSON keys are correct
                    raw_start = answer_data["start_time"]
                    raw_end = answer_data["end_time"]
                    if raw_start is not None and str(raw_start).strip() != "":
                        parsed_start = float(raw_start)
                    if raw_end is not None and str(raw_end).strip() != "":
                        parsed_end = float(raw_end)
    except json.JSONDecodeError:
        logger.debug(f"JSONDecodeError parsing S1 grounding content: {response_text}")
    except ValueError:
        logger.debug(f"ValueError converting S1 times from grounding content: {response_text}")
    except Exception:
        logger.debug(f"Generic error parsing S1 grounding content: {response_text}")

    # Calculate IoU (tvg_accuracy)
    if parsed_start is not None and parsed_end is not None and \
       gt_times[0] is not None and gt_times[1] is not None:
        gt_start, gt_end = gt_times
        if parsed_start <= parsed_end and gt_start <= gt_end:
            overlap_start = max(gt_start, parsed_start)
            overlap_end = min(gt_end, parsed_end)
            overlap_duration = max(0, overlap_end - overlap_start)
            gt_duration = gt_end - gt_start
            parsed_duration = parsed_end - parsed_start
            union_duration = gt_duration + parsed_duration - overlap_duration
            if union_duration > 1e-6: tvg_accuracy = overlap_duration / union_duration
            elif overlap_duration > 1e-6: tvg_accuracy = 1.0
            else: tvg_accuracy = 0.0
        else: tvg_accuracy = 0.0
    else: tvg_accuracy = 0.0

    # Final format score includes bonus for thinking tag
    total_tvg_format_score = base_tvg_format_score + (thinking_bonus if base_tvg_format_score > 0.5 else 0) # Add thinking bonus only if base format is good
    total_tvg_format_score = min(total_tvg_format_score, 1.0) # Cap at 1.0

    # Score: e.g. 0.3 for format, 0.7 for accuracy (can be tuned)
    if base_tvg_format_score > 0.5: # Only give accuracy points if essential grounding format is met
        final_score = (0.3 * total_tvg_format_score) + (0.7 * tvg_accuracy)
    else:
        final_score = 0.0 # No score if basic grounding tag/JSON is wrong

    return {"score": final_score, "tvg_accuracy": tvg_accuracy, "tvg_format": total_tvg_format_score, "base_tvg_format": base_tvg_format_score, "thinking_present_s1": float(thinking_bonus > 0)}


class VQAMultiStageRewardManager:
    """
    Reward manager for two-stage Video QA, expecting <thinking> tags.
    S1: <thinking>...</thinking><grounding>{"start_time":X, "end_time":Y}</grounding>
    S2: <thinking>...</thinking><answer>A</answer>
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        grounding_weight: float = 0.4,
        qa_accuracy_weight: float = 0.4,
        qa_format_weight: float = 0.2,
        s1_grounding_tag_regex: str = r"<answer>(.*?)</answer>", # For S1
        s2_answer_tag_regex: str = r"<answer>\s*([A-Z])\s*</answer>",    # For S2
        thinking_tag_bonus: float = 0.1, # Applied to both S1 and S2 format scores if <thinking> is present
        num_examine: int = 1
    ) -> None:
        self.tokenizer = tokenizer
        self.grounding_weight = grounding_weight
        self.qa_accuracy_weight = qa_accuracy_weight
        self.qa_format_weight = qa_format_weight
        self.thinking_tag_bonus = thinking_tag_bonus # Bonus if <thinking> is found before main tag

        try:
            self.s1_grounding_tag_pattern = re.compile(s1_grounding_tag_regex, re.IGNORECASE | re.DOTALL)
            self.s2_answer_tag_pattern = re.compile(s2_answer_tag_regex, re.IGNORECASE | re.DOTALL)
        except re.error as e:
            logger.error(f"Invalid regex. S1: '{s1_grounding_tag_regex}', S2: '{s2_answer_tag_regex}'. Error: {e}")
            raise
        self.num_examine = max(0, num_examine)

        total_weight = self.grounding_weight + self.qa_accuracy_weight + self.qa_format_weight
        if total_weight <= 1e-6:
             logger.warning("All reward weights sum to zero or less. Rewards will be zero.")
             self.grounding_weight = 0.0; self.qa_accuracy_weight = 0.0; self.qa_format_weight = 0.0
        elif abs(total_weight - 1.0) > 1e-6:
             logger.warning(f"Normalizing reward weights to sum to 1.")
             self.grounding_weight /= total_weight
             self.qa_accuracy_weight /= total_weight
             self.qa_format_weight /= total_weight

        print(
            f"VQAMultiStageRewardManager (with Thinking Tags) initialized. Weights: "
            f"Grounding={self.grounding_weight:.2f}, QA_Acc={self.qa_accuracy_weight:.2f}, "
            f"QA_Format={self.qa_format_weight:.2f}. ThinkingBonus={self.thinking_tag_bonus}. "
            f"S1_Regex: '{s1_grounding_tag_regex}', S2_Regex: '{s2_answer_tag_regex}'. "
            f"Examine: {self.num_examine}."
        )

    def _extract_s2_answer_letter(self, decoded_s2_response: str) -> Optional[str]:
        match = self.s2_answer_tag_pattern.search(decoded_s2_response)
        if match and match.group(1):
            return match.group(1).upper()
        return None

    def __call__(self, data: DataProto, return_dict=False) -> Union[torch.Tensor, Dict[str, Any]]:
        required_batch_keys = ['responses']
        required_non_tensor_keys = ['decoded_grounding_texts', 'start_time', 'end_time', 'ground_truth_answer']
        # ... (Input validation remains the same) ...
        if 'response_mask' in data.batch:
             response_masks = data.batch['response_mask']
        else:
             logger.warning("VQA Reward: 'response_mask' not found. Assuming all tokens in 'responses' are valid.")
             response_masks = torch.ones_like(data.batch['responses'])

        batch_size = len(data.batch['responses']) # Should be s2_responses
        # Ensure non_tensor_batch items have correct length
        for key in required_non_tensor_keys:
            if len(data.non_tensor_batch[key]) != batch_size:
                raise ValueError(f"Length mismatch: Batch size {batch_size}, non_tensor_batch['{key}'] length {len(data.non_tensor_batch[key])}")


        reward_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)
        reward_extra_info = defaultdict(list)
        already_print_count = 0

        for i in range(batch_size):
            try:
                # Stage 2 (QA) response details
                s2_response_ids = data.batch['responses'][i] # This is the S2 response from PPO actor
                s2_res_mask = response_masks[i]
                decoded_s2_response_full = ""
                pred_qa_answer_letter = None
                last_token_index_s2 = 0
                thinking_present_s2 = False

                valid_indices_s2 = torch.where(s2_res_mask != 0)[0]
                if len(valid_indices_s2) == 0:
                    logger.warning(f"VQA Reward: Sample {i} has an empty Stage 2 response mask.")
                else:
                    valid_s2_response_ids = s2_response_ids[valid_indices_s2]
                    decoded_s2_response_full = self.tokenizer.decode(valid_s2_response_ids, skip_special_tokens=True).strip()
                    pred_qa_answer_letter = self._extract_s2_answer_letter(decoded_s2_response_full)
                    if re.search(r"<thinking>(.*?)</thinking>", decoded_s2_response_full, re.IGNORECASE | re.DOTALL):
                        thinking_present_s2 = True
                    last_token_index_s2 = valid_indices_s2[-1].item()

                # Stage 1 (Grounding) response text (comes from PPO non_tensor_batch)
                s1_response_text_full = data.non_tensor_batch['decoded_grounding_texts'][i]

                # Ground Truth
                gt_start_time = data.non_tensor_batch['start_time'][i]
                gt_end_time = data.non_tensor_batch['end_time'][i]
                gt_qa_answer_letter = data.non_tensor_batch['ground_truth_answer'][i]

                # --- Grounding Score (S1) ---
                # tvg_compute_score expects full S1 response text and GT times
                s1_score_details = tvg_compute_score(
                    s1_response_text_full,
                    [gt_start_time, gt_end_time],
                    grounding_tag_regex=self.s1_grounding_tag_pattern.pattern, # Pass the string pattern
                    thinking_tag_present_bonus=self.thinking_tag_bonus
                )
                grounding_score_component = s1_score_details["score"] # This score combines S1 format (grounding tag, JSON, thinking) & IoU

                # --- QA Accuracy Score (S2) ---
                qa_accuracy_score = 0.0
                if pred_qa_answer_letter:
                    qa_accuracy_score = 1.0 if pred_qa_answer_letter == gt_qa_answer_letter.upper() else 0.0
                
                # --- QA Format Score (S2) ---
                # Base format: successful extraction of answer letter via regex
                base_s2_format_score = 1.0 if pred_qa_answer_letter is not None else 0.0
                # Add bonus if thinking tag is present and base format is good
                s2_thinking_bonus_applied = self.thinking_tag_bonus if thinking_present_s2 and base_s2_format_score > 0.5 else 0.0
                qa_format_score = min(base_s2_format_score + s2_thinking_bonus_applied, 1.0)

                # --- Combine Rewards ---
                final_score = (
                    self.grounding_weight * grounding_score_component +
                    self.qa_accuracy_weight * qa_accuracy_score +
                    self.qa_format_weight * qa_format_score
                )

                # --- Store Rewards and Metrics ---
                reward_extra_info['s1_grounding_iou'].append(s1_score_details["tvg_accuracy"])
                reward_extra_info['s1_grounding_format_score'].append(s1_score_details["tvg_format"]) # Includes thinking bonus
                reward_extra_info['s1_base_format_correct'].append(s1_score_details["base_tvg_format"])
                reward_extra_info['s1_thinking_present'].append(s1_score_details["thinking_present_s1"])
                reward_extra_info['s1_grounding_combined_score_unweighted'].append(grounding_score_component)

                reward_extra_info['s2_qa_accuracy_correct'].append(qa_accuracy_score)
                reward_extra_info['s2_qa_format_score'].append(qa_format_score) # Includes thinking bonus
                reward_extra_info['s2_base_format_correct'].append(base_s2_format_score)
                reward_extra_info['s2_thinking_present'].append(float(thinking_present_s2))

                reward_extra_info['final_weighted_score'].append(final_score)

                if len(valid_indices_s2) > 0:
                    reward_tensor[i, last_token_index_s2] = final_score

                if already_print_count < self.num_examine:
                    print("-" * 50)
                    print(f"VQA Reward Example {i+1}/{batch_size} (Print {already_print_count + 1}/{self.num_examine})")
                    print(f"[S1 Full Response]: {s1_response_text_full}")
                    print(f"[S2 Full Response]: {decoded_s2_response_full if len(valid_indices_s2) > 0 else '<Empty S2 Response>'}")
                    print(f"[GT Times]: [{gt_start_time}, {gt_end_time}]")
                    print(f"[GT Answer]: {gt_qa_answer_letter}")
                    print(f"[Extracted S2 Answer Letter]: {pred_qa_answer_letter if pred_qa_answer_letter else 'N/A'}")

                    print(f"  S1 Grounding IoU: {s1_score_details['tvg_accuracy']:.4f}")
                    print(f"  S1 Grounding Format (incl. thinking bonus): {s1_score_details['tvg_format']:.2f} (Base: {s1_score_details['base_tvg_format']:.1f}, Thinking: {s1_score_details['thinking_present_s1']:.1f})")
                    print(f"  S1 Grounding Score (unweighted): {grounding_score_component:.4f}")

                    print(f"  S2 QA Accuracy Correct: {qa_accuracy_score:.1f}")
                    print(f"  S2 QA Format (incl. thinking bonus): {qa_format_score:.2f} (Base: {base_s2_format_score:.1f}, Thinking: {float(thinking_present_s2):.1f})")

                    print(f"  -> S1 Weighted Contrib (W={self.grounding_weight:.2f}): {self.grounding_weight * grounding_score_component:.4f}")
                    print(f"  -> S2 Acc Weighted Contrib (W={self.qa_accuracy_weight:.2f}): {self.qa_accuracy_weight * qa_accuracy_score:.4f}")
                    print(f"  -> S2 Format Weighted Contrib (W={self.qa_format_weight:.2f}): {self.qa_format_weight * qa_format_score:.4f}")
                    print(f"[Final Combined Reward]: {final_score:.4f}")
                    print("-" * 50)
                    already_print_count += 1

            except KeyError as e:
                logger.error(f"VQA Reward: KeyError processing sample {i}: {e}. Zero reward & metrics.")
                # Append defaults for all keys
                for key_default in [
                    's1_grounding_iou', 's1_grounding_format_score', 's1_base_format_correct', 's1_thinking_present',
                    's1_grounding_combined_score_unweighted', 's2_qa_accuracy_correct', 's2_qa_format_score',
                    's2_base_format_correct', 's2_thinking_present', 'final_weighted_score'
                ]: reward_extra_info[key_default].append(0.0)
            except Exception as e:
                logger.error(f"VQA Reward: Unexpected error processing sample {i}: {e}")
                import traceback; traceback.print_exc()
                for key_default in [
                    's1_grounding_iou', 's1_grounding_format_score', 's1_base_format_correct', 's1_thinking_present',
                    's1_grounding_combined_score_unweighted', 's2_qa_accuracy_correct', 's2_qa_format_score',
                    's2_base_format_correct', 's2_thinking_present', 'final_weighted_score'
                ]: reward_extra_info[key_default].append(0.0)


        final_reward_extra_info = {k: np.array(v) for k, v in reward_extra_info.items()}

        if return_dict:
            return {"reward_tensor": reward_tensor, "reward_extra_info": final_reward_extra_info}
        else:
            return reward_tensor