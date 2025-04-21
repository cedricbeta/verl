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

import os
import re
from typing import List, Union, Optional, Dict, Any
import copy
import datasets
from collections import defaultdict
import logging

import torch
import numpy as np
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer, ProcessorMixin
from omegaconf import ListConfig, DictConfig

from verl.utils.model import compute_position_id_with_mask
import verl.utils.torch_functional as verl_F

logger = logging.getLogger(__name__)

def collate_fn(data_list: list[dict]) -> dict:
    """
    Collates a list of dictionaries into a single dictionary containing
    stacked tensors and numpy arrays of non-tensor data.
    Handles potential None values in non-tensor lists by filtering them out
    before converting to numpy arrays.
    """
    tensors = defaultdict(list)
    non_tensors = defaultdict(lambda: {'values': [], 'indices': []}) # Store values and original indices

    # Filter out potential error dictionaries before processing
    valid_data_list = [d for d in data_list if not isinstance(d, dict) or 'error' not in d]
    if len(valid_data_list) < len(data_list):
        logger.warning(f"Filtered out {len(data_list) - len(valid_data_list)} error items during collation.")
    if not valid_data_list:
        logger.error("Collate function received no valid data items.")
        # Return an empty dict or handle as appropriate for the trainer
        return {}


    for i, data in enumerate(valid_data_list):
        original_index = data.get('original_index', i) # Use original index if provided by __getitem__ error handling
        for key, val in data.items():
            if key == 'error' or key == 'original_index': continue # Skip special keys

            if isinstance(val, torch.Tensor):
                 # Ensure tensor keys are consistently present across the batch
                if key not in tensors and i > 0:
                     # This indicates an inconsistency in the batch items
                     logger.warning(f"Tensor key '{key}' missing in item {original_index}, present in others. Check dataset processing.")
                     # Decide how to handle: skip item, pad, raise error? Skipping key for now.
                     continue
                tensors[key].append(val)
            elif val is not None: # Only add non-None non-tensors
                non_tensors[key]['values'].append(val)
                non_tensors[key]['indices'].append(original_index) # Store original index for this value
            # Implicitly skips None values for non-tensors


    collated_batch = {}
    # Stack tensors
    tensor_keys = list(tensors.keys()) # Get keys from the first valid item if possible
    if tensors:
        ref_keys = list(tensors.keys())
        for key in ref_keys:
            val_list = tensors[key]
            if not val_list: continue # Skip if somehow empty after filtering
            try:
                # Pad tensors if they have different lengths (e.g., raw_prompt_ids)
                if key == 'raw_prompt_ids' and len(val_list) > 1:
                     # Check if padding is needed
                     if any(val_list[0].shape != t.shape for t in val_list[1:]):
                          collated_batch[key] = torch.nn.utils.rnn.pad_sequence(
                              val_list, batch_first=True, padding_value=0 # Use appropriate padding value
                          )
                     else:
                          collated_batch[key] = torch.stack(val_list, dim=0)
                else:
                     collated_batch[key] = torch.stack(val_list, dim=0)
            except RuntimeError as e:
                 logger.error(f"Error stacking tensor key '{key}': {e}. Check tensor shapes.")
                 shapes = [t.shape for t in val_list]
                 logger.error(f"Shapes for key '{key}': {shapes}")
                 raise e # Re-raise after logging shapes

    # Process non-tensors: Convert lists to numpy object arrays
    for key, data_dict in non_tensors.items():
        values = data_dict['values']
        if not values: continue # Skip if list is empty after filtering Nones

        try:
             # Convert the list of potentially complex objects
             collated_batch[key] = np.array(values, dtype=object)
        except Exception as e:
             logger.error(f"Error converting non-tensor key '{key}' to numpy array: {e}")
             collated_batch[key] = values # Keep as list as a fallback

    return collated_batch


class RLHFDataset(Dataset):
    """
    Dataset for RLHF, modified for multi-candidate temporal grounding ONLY.
    Assumes input data has 'problem' (grounding query) and
    'solution' (grounding answer list [start, end]).
    """

    def __init__(
        self,
        data_files: Union[str, List[str]],
        tokenizer: PreTrainedTokenizer,
        config: DictConfig,
        processor: Optional[ProcessorMixin] = None,
    ):
        if not isinstance(data_files, (List, ListConfig)):
            data_files = [data_files]

        self.data_files = copy.deepcopy(data_files)
        self.original_data_files = copy.deepcopy(data_files)
        self.tokenizer = tokenizer
        self.processor = processor
        self.config = config

        # --- Configuration ---
        self.cache_dir = os.path.expanduser(config.get("cache_dir", "~/.cache/verl/rlhf"))
        self.prompt_key = config.get("prompt_key", "problem")
        self.answer_key = config.get("answer_key", "solution") # Grounding GT
        # self.qa_key = config.get("qa_key", "qa") # REMOVED
        self.image_key = config.get("image_key", "images")
        self.video_key = config.get("video_key", "videos")
        self.max_prompt_length = config.get("max_prompt_length", 2048)
        self.system_prompt = config.get("system_prompt", None)

        self.return_raw_chat = config.get('return_raw_chat', False)
        self.truncation = config.get('truncation', 'error')
        self.filter_overlong_prompts = config.get("filter_overlong_prompts", True)

        self.num_workers = config.get("filter_overlong_prompts_workers", max(1, os.cpu_count() // 4))
        self.num_workers = min(self.num_workers, os.cpu_count())

        self.serialize_dataset = False
        self._download()
        self._read_files_and_tokenize()

    def _download(self, use_origin_parquet=False):
        """Downloads or copies data files to a local cache."""
        from verl.utils.fs import copy_to_local
        data_files = self.data_files if not use_origin_parquet else self.original_data_files
        for i, file_path in enumerate(data_files):
            if file_path.startswith(('http://', 'https://', 'hdfs://', 's3://')):
                 self.data_files[i] = copy_to_local(src=file_path, cache_dir=self.cache_dir)
            else:
                 if not os.path.exists(file_path):
                      raise FileNotFoundError(f"Local data file not found: {file_path}")
                 self.data_files[i] = file_path

    def _read_files_and_tokenize(self):
        """Reads data files and optionally filters long prompts."""
        dataframes = []
        for data_file_path in self.data_files:
            file_type = None
            if data_file_path.endswith((".json", ".jsonl")): file_type = "json"
            elif data_file_path.endswith(".parquet"): file_type = "parquet"

            if file_type is None:
                logger.warning(f"Skipping file with unrecognized extension: {data_file_path}")
                continue

            try:
                logger.info(f"Loading dataset: type='{file_type}', path='{data_file_path}'")
                dataframe = datasets.load_dataset(file_type, data_files=data_file_path)["train"]
                dataframes.append(dataframe)
            except Exception as e:
                 logger.error(f"Error loading dataset file {data_file_path}: {e}", exc_info=True)

        if not dataframes: raise ValueError("No dataframes loaded.")

        self.dataframe: datasets.Dataset = datasets.concatenate_datasets(dataframes)
        logger.info(f'Initial dataset length: {len(self.dataframe)}')

        # Filter out prompts that are too long
        if self.filter_overlong_prompts:
            tokenizer = self.tokenizer
            prompt_key = self.prompt_key
            system_prompt = self.system_prompt
            max_len = self.max_prompt_length

            def check_length(doc):
                # Simulate prompt construction (without QA)
                messages = []
                if system_prompt: messages.append({"role": "system", "content": system_prompt})
                # Directly use the prompt data, assuming it's already in message format
                prompt_data = doc.get(prompt_key)
                if isinstance(prompt_data, list):
                     messages.extend(prompt_data)
                elif isinstance(prompt_data, str): # Handle simple string prompt if needed
                     messages.append({"role": "user", "content": prompt_data})
                else:
                     logger.warning(f"Unexpected prompt format in doc for filtering: {prompt_data}")
                     return False # Exclude if prompt format is wrong

                try:
                    # Tokenize using the appropriate method
                    if self.processor:
                         tokenized_len = len(self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=True).get('input_ids', []))
                    else:
                         tokenized_len = len(tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=True).get('input_ids', []))
                    return tokenized_len <= max_len
                except Exception as e:
                     logger.warning(f"Error tokenizing during filtering: {e}. Excluding sample.")
                     return False

            original_len = len(self.dataframe)
            self.dataframe = self.dataframe.filter(
                check_length,
                num_proc=self.num_workers,
                desc=f"Filtering prompts longer than {max_len} tokens"
            )
            filtered_len = len(self.dataframe)
            logger.info(f'Filtered dataset length: {filtered_len} (removed {original_len - filtered_len})')

    def resume_dataset_state(self):
        """Resumes dataset state after checkpoint loading."""
        self.serialize_dataset = False if hasattr(self, 'original_data_files') else True
        if not self.serialize_dataset:
            logger.info("Resuming dataset from original files...")
            self._download(use_origin_parquet=True)
            self._read_files_and_tokenize()
        else:
            logger.warning('Old dataloader ckpt file used (dataset serialized).')

    def __len__(self):
        """Returns the number of samples in the dataset."""
        return len(self.dataframe)

    def _build_messages(self, example: dict) -> List[Dict[str, Any]]:
        """
        Constructs the list of message dictionaries, handling multimodal placeholders.
        (Simplified: No QA appending)
        """
        if self.prompt_key not in example or not isinstance(example[self.prompt_key], list):
             logger.error(f"Missing or invalid '{self.prompt_key}' in example: {example}")
             return [{"role": "user", "content": "Error: Invalid prompt data."}]

        messages: list = example.pop(self.prompt_key) # Pop to avoid duplication

        has_media = self.image_key in example or self.video_key in example
        processed_messages = []
        for message in messages:
             if not isinstance(message, dict) or 'role' not in message or 'content' not in message:
                  logger.warning(f"Skipping invalid message structure: {message}")
                  continue

             content = message["content"]
             if has_media and isinstance(content, str):
                 content_list = []
                 for segment in re.split(r'(<image>|<video>)', content):
                     if not segment: continue
                     if segment == "<image>": content_list.append({"type": "image"})
                     elif segment == "<video>": content_list.append({"type": "video"})
                     else: content_list.append({"type": "text", "text": segment})
                 message["content"] = content_list
             processed_messages.append(message)

        return processed_messages

    def __getitem__(self, item: int) -> Dict[str, Any]:
        """
        Retrieves and processes a single data sample for grounding ONLY.
        """
        original_index = item # Store original index for potential error reporting
        try:
            row_dict: dict = self.dataframe[item]
            messages_from_data = self._build_messages(copy.deepcopy(row_dict))

            # --- Construct Final Messages (System Prompt + Data Prompt) ---
            messages = []
            if self.system_prompt:
                messages.append({"role": "system", "content": self.system_prompt})
            messages.extend(messages_from_data)
            # --- REMOVED QA Appending Logic ---

            # --- Tokenization and Processing ---
            model_inputs = {}
            raw_prompt = ""

            if self.processor is not None:
                # --- Multimodal Processing ---
                from verl.utils.dataset.vision_utils import process_image, process_video

                try:
                     raw_prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
                except Exception as e:
                     logger.error(f"Item {item}: Error applying chat template with processor: {e}", exc_info=True)
                     return {"error": f"Chat template error: {e}", "original_index": original_index}

                multi_modal_data = {}
                images = None
                videos = None

                if self.image_key in row_dict and row_dict[self.image_key]:
                    try:
                        images = [process_image(img) for img in row_dict.pop(self.image_key)]
                        multi_modal_data["image"] = images
                    except Exception as e: logger.warning(f"Item {item}: Error processing image: {e}")

                if self.video_key in row_dict and row_dict[self.video_key]:
                    try:
                        videos = [process_video(vid) for vid in row_dict.pop(self.video_key)]
                        multi_modal_data["video"] = videos
                    except Exception as e: logger.warning(f"Item {item}: Error processing video: {e}")

                try:
                    model_inputs = self.processor(text=[raw_prompt], images=images, videos=videos, return_tensors="pt")
                except Exception as e:
                     logger.error(f"Item {item}: Error calling processor: {e}", exc_info=True)
                     return {"error": f"Processor call error: {e}", "original_index": original_index}

                input_ids = model_inputs.get("input_ids")
                attention_mask = model_inputs.get("attention_mask")

                if input_ids is None or attention_mask is None:
                     logger.error(f"Item {item}: Processor missing 'input_ids' or 'attention_mask'.")
                     return {"error": "Missing tensors from processor.", "original_index": original_index}

                model_inputs.pop("input_ids", None); model_inputs.pop("attention_mask", None)
                model_inputs.pop("second_per_grid_ts", None)
                row_dict["multi_modal_data"] = multi_modal_data
                row_dict["multi_modal_inputs"] = {k: v for k, v in model_inputs.items()}

            else:
                # --- Text-only Processing ---
                try:
                    raw_prompt = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
                    model_inputs = self.tokenizer(raw_prompt, return_tensors='pt', add_special_tokens=False)
                    input_ids = model_inputs.get('input_ids')
                    attention_mask = model_inputs.get('attention_mask')
                    if input_ids is None or attention_mask is None:
                         logger.error(f"Item {item}: Tokenizer missing 'input_ids' or 'attention_mask'.")
                         return {"error": "Missing tensors from tokenizer.", "original_index": original_index}
                except Exception as e:
                     logger.error(f"Item {item}: Error applying chat template/tokenizing: {e}", exc_info=True)
                     return {"error": f"Tokenizer/template error: {e}", "original_index": original_index}

            # --- Post-processing (Padding/Truncation) ---
            input_ids, attention_mask = verl_F.postprocess_data(
                input_ids=input_ids, attention_mask=attention_mask,
                max_length=self.max_prompt_length,
                pad_token_id=self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id,
                left_pad=True, truncation=self.truncation
            )

            # --- Position IDs ---
            position_ids = None
            if self.processor and self.processor.__class__.__name__ == "Qwen2VLProcessor":
                 try:
                      from verl.models.transformers.qwen2_vl import get_rope_index
                      position_ids = get_rope_index(
                           self.processor, input_ids=input_ids[0],
                           image_grid_thw=model_inputs.get("image_grid_thw"), video_grid_thw=model_inputs.get("video_grid_thw"),
                           second_per_grid_ts=model_inputs.get("second_per_grid_ts"), attention_mask=attention_mask[0]
                      ).unsqueeze(0)
                 except ImportError: logger.warning("Qwen2VL 'get_rope_index' not found. Using default position IDs.")
                 except Exception as e: logger.warning(f"Error generating Qwen2VL position IDs: {e}. Using default.")

            if position_ids is None:
                 position_ids = compute_position_id_with_mask(attention_mask)

            # --- Final Output Dictionary ---
            output_dict = {
                'input_ids': input_ids[0],
                'attention_mask': attention_mask[0],
                'position_ids': position_ids[0],
                # --- Ground Truth ---
                'ground_truth_grounding': row_dict.get(self.answer_key), # Grounding answer only
                # 'ground_truth_qa': ground_truth_qa_answers, # REMOVED
                # --- Other Info ---
                'data_source': row_dict.get('data_source', 'unknown'),
                'problem_type': 'tvg_multi_candidate', # Updated task type
                'index': row_dict.get("extra_info", {}).get("index", item),
                'videos': row_dict.get(self.video_key)
            }

            if "multi_modal_inputs" in row_dict: output_dict['multi_modal_inputs'] = row_dict['multi_modal_inputs']
            if "multi_modal_data" in row_dict: output_dict['multi_modal_data'] = row_dict['multi_modal_data']

            # Store raw prompt tokens (optional)
            try:
                 raw_prompt_ids_list = self.tokenizer.encode(raw_prompt, add_special_tokens=False)
                 if len(raw_prompt_ids_list) > self.max_prompt_length:
                      if self.truncation == "left": raw_prompt_ids_list = raw_prompt_ids_list[-self.max_prompt_length:]
                      elif self.truncation == "right": raw_prompt_ids_list = raw_prompt_ids_list[:self.max_prompt_length]
                 output_dict['raw_prompt_ids'] = torch.tensor(raw_prompt_ids_list, dtype=torch.long)
            except Exception as e:
                 logger.warning(f"Item {item}: Could not encode raw_prompt: {e}")
                 output_dict['raw_prompt_ids'] = torch.tensor([], dtype=torch.long)

            if self.return_raw_chat: output_dict['raw_prompt'] = messages

            return output_dict

        except Exception as e:
            logger.error(f"Error processing item {item}: {e}", exc_info=True)
            # Return error dict with original index for collation filtering
            return {"error": f"Failed to process item {item}: {e}", "original_index": original_index}


    def __getstate__(self):
        """Controls object state for pickling."""
        if not self.serialize_dataset:
            state = self.__dict__.copy()
            if 'dataframe' in state: del state['dataframe']
            return state
        return self.__dict__.copy()

    def __setstate__(self, state):
        """Restores object state from pickle."""
        self.__dict__.update(state)
        if 'dataframe' not in state and not self.serialize_dataset:
            logger.info("Reloading dataframe after deserialization...")
            try:
                self._download()
                self._read_files_and_tokenize()
            except Exception as e:
                logger.error(f"Failed to reload dataframe after deserialization: {e}", exc_info=True)
                self.dataframe = None

