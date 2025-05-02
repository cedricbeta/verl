# coding=utf-8
# Copyright 2024 The Qwen team, Bytedance Ltd. and/or its affiliates.
# ... (License header) ...
"""
Dataset loader for End-to-End Video QA.
Loads QA question, full video, and prepares inputs for a single generation step,
compatible with the standard VERL PPO fit loop structure.
Integrates video processing utilities.
"""
from __future__ import annotations

import os
import copy
import logging
import math # Ensure math is imported
# ... other necessary imports ...
import torch
import numpy as np
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer, ProcessorMixin
from omegaconf import DictConfig, ListConfig
import base64
import logging
import math
import os
import sys
import time
import warnings
from functools import lru_cache
from io import BytesIO

import requests
import torch
import torchvision
from packaging import version
from PIL import Image
from torchvision import io, transforms
from torchvision.transforms import InterpolationMode
from typing import Optional


logger = logging.getLogger(__name__)

IMAGE_FACTOR = 28
MIN_PIXELS = 4 * 28 * 28
MAX_PIXELS = 16384 * 28 * 28
MAX_RATIO = 200

VIDEO_MIN_PIXELS = 128 * 28 * 28
VIDEO_MAX_PIXELS = 768 * 28 * 28
FRAME_FACTOR = 2
FPS = 2.0
FPS_MIN_FRAMES = 4
FPS_MAX_FRAMES = 768

# Set the maximum number of video token inputs.
# Here, 128K represents the maximum number of input tokens for the VLLM model.
# Remember to adjust it according to your own configuration.
VIDEO_TOTAL_PIXELS = int(float(os.environ.get('VIDEO_MAX_PIXELS', 128000 * 28 * 28 * 0.9)))
logger.info(f"set VIDEO_TOTAL_PIXELS: {VIDEO_TOTAL_PIXELS}")


def round_by_factor(number: int, factor: int) -> int:
    """Returns the closest integer to 'number' that is divisible by 'factor'."""
    return round(number / factor) * factor


def ceil_by_factor(number: int, factor: int) -> int:
    """Returns the smallest integer greater than or equal to 'number' that is divisible by 'factor'."""
    return math.ceil(number / factor) * factor


def floor_by_factor(number: int, factor: int) -> int:
    """Returns the largest integer less than or equal to 'number' that is divisible by 'factor'."""
    return math.floor(number / factor) * factor


def smart_resize(
    height: int, width: int, factor: int = IMAGE_FACTOR, min_pixels: int = MIN_PIXELS, max_pixels: int = MAX_PIXELS
) -> tuple[int, int]:
    """
    Rescales the image so that the following conditions are met:

    1. Both dimensions (height and width) are divisible by 'factor'.

    2. The total number of pixels is within the range ['min_pixels', 'max_pixels'].

    3. The aspect ratio of the image is maintained as closely as possible.
    """
    if max(height, width) / min(height, width) > MAX_RATIO:
        raise ValueError(
            f"absolute aspect ratio must be smaller than {MAX_RATIO}, got {max(height, width) / min(height, width)}"
        )
    h_bar = max(factor, round_by_factor(height, factor))
    w_bar = max(factor, round_by_factor(width, factor))
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = floor_by_factor(height / beta, factor)
        w_bar = floor_by_factor(width / beta, factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = ceil_by_factor(height * beta, factor)
        w_bar = ceil_by_factor(width * beta, factor)
    return h_bar, w_bar


def to_rgb(pil_image: Image.Image) -> Image.Image:
      if pil_image.mode == 'RGBA':
          white_background = Image.new("RGB", pil_image.size, (255, 255, 255))
          white_background.paste(pil_image, mask=pil_image.split()[3])  # Use alpha channel as mask
          return white_background
      else:
          return pil_image.convert("RGB")


def fetch_image(ele: dict[str, str | Image.Image], size_factor: int = IMAGE_FACTOR) -> Image.Image:
    if "image" in ele:
        image = ele["image"]
    else:
        image = ele["image_url"]
    image_obj = None
    if isinstance(image, Image.Image):
        image_obj = image
    elif image.startswith("http://") or image.startswith("https://"):
        response = requests.get(image, stream=True)
        image_obj = Image.open(BytesIO(response.content))
    elif image.startswith("file://"):
        image_obj = Image.open(image[7:])
    elif image.startswith("data:image"):
        if "base64," in image:
            _, base64_data = image.split("base64,", 1)
            data = base64.b64decode(base64_data)
            image_obj = Image.open(BytesIO(data))
    else:
        image_obj = Image.open(image)
    if image_obj is None:
        raise ValueError(f"Unrecognized image input, support local path, http url, base64 and PIL.Image, got {image}")
    image = to_rgb(image_obj)
    ## resize
    if "resized_height" in ele and "resized_width" in ele:
        resized_height, resized_width = smart_resize(
            ele["resized_height"],
            ele["resized_width"],
            factor=size_factor,
        )
    else:
        width, height = image.size
        min_pixels = ele.get("min_pixels", MIN_PIXELS)
        max_pixels = ele.get("max_pixels", MAX_PIXELS)
        resized_height, resized_width = smart_resize(
            height,
            width,
            factor=size_factor,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
        )
    image = image.resize((resized_width, resized_height))

    return image


def smart_nframes(
    ele: dict,
    total_frames: int,
    video_fps: int | float,
) -> int:
    """
    Calculates the target number of frames for video used for model inputs.
    Prioritizes 'nframes' if specified and valid, otherwise uses 'fps'.
    """
    # assert not ("fps" in ele and "nframes" in ele), "Only accept either `fps` or `nframes`" # REMOVED ASSERTION

    target_nframes = 0 # Initialize

    # --- Prioritize 'nframes' if provided and not None ---
    if "nframes" in ele and ele["nframes"] is not None:
        try:
            requested_nframes = int(ele["nframes"])
            target_nframes = round_by_factor(requested_nframes, FRAME_FACTOR)
            logger.debug(f"smart_nframes: Using 'nframes' config: {requested_nframes} -> rounded {target_nframes}")

            # Apply min/max constraints directly to nframes
            min_frames_nf = ceil_by_factor(ele.get("min_frames", FPS_MIN_FRAMES), FRAME_FACTOR)
            max_frames_nf_config = ele.get("max_frames", FPS_MAX_FRAMES)
            # Ensure max_frames constraint doesn't exceed available total_frames
            # Handle total_frames=0 case
            max_frames_nf = floor_by_factor(min(max_frames_nf_config, total_frames), FRAME_FACTOR) if total_frames > 0 else 0

            target_nframes = min(max(target_nframes, min_frames_nf), max_frames_nf) if total_frames > 0 else 0
            logger.debug(f"smart_nframes: Clamped 'nframes' to {target_nframes} based on min/max/total ({min_frames_nf}/{max_frames_nf}/{total_frames})")

        except (ValueError, TypeError) as e:
            logger.warning(f"smart_nframes: Invalid 'nframes' value ({ele['nframes']}): {e}. Falling back to FPS calculation.")
            target_nframes = 0 # Reset to trigger FPS calculation

    # --- Use 'fps' if 'nframes' was not prioritized or was invalid ---
    if target_nframes == 0 and total_frames > 0: # Only calculate if needed and possible
        fps = ele.get("fps", FPS) # Use configured fps or default
        min_frames = ceil_by_factor(ele.get("min_frames", FPS_MIN_FRAMES), FRAME_FACTOR)
        max_frames_config = ele.get("max_frames", FPS_MAX_FRAMES)
        max_frames = floor_by_factor(min(max_frames_config, total_frames), FRAME_FACTOR)
        logger.debug(f"smart_nframes: Using 'fps' config: {fps} (min: {min_frames}, max: {max_frames}, total: {total_frames})")

        if video_fps > 1e-6: # Check for valid video_fps before division
            calculated_nframes = total_frames / video_fps * fps
        else:
            logger.warning(f"smart_nframes: video_fps is {video_fps:.2f}. Cannot use fps config. Using min_frames={min_frames}.")
            calculated_nframes = min_frames

        # Apply constraints
        if calculated_nframes > total_frames:
            logger.warning(f"smart_nframes: Calculated nframes[{calculated_nframes:.2f}] > total_frames[{total_frames}]. Clamping.")

        target_nframes = min(min(max(calculated_nframes, min_frames), max_frames), total_frames)
        target_nframes = floor_by_factor(target_nframes, FRAME_FACTOR) # Ensure divisible by factor

    # --- Final Validation and Clamping ---
    # Ensure at least FRAME_FACTOR frames unless total_frames is smaller, handle total_frames=0
    if total_frames == 0:
        target_nframes = 0
    else:
        min_possible_frames = FRAME_FACTOR if total_frames >= FRAME_FACTOR else max(1, total_frames) # At least 1 frame
        target_nframes = max(min_possible_frames, int(round(target_nframes)))
        # Ensure not more than total_frames
        target_nframes = min(target_nframes, total_frames)


    logger.debug(f"smart_nframes: Final target frames = {target_nframes}")

    # Final check (optional logging)
    # if total_frames > 0 and not (min_possible_frames <= target_nframes <= total_frames):
    #    logger.error(f"Post-calculation check failed: target={target_nframes}, min_poss={min_possible_frames}, total={total_frames}")

    return target_nframes

def _read_video_torchvision(
    ele: dict,
) -> (torch.Tensor, float):
    """read video using torchvision.io.read_video

    Args:
        ele (dict): a dict contains the configuration of video.
        support keys:
            - video: the path of video. support "file://", "http://", "https://" and local path.
            - video_start: the start time of video.
            - video_end: the end time of video.
    Returns:
        torch.Tensor: the video tensor with shape (T, C, H, W).
    """
    video_path = ele["video"]
    if version.parse(torchvision.__version__) < version.parse("0.19.0"):
        if "http://" in video_path or "https://" in video_path:
            warnings.warn("torchvision < 0.19.0 does not support http/https video path, please upgrade to 0.19.0.")
        if "file://" in video_path:
            video_path = video_path[7:]
    st = time.time()
    video, audio, info = io.read_video(
        video_path,
        start_pts=ele.get("video_start", 0.0),
        end_pts=ele.get("video_end", None),
        pts_unit="sec",
        output_format="TCHW",
    )
    total_frames, video_fps = video.size(0), info["video_fps"]
    logger.info(f"torchvision:  {video_path=}, {total_frames=}, {video_fps=}, time={time.time() - st:.3f}s")
    nframes = smart_nframes(ele, total_frames=total_frames, video_fps=video_fps)
    idx = torch.linspace(0, total_frames - 1, nframes).round().long()
    sample_fps = nframes / max(total_frames, 1e-6) * video_fps
    video = video[idx]
    return video, sample_fps


def is_decord_available() -> bool:
    import importlib.util

    return importlib.util.find_spec("decord") is not None


def _read_video_decord(
    ele: dict,
) -> (torch.Tensor, float):
    """read video using decord.VideoReader

    Args:
        ele (dict): a dict contains the configuration of video.
        support keys:
            - video: the path of video. support "file://", "http://", "https://" and local path.
            - video_start: the start time of video.
            - video_end: the end time of video.
    Returns:
        torch.Tensor: the video tensor with shape (T, C, H, W).
    """
    import decord
    video_path = ele["video"]
    st = time.time()
    vr = decord.VideoReader(video_path)
    # TODO: support start_pts and end_pts
    if 'video_start' in ele or 'video_end' in ele:
        raise NotImplementedError("not support start_pts and end_pts in decord for now.")
    total_frames, video_fps = len(vr), vr.get_avg_fps()
    logger.info(f"decord:  {video_path=}, {total_frames=}, {video_fps=}, time={time.time() - st:.3f}s")
    nframes = smart_nframes(ele, total_frames=total_frames, video_fps=video_fps)
    idx = torch.linspace(0, total_frames - 1, nframes).round().long().tolist()
    video = vr.get_batch(idx).asnumpy()
    video = torch.tensor(video).permute(0, 3, 1, 2)  # Convert to TCHW format
    sample_fps = nframes / max(total_frames, 1e-6) * video_fps
    return video, sample_fps


VIDEO_READER_BACKENDS = {
    "decord": _read_video_decord,
    "torchvision": _read_video_torchvision,
}

FORCE_QWENVL_VIDEO_READER = os.getenv("FORCE_QWENVL_VIDEO_READER", None)


@lru_cache(maxsize=1)
def get_video_reader_backend() -> str:
    if FORCE_QWENVL_VIDEO_READER is not None:
        video_reader_backend = FORCE_QWENVL_VIDEO_READER
    elif is_decord_available():
        video_reader_backend = "decord"
    else:
        video_reader_backend = "torchvision"
    print(f"qwen-vl-utils using {video_reader_backend} to read video.", file=sys.stderr)
    return video_reader_backend


def fetch_video(ele: dict, image_factor: int = IMAGE_FACTOR, return_video_sample_fps: bool = False) -> torch.Tensor | list[Image.Image]:
    if isinstance(ele["video"], str):
        video_reader_backend = get_video_reader_backend()
        try:
            video, sample_fps = VIDEO_READER_BACKENDS[video_reader_backend](ele)
        except Exception as e:
            logger.warning(f"video_reader_backend {video_reader_backend} error, use torchvision as default, msg: {e}")
            video, sample_fps = VIDEO_READER_BACKENDS["torchvision"](ele)

        nframes, _, height, width = video.shape
        min_pixels = ele.get("min_pixels", VIDEO_MIN_PIXELS)
        total_pixels = ele.get("total_pixels", VIDEO_TOTAL_PIXELS)
        max_pixels = max(min(VIDEO_MAX_PIXELS, total_pixels / nframes * FRAME_FACTOR), int(min_pixels * 1.05))
        max_pixels_supposed = ele.get("max_pixels", max_pixels)
        if max_pixels_supposed > max_pixels:
            logger.warning(f"The given max_pixels[{max_pixels_supposed}] exceeds limit[{max_pixels}].")
        max_pixels = min(max_pixels_supposed, max_pixels)
        if "resized_height" in ele and "resized_width" in ele:
            resized_height, resized_width = smart_resize(
                ele["resized_height"],
                ele["resized_width"],
                factor=image_factor,
            )
        else:
            resized_height, resized_width = smart_resize(
                height,
                width,
                factor=image_factor,
                min_pixels=min_pixels,
                max_pixels=max_pixels,
            )
        video = transforms.functional.resize(
            video,
            [resized_height, resized_width],
            interpolation=InterpolationMode.BICUBIC,
            antialias=True,
        ).float()
        if return_video_sample_fps:
            return video, sample_fps
        return video
    else:
        assert isinstance(ele["video"], (list, tuple))
        process_info = ele.copy()
        process_info.pop("type", None)
        process_info.pop("video", None)
        images = [
            fetch_image({"image": video_element, **process_info}, size_factor=image_factor)
            for video_element in ele["video"]
        ]
        nframes = ceil_by_factor(len(images), FRAME_FACTOR)
        if len(images) < nframes:
            images.extend([images[-1]] * (nframes - len(images)))
        if return_video_sample_fps:
            return images, process_info.pop("fps", 2.0)
        return images


def extract_vision_info(conversations: list[dict] | list[list[dict]]) -> list[dict]:
    vision_infos = []
    if isinstance(conversations[0], dict):
        conversations = [conversations]
    for conversation in conversations:
        for message in conversation:
            if isinstance(message["content"], list):
                for ele in message["content"]:
                    if (
                        "image" in ele
                        or "image_url" in ele
                        or "video" in ele
                        or ele["type"] in ("image", "image_url", "video")
                    ):
                        vision_infos.append(ele)
    return vision_infos


def process_vision_info(
    conversations: list[dict] | list[list[dict]],
    return_video_kwargs: bool = False,
) -> tuple[list[Image.Image] | None, list[torch.Tensor | list[Image.Image]] | None, Optional[dict]]:

    vision_infos = extract_vision_info(conversations)
    ## Read images or videos
    image_inputs = []
    video_inputs = []
    video_sample_fps_list = []
    for vision_info in vision_infos:
        if "image" in vision_info or "image_url" in vision_info:
            image_inputs.append(fetch_image(vision_info))
        elif "video" in vision_info:
            video_input, video_sample_fps = fetch_video(vision_info, return_video_sample_fps=True)
            video_sample_fps_list.append(video_sample_fps)
            video_inputs.append(video_input)
        else:
            raise ValueError("image, image_url or video should in content.")
    if len(image_inputs) == 0:
        image_inputs = None
    if len(video_inputs) == 0:
        video_inputs = None
    if return_video_kwargs:
        return image_inputs, video_inputs, {'fps': video_sample_fps_list}
    return image_inputs, video_inputs


# Assuming verl utils are importable
import verl.utils.torch_functional as verl_F
from verl.utils.model import compute_position_id_with_mask
import datasets # Ensure datasets is imported
from collections import defaultdict # Ensure defaultdict is imported
from typing import Union, List, Optional

logger = logging.getLogger(__name__)

# --- Include the robust collate_fn here ---
# (See previous responses for the full collate_fn code that handles None, np arrays)
def collate_fn(data_list: list[dict]) -> dict:
    tensors = defaultdict(list)
    non_tensors = defaultdict(list)
    valid_data_list = [data for data in data_list if data is not None]
    if not valid_data_list: return {}
    # --- Batching Logic (handle tensors, np arrays, others) ---
    for data in valid_data_list:
        for key, val in data.items():
            if isinstance(val, torch.Tensor): tensors[key].append(val)
            elif isinstance(val, np.ndarray): non_tensors[key].append(val) # Keep as list of arrays for now
            else: non_tensors[key].append(val)
    for key, val in tensors.items():
        try: tensors[key] = torch.stack(val, dim=0)
        except Exception as e:
            logger.warning(f"Collate: Error stacking tensor key '{key}': {e}. Keeping as list in non_tensors.")
            if key not in non_tensors: non_tensors[key] = []
            non_tensors[key].extend(val) # Add elements to non_tensors list
    for key, val in list(tensors.items()): # Iterate over copy if removing items
        if key in non_tensors: del tensors[key] # Remove key if moved
    for key, val in non_tensors.items():
        if isinstance(val, list) and all(isinstance(elem, np.ndarray) for elem in val): non_tensors[key] = val # Keep list of np arrays
        else:
             try:
                 if not (isinstance(val, list) and all(isinstance(elem, np.ndarray) for elem in val)):
                     # Only convert if not already list of arrays
                     non_tensors[key] = np.array(val, dtype=object)
             except Exception as e:
                 logger.warning(f"Collate: Could not convert key '{key}' to numpy object array: {e}. Keeping as list.")
                 non_tensors[key] = val # Keep as list if conversion fails
    final_batch = {}; final_batch.update(non_tensors); final_batch.update(tensors)
    return final_batch
# --- End collate_fn ---


class TwoStageVideoQADataset(Dataset):
    """
    Dataset loader providing inputs for a two-stage Video QA process.
    Prepares Stage 1 inputs and raw components for Stage 2.
    """
    def __init__(
        self,
        data_files: Union[str, List[str]],
        tokenizer: PreTrainedTokenizer,
        config: DictConfig,
        processor: Optional[ProcessorMixin] = None,
    ):
        if processor is None: raise ValueError("Processor is required.")
        if not isinstance(data_files, (List, ListConfig)): data_files = [data_files]
        # --- Store essential components ---
        self.data_files = copy.deepcopy(data_files); self.original_data_files = copy.deepcopy(data_files)
        self.tokenizer = tokenizer; self.processor = processor; self.config = config

        # --- Configuration ---
        self.cache_dir = os.path.expanduser(config.get("cache_dir", "~/.cache/verl/videoqa_2stage"))
        # Keys for data fields
        self.action_key = config.get("action_key", "action") # Text describing the action for grounding
        self.question_key = config.get("question_key", "question") # QA question text + options
        self.answer_key = config.get("answer_key", "answer") # Target QA answer letter
        self.video_id_key = config.get("video_id_key", "video_id")
        self.temproal_key = config.get("temporal_key", "temporal_grounding") # Optional, for Stage 2
        # Video file parameters
        self.video_base_path = config.get("video_base_path", '/home/chendong/video-rl/charades_sta/Charades_v1_480')
        if not self.video_base_path or not os.path.isdir(self.video_base_path):
             raise ValueError(f"`video_base_path` ('{self.video_base_path}') must be specified and exist.")
        self.video_extension = config.get("video_extension", ".mp4")
        # Prompting and processing parameters
        self.grounding_prompt_template = config.get("grounding_prompt_template", "<video> Find the start and end time for the action: {}")
        self.qa_prompt_prefix = config.get("qa_prompt_prefix", "<video> ") # Prefix only for QA prompt
        self.max_prompt_length = config.get("max_prompt_length", 1024) # Max length for Stage 1 tokenized input
        self.system_prompt = config.get("system_prompt", None) # Applied to both stages if present
        self.truncation = config.get("truncation", "error")
        self.filter_overlong_prompts = config.get("filter_overlong_prompts", True)
        self.num_workers = min(config.get("filter_overlong_prompts_workers", max(1, os.cpu_count() // 4)), os.cpu_count())
        # Video processing config for FULL video (Stage 1) AND for resampling (Stage 2)
        # Store it once to be passed to fit loop
        self.video_processing_config = {
             "fps": config.get("video_sample_fps", FPS), # Use constants from vision_utils
             "nframes": config.get("video_sample_nframes", None),
             "min_frames": config.get("video_min_frames", FPS_MIN_FRAMES),
             "max_frames": config.get("video_max_frames", FPS_MAX_FRAMES),
             "min_pixels": config.get("video_min_pixels", VIDEO_MIN_PIXELS),
             "total_pixels": config.get("video_total_pixels", VIDEO_TOTAL_PIXELS),
             "image_factor": config.get("video_image_factor", IMAGE_FACTOR),
        }
        self.serialize_dataset = False
        self.add_log_keys = config.get("add_log_keys", True)

        # --- Load Data ---
        self._download()
        self._read_files_and_tokenize() # Uses action_key + question_key for filtering now

    # --- _download, _read_files_and_tokenize, resume_dataset_state, __len__, etc. ---
    # (Use robust versions from previous response - Ensure _read_files_and_tokenize checks necessary keys)
    def _download(self, use_origin_files=False):
        # Reuse download logic, assuming verl.utils.fs or similar
        try: from verl.utils.fs import copy_to_local; fs_available = True
        except ImportError: fs_available = False; logger.warning("verl.utils.fs not found. Assuming files are local.")
        data_files = self.data_files if not use_origin_files else self.original_data_files
        new_data_files = []
        for i, file_path in enumerate(data_files):
             is_remote = not os.path.exists(file_path) and ('://' in file_path or not os.path.isabs(file_path))
             if is_remote and fs_available:
                  logger.info(f"Attempting to copy remote file: {file_path}")
                  try: new_path = copy_to_local(src=file_path, cache_dir=self.cache_dir); new_data_files.append(new_path)
                  except Exception as e: logger.error(f"Failed to copy {file_path}: {e}. Skipping.")
             elif is_remote and not fs_available: logger.error(f"File seems remote but copy utility unavailable: {file_path}. Skipping.")
             elif not os.path.exists(file_path): logger.error(f"Local file not found: {file_path}. Skipping.")
             else: logger.info(f"Using local file: {file_path}"); new_data_files.append(file_path)
        self.data_files = new_data_files

    def _read_files_and_tokenize(self):
        dataframes = []
        if not self.data_files: raise ValueError("No valid data files.")
        for data_file_path in self.data_files:
            file_type = None
            if data_file_path.endswith(".json"): file_type = "json"
            elif data_file_path.endswith(".jsonl"): file_type = "json"
            else: logger.warning(f"Skipping unsupported file: {data_file_path}"); continue
            try:
                logger.info(f"Loading dataset: {data_file_path}")
                dataframe = datasets.load_dataset(file_type, data_files=data_file_path)["train"]
                # Check keys needed for this dataset structure
                required_keys = [self.action_key, self.question_key, self.answer_key, self.video_id_key]
                missing_keys = [k for k in required_keys if k not in dataframe.column_names]
                if missing_keys: logger.error(f"File {data_file_path} missing keys: {missing_keys}. Skipping."); continue
                dataframes.append(dataframe)
            except Exception as e: logger.error(f"Error loading {data_file_path}: {e}")
        if not dataframes: raise ValueError("No dataframes loaded.")
        self.dataframe = datasets.concatenate_datasets(dataframes) if len(dataframes) > 1 else dataframes[0]
        logger.info(f"Loaded dataset length: {len(self.dataframe)}")
        # Filtering (check combined length?) - still approximate
        if self.filter_overlong_prompts:
            tokenizer = self.tokenizer; initial_len = len(self.dataframe)
            action_k, question_k = self.action_key, self.question_key
            try:
                # Estimate combined length (without tags/template)
                self.dataframe = self.dataframe.filter(
                    lambda doc: len(tokenizer.encode(doc[action_k] + " " + doc[question_k], add_special_tokens=False)) <= self.max_prompt_length,
                    num_proc=self.num_workers, desc=f"Filtering items > ~{self.max_prompt_length} tokens"
                )
                logger.info(f"Filtered {initial_len - len(self.dataframe)} items. New length: {len(self.dataframe)}")
            except Exception as filter_err: logger.error(f"Error filtering: {filter_err}. Proceeding without filtering.")
        if len(self.dataframe) == 0: logger.warning("Dataset empty after loading/filtering!")

    def resume_dataset_state(self): # Basic reload
        logger.info("Reloading dataset files for resume...")
        self._download(use_origin_files=True)
        self._read_files_and_tokenize()

    def __len__(self): return len(self.dataframe)
    def __getstate__(self): # No dataframe saving
        if not self.serialize_dataset: state = self.__dict__.copy(); state.pop('dataframe', None); return state
        return self.__dict__.copy()
    def __setstate__(self, state):
         self.__dict__.update(state)
         if 'dataframe' not in self.__dict__: logger.info("Restored state w/o dataframe.")
    # --- End standard methods ---

    def __getitem__(self, item):
        """
        Loads data, processes FULL video, prepares tokenized inputs for Stage 1.
        Also returns raw components needed for Stage 2.
        """
        try:
            row_dict: dict = self.dataframe[item]

            # --- 1. Extract core data ---
            action_text = row_dict.get(self.action_key)
            question_text = row_dict.get(self.question_key) # Includes options
            answer_letter = row_dict.get(self.answer_key)
            video_id = row_dict.get(self.video_id_key)
            start_time = row_dict.get(self.temproal_key)["start_time"] # Optional for Stage 2
            end_time = row_dict.get(self.temproal_key)["end_time"] # Optional for Stage 2
            
            if not action_text or not question_text or not answer_letter or not video_id:
                logger.warning(f"Skipping item {item} due to missing required fields.")
                return None

            # --- 2. Construct Video Path ---
            video_path = os.path.join(self.video_base_path, f"{video_id}{self.video_extension}")
            if not os.path.exists(video_path):
                logger.warning(f"Video file not found: {video_path}. Skipping item {item}.")
                return None

            # --- 3. Fetch and Process FULL Video (for Stage 1) ---
            ele_full_video = {"video": video_path, **self.video_processing_config}
            full_video_frames_tensor = None
            try:
                # Use fetch_video to get processed tensor TCHW for the *whole* video
                full_video_frames_tensor = fetch_video(ele_full_video, image_factor=self.video_processing_config["image_factor"])
            except Exception as e:
                 logger.error(f"Error processing FULL video for item {item} ({video_path}): {e}", exc_info=True)
                 return None
            if full_video_frames_tensor is None or full_video_frames_tensor.nelement() == 0:
                 logger.warning(f"FULL video processing resulted in empty tensor for item {item}. Skipping.")
                 return None

            # --- 4. Prepare Stage 1 Tokenized Inputs (Grounding Prompt + Full Video) ---
            stage1_messages = []
            if self.system_prompt:
                stage1_messages.append({"role": "system", "content": self.system_prompt})
            # Format grounding prompt using template and action text
            grounding_user_content = self.grounding_prompt_template.format(action_text)
            stage1_messages.append({"role": "user", "content": grounding_user_content})

            stage1_input_ids, stage1_attention_mask, stage1_position_ids = None, None, None
            stage1_model_inputs_remaining = {}
            try:
                # Tokenize Stage 1 prompt WITH full video features
                stage1_raw_prompt = self.processor.apply_chat_template(stage1_messages, add_generation_prompt=True, tokenize=False)
                stage1_model_inputs = self.processor(text=[stage1_raw_prompt], images=None, videos=[full_video_frames_tensor], return_tensors="pt")

                if "input_ids" not in stage1_model_inputs or "attention_mask" not in stage1_model_inputs:
                     raise ValueError("Processor output missing keys for Stage 1")
                s1_input_ids_raw = stage1_model_inputs.pop("input_ids")
                s1_attn_mask_raw = stage1_model_inputs.pop("attention_mask")

                # Pad/Truncate Stage 1 input
                stage1_input_ids, stage1_attention_mask = verl_F.postprocess_data(
                     input_ids=s1_input_ids_raw, attention_mask=s1_attn_mask_raw, max_length=self.max_prompt_length,
                     pad_token_id=self.tokenizer.pad_token_id, left_pad=True, truncation=self.truncation
                 )

                # Calculate Stage 1 Position IDs
                stage1_model_inputs_remaining = dict(stage1_model_inputs) # Keep remaining (e.g., grid info)
                if hasattr(self.processor, 'image_processor') and \
                   self.processor.image_processor.__class__.__name__ == "Qwen2VLImageProcessor":
                     try:
                         from verl.models.transformers.qwen2_vl import get_rope_index
                         s1_pos_ids_list = [ get_rope_index(self.processor, input_ids=stage1_input_ids[0], **stage1_model_inputs_remaining, attention_mask=stage1_attention_mask[0]) ]
                         stage1_position_ids = s1_pos_ids_list[0]
                     except ImportError: position_ids = compute_position_id_with_mask(stage1_attention_mask)[0]; logger.warning("Qwen func not found.") # Basic fallback
                     except KeyError as ke: position_ids = compute_position_id_with_mask(stage1_attention_mask)[0]; logger.warning(f"KeyError Qwen pos ID: {ke}") # Basic fallback
                else:
                     stage1_position_ids = compute_position_id_with_mask(stage1_attention_mask)[0]

            except RuntimeError as trunc_err:
                 logger.error(f"Stage 1 prompt too long for item {item}: {trunc_err}. Skipping.")
                 return None
            except Exception as stage1_err:
                 logger.error(f"Error preparing stage 1 inputs for item {item}: {stage1_err}", exc_info=True)
                 return None

            # --- 5. Prepare Final Output Dictionary ---
            # This dictionary contains inputs ready for STAGE 1, and raw data for STAGE 2 prep
            final_output_dict = {
                # == Stage 1 Inputs (Processed) ==
                "input_ids": stage1_input_ids[0],        # Remove batch dim
                "attention_mask": stage1_attention_mask[0],# Remove batch dim
                "position_ids": stage1_position_ids,     # Shape depends on calculation
                "multi_modal_inputs": stage1_model_inputs_remaining, # Other outputs like pixel_values

                # == Raw Components for Stage 2 (Used by fit loop) ==
                "question_text": question_text,                 # QA Question + Options
                "ground_truth_answer": answer_letter,           # Target letter for final reward
                "video_path": video_path,                       # Raw path for resampling
                "video_processing_config": self.video_processing_config, # Params for resampling
                "start_time": start_time,
                "end_time": end_time,

                # == Other Useful Info ==
                "video_id": video_id,
                "action_text": action_text,                     # Original action text
            }

            # Add Optional Logging Keys If Configured (for standard fit loop logging)
            # Note: These keys might be confusing in a two-stage context if not handled carefully in fit
            if self.add_log_keys:
                 # Create a combined prompt string for general logging purposes
                 log_prompt_str = ""
                 if self.system_prompt: log_prompt_str += f"System: {self.system_prompt}\n"
                 # Include BOTH prompts for context? Or just stage 1? Let's include both for now.
                 log_prompt_str += f"Stage1: {grounding_user_content}\nStage2_Q: {self.qa_prompt_prefix + question_text}"
                 final_output_dict["prompts"] = log_prompt_str
                 final_output_dict["videos"] = [video_path] # Keep original video path for logging

            return final_output_dict

        except Exception as e:
            logger.error(f"FATAL Error processing item {item}: {e}", exc_info=True)
            return None # Skip item on any unexpected error