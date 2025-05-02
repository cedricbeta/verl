# video-verl/video_utils.py
import decord
import torch
import numpy as np
from typing import List, Tuple, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def sample_frames_from_interval(
    video_path: str,
    start_time: float,
    end_time: float,
    num_frames: int = 8, # Number of frames to sample from the clip
    # sampling_rate: int = 30, # Often better to rely on video's actual FPS
    target_fps: Optional[int] = None, # Optional: Resample to a target FPS (more complex)
) -> Optional[torch.Tensor]:
    """
    Samples frames from a video file within a given time interval.

    Args:
        video_path: Path to the video file.
        start_time: Start time of the interval in seconds.
        end_time: End time of the interval in seconds.
        num_frames: Number of frames to sample uniformly from the interval.
        target_fps: If set, resamples frames to this rate before selection (not fully implemented here).

    Returns:
        A Tensor of shape (num_frames, H, W, C) or None if an error occurs.
    """
    try:
        # Ensure start_time is not negative and end_time is >= start_time
        start_time = max(0.0, start_time)
        end_time = max(start_time, end_time)

        # Use CPU context for decord unless GPU is specifically needed and managed
        # Using GPU might require careful memory management in a distributed setting
        ctx = decord.cpu(0)
        vr = decord.VideoReader(video_path, ctx=ctx)
        video_fps = vr.get_avg_fps()
        total_frame_count = len(vr)

        if total_frame_count == 0:
            logger.warning(f"Video {video_path} has 0 frames.")
            return None

        if video_fps <= 0:
             logger.warning(f"Warning: Invalid FPS {video_fps} for video {video_path}. Using estimated rate based on duration if possible, else skipping.")
             # Cannot reliably calculate frame indices without FPS
             return None # Or attempt estimation if duration is known? Too risky.

        # Calculate start and end frame indices
        start_frame = int(start_time * video_fps)
        end_frame = int(end_time * video_fps)

        # Clip indices to valid range [0, total_frame_count - 1]
        start_frame = max(0, start_frame)
        end_frame = min(total_frame_count - 1, end_frame)

        # Ensure start_frame is not greater than end_frame after clipping
        if start_frame > end_frame:
             logger.warning(f"Calculated start frame {start_frame} > end frame {end_frame} for interval [{start_time}, {end_time}] in {video_path}. Sampling single frame at start.")
             # Option 1: Sample single frame at start_frame (clipped)
             frame_indices = [start_frame]
             # Option 2: Return None as interval is invalid after clipping
             # return None
        elif start_frame == end_frame:
            # If interval maps to a single frame, sample that frame num_frames times
            frame_indices = [start_frame] * num_frames
        else:
            # Generate indices to sample uniformly across the interval
            # Use linspace to get evenly spaced indices including start and end
            indices = np.linspace(start_frame, end_frame, num=num_frames, dtype=int)
            # Ensure indices are within the actual video bounds (redundant due to clipping?)
            indices = np.clip(indices, 0, total_frame_count - 1)
            frame_indices = sorted(list(set(indices))) # Ensure unique and sorted

            # If fewer unique frames available than num_frames, duplicate the last one
            # Or could implement more sophisticated padding/sampling strategies
            if len(frame_indices) < num_frames:
                 padding_needed = num_frames - len(frame_indices)
                 frame_indices.extend([frame_indices[-1]] * padding_needed)
            frame_indices = frame_indices[:num_frames] # Ensure exactly num_frames


        # Handle target FPS resampling (complex, omitted for brevity)
        # if target_fps and video_fps > 0:
        #     # This requires calculating which original frames correspond
        #     # to the desired target_fps timestamps within the interval.
        #     logger.warning("Target FPS resampling is not fully implemented.")
        #     pass

        if not frame_indices:
             logger.warning(f"No valid frame indices generated for {video_path} [{start_time}-{end_time}]")
             return None

        # Read the selected frames
        # get_batch is efficient for reading multiple frames
        frames = vr.get_batch(frame_indices).asnumpy()

        # Expected output shape: (num_frames, H, W, C)
        if frames.shape[0] != num_frames:
             logger.warning(f"Requested {num_frames} frames but got {frames.shape[0]} for {video_path}. Padding/adjusting.")
             # Handle discrepancy if needed (e.g., padding)
             if frames.shape[0] == 0: return None # No frames read
             diff = num_frames - frames.shape[0]
             if diff > 0:
                 padding = np.repeat(frames[-1:], diff, axis=0)
                 frames = np.concatenate([frames, padding], axis=0)
             else: # Too many frames? Should not happen with get_batch(indices)
                 frames = frames[:num_frames]


        return torch.from_numpy(frames)

    except decord.DECORDError as e:
        logger.error(f"DECORD Error processing video {video_path}: {e}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error sampling frames from {video_path} [{start_time}-{end_time}]: {e}")
        return None

