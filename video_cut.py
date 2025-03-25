import os
import cv2
import random
import numpy as np
import json
import argparse
from tqdm import tqdm
from reconstruction import colmap_reconstruction

def read_video_list(video_list_path):
    """Read the list of videos from a text file."""
    with open(video_list_path, 'r') as f:
        videos = [line.strip() for line in f.readlines()]
    return videos

def extract_frames(video_path, output_dir, segment_duration, seed):
    """Extract frames from a video segment at 3 frames per second."""
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Open the video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return
    
    # Get video properties
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    video_duration = total_frames / video_fps  # Duration in seconds
    
    # Check if video is long enough
    if video_duration < segment_duration:
        print(f"Warning: Video {video_path} is shorter than the requested segment duration ({segment_duration}s)")
        segment_duration = video_duration
    
    # Set random seed for reproducibility
    random_generator = random.Random(seed)
    
    # Calculate start frame for random segment
    max_start_time = video_duration - segment_duration
    start_time = random_generator.uniform(0, max(0, max_start_time))
    start_frame = int(start_time * video_fps)
    
    # Calculate sampling rate to get 3 frames per second
    sampling_rate = int(video_fps / 3)
    if sampling_rate < 1:
        sampling_rate = 1
    
    # Position the video at the start frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    
    # Calculate the total number of frames to process
    frames_to_process = int(segment_duration * video_fps)
    
    # Extract frames
    for i in range(0, frames_to_process, sampling_rate):
        success, frame = cap.read()
        if not success:
            break
        
        # Skip frames according to sampling rate
        for _ in range(sampling_rate - 1):
            cap.read()
        
        # Calculate timestamp in HH_MM_SS_mmm format
        total_seconds = start_time + (i / video_fps)
        hours = int(total_seconds // 3600)
        minutes = int((total_seconds % 3600) // 60)
        seconds = int(total_seconds % 60)
        milliseconds = int((total_seconds * 1000) % 1000)
        
        frame_filename = f"{hours:02d}_{minutes:02d}_{seconds:02d}_{milliseconds:09d}.jpg"
        output_path = os.path.join(output_dir, frame_filename)
        
        # Save the frame
        cv2.imwrite(output_path, frame)
    
    # Release the video capture
    cap.release()

def process_videos(base_path, video_list_path, output_base_dir, version, seed=42):
    """Process all videos in the list."""
    # Read video list
    video_list = read_video_list(os.path.join(base_path, video_list_path))
    
    # # Read metadata
    # metadata = read_metadata(os.path.join(base_path, metadata_path))
    
    # Create a fixed iterator with the seed
    random.seed(seed)
    np.random.seed(seed)
    
    # Segment durations in seconds
    durations = [30, 60, 120]
    
    # Process each video
    for video_name in tqdm(video_list, desc="Processing videos"):
        video_path = os.path.join(base_path, 'video_1080p/' + video_name + '.mp4')
        
        # Get video filename without extension
        video_basename = os.path.splitext(os.path.basename(video_name))[0]
        
        # Process different segment durations
        for duration in durations:
            output_dir = os.path.join(output_base_dir, video_basename, f"{duration}s")
            # Use video name as part of the seed for variation between videos
            video_seed = seed + duration + hash(video_name) % 10000
            extract_frames(video_path, output_dir + '/images', duration, video_seed)
            colmap_reconstruction(version, root_dir=output_dir)

def main():
    parser = argparse.ArgumentParser(description="Extract frames from videos.")
    parser.add_argument("--base_path", required=True, help="Base path containing videos")
    parser.add_argument("--video_list", required=True, help="Path to text file with video list")
    parser.add_argument("--output_dir", required=True, help="Base output directory for frames")
    parser.add_argument('--version', type=str, choices={'gim_dkm', 'gim_lightglue'}, default='gim_dkm')
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    
    args = parser.parse_args()
    
    process_videos(
        args.base_path,
        args.video_list,
        args.output_dir,
        args.version,
        args.seed
    )

if __name__ == "__main__":
    main()
