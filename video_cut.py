import os
import cv2
import random
import numpy as np
import json
import argparse
from tqdm import tqdm
from reconstruction import colmap_reconstruction
import logging
from pathlib import Path
import torch
import signal
import time
import pycolmap
import multiprocessing
from functools import partial
import torch.multiprocessing as mp

class TimeoutException(Exception):
    pass

def timeout_handler(signum, frame):
    raise TimeoutException("Task timed out")

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

def process_single_video(video_name, base_path, output_base_dir, version, seed, gpu_id, durations, timeout):
    """Process a single video on specified GPU."""
    # Set independent deterministic random seed for each process
    process_seed = seed + hash(video_name) % 10000
    random.seed(process_seed)
    np.random.seed(process_seed)
    torch.manual_seed(process_seed)
    torch.cuda.manual_seed(process_seed)
    
    # Set GPU device
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    torch.cuda.set_device(0)
    
    # Set PyTorch to deterministic mode
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    video_path = os.path.join(base_path, 'video_1080p/' + video_name + '.mp4')
    video_basename = os.path.splitext(os.path.basename(video_name))[0]
    
    # Store camera parameters for different durations
    camera_params = {}
    camera_params_out_duration = []
    
    for duration in durations:
        try:
            output_dir = os.path.join(output_base_dir, video_basename, f"{duration}s")
            video_seed = process_seed + duration
            
            # Set signal handler
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(timeout)
            
            start_time = time.time()
            extract_frames(video_path, output_dir + '/images', duration, video_seed)
            colmap_reconstruction(version, root_dir=output_dir)

            reconstruction_dir = output_dir + '/' + version + '/sparse'
            for filename in ['images.bin', 'cameras.bin', 'points3D.bin']:
                if not Path(reconstruction_dir + '/' + filename).exists():
                    logging.error(f"{video_name}: {filename} reconstruction failed")
                    continue

            reconstruction = pycolmap.Reconstruction(reconstruction_dir)
            # Collect camera parameters instead of printing
            camera_params_in_duration = []
            for _, camera in reconstruction.cameras.items():
                camera_params_in_duration.append(camera.params.tolist())
                camera_params_out_duration.append(camera.params.tolist())
            camera_params_in_duration = np.array(camera_params_in_duration)
            camera_params[duration] = np.concatenate((np.mean(camera_params_in_duration, axis=0), 
                                                        np.std(camera_params_in_duration, axis=0)), axis=-1)
                            
        except Exception as e:
            # Handle all exceptions uniformly
            if isinstance(e, TimeoutException):
                elapsed_time = int(time.time() - start_time)
                error_msg = f"{video_name} {duration}s: Timeout after {elapsed_time} seconds"
            else:
                error_msg = f"{video_name} {duration}s: {str(e)}"
            
            logging.error(error_msg)
            print(f"Error processing {video_name} {duration}s. See log file for details.")
            
        finally:
            # Cancel timer
            signal.alarm(0)
    
    camera_params['total'] = np.concatenate((np.mean(camera_params_out_duration, axis=0), 
                                                        np.std(camera_params_out_duration, axis=0)), axis=-1)
    
    return video_basename, camera_params

def process_videos(base_path, video_list_path, output_base_dir, version, seed=42, durations=None, timeout=3600):
    """Process videos in parallel using multiple GPUs."""
    # Read video list
    video_list = sorted(read_video_list(os.path.join(base_path, video_list_path)))
    
    if durations is None:
        durations = [30, 60, 120]  # Default durations if not specified
    
    # Set global random seeds
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # Set up logging
    log_path = Path('reconstruction_out') / 'reconstruction_errors.log'
    logging.basicConfig(
        filename=log_path,
        filemode='w',
        level=logging.ERROR,
        format='%(message)s'
    )
    
    # Get number of available GPUs
    num_gpus = torch.cuda.device_count()
    processes_per_gpu = 4
    total_processes = num_gpus * processes_per_gpu
    
    # Create process pool
    mp.set_start_method('spawn', force=True)
    pool = mp.Pool(processes=total_processes)
    
    # Prepare arguments for each video
    process_args = []
    for i, video_name in enumerate(video_list):
        gpu_id = (i // processes_per_gpu) % num_gpus
        process_args.append((video_name, base_path, output_base_dir, version, seed, gpu_id, durations, timeout))
    
    # Process videos in parallel with fixed chunk size
    results = pool.starmap(process_single_video, process_args, chunksize=1)
    
    # Close the pool
    pool.close()
    pool.join()
    
    # Combine results
    all_camera_params = {}
    for video_basename, camera_params in results:
        if camera_params:  # Only add if we have valid results
            all_camera_params[video_basename] = camera_params
    
    # Save results
    np.savez(os.path.join(output_base_dir, 'camera_stats.npz'), **all_camera_params)
    print(f"Camera statistics saved to {os.path.join(output_base_dir, 'camera_stats.npz')}")



def main():
    parser = argparse.ArgumentParser(description="Extract frames from videos.")
    parser.add_argument("--base_path", required=True, help="Base path containing videos")
    parser.add_argument("--video_list", required=True, help="Path to text file with video list")
    parser.add_argument("--output_dir", required=True, help="Base output directory for frames")
    parser.add_argument('--version', type=str, choices={'gim_dkm', 'gim_lightglue'}, default='gim_dkm')
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--durations", type=int, nargs='+', default=[30, 60, 120],
                      help="List of video segment durations in seconds (default: [30, 60, 120])")
    parser.add_argument("--timeout", type=int, default=3600,
                      help="Timeout for processing each video segment in seconds (default: 3600)")
    
    args = parser.parse_args()

    process_videos(
        args.base_path,
        args.video_list,
        args.output_dir,
        args.version,
        args.seed,
        args.durations,
        args.timeout
    )

if __name__ == "__main__":
    main()
