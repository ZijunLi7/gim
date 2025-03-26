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

def process_videos(base_path, video_list_path, output_base_dir, version, seed=42):
    """Process all videos in the list."""
    # Read video list
    video_list = read_video_list(os.path.join(base_path, video_list_path))
    
    # Create a fixed iterator with the seed
    random.seed(seed)
    np.random.seed(seed)
    
    # Segment durations in seconds
    durations = [30, 60, 120]
    
    # 设置日志
    log_path = Path('reconstruction_out') / 'reconstruction_errors.log'
    logging.basicConfig(
        filename=log_path,
        filemode='w',
        level=logging.ERROR,
        format='%(message)s'
    )
    
    TIMEOUT = 3600 
    
    # 创建字典存储所有视频的相机参数
    all_camera_params = {}

    # Process each video
    for video_name in tqdm(video_list, desc="Processing videos"):
        video_path = os.path.join(base_path, 'video_1080p/' + video_name + '.mp4')
        video_basename = os.path.splitext(os.path.basename(video_name))[0]
        
        # 存储当前视频不同duration的相机参数
        all_camera_params[video_basename] = {}
        camera_params_out_duration = []
        
        for duration in durations:
            try:
                output_dir = os.path.join(output_base_dir, video_basename, f"{duration}s")
                video_seed = seed + duration + hash(video_name) % 10000
                
                # 设置信号处理器
                signal.signal(signal.SIGALRM, timeout_handler)
                signal.alarm(TIMEOUT)
                
                start_time = time.time()
                extract_frames(video_path, output_dir + '/images', duration, video_seed)
                colmap_reconstruction(version, root_dir=output_dir)

                reconstruction_dir = output_dir + '/' + version + '/sparse'
                for filename in ['images.bin', 'cameras.bin', 'points3D.bin']:
                    if not Path(reconstruction_dir + '/' + filename).exists():
                        logging.error(f"{video_name}: {filename} reconstruction failed")
                        continue

                reconstruction = pycolmap.Reconstruction(reconstruction_dir)
                # 收集相机参数而不是打印
                camera_params_in_duration = []
                for _, camera in reconstruction.cameras.items():
                    camera_params_in_duration.append(camera.params.tolist())
                    camera_params_out_duration.append(camera.params.tolist())
                camera_params_in_duration = np.array(camera_params_in_duration)
                all_camera_params[video_basename][duration] = np.concatenate((np.mean(camera_params_in_duration, axis=0), 
                                                                        np.std(camera_params_in_duration, axis=0)), axis=-1)
                            
            except Exception as e:
                # 统一处理所有异常
                if isinstance(e, TimeoutException):
                    elapsed_time = int(time.time() - start_time)
                    error_msg = f"{video_name} {duration}s: Timeout after {elapsed_time} seconds"
                else:
                    error_msg = f"{video_name} {duration}s: {str(e)}"
                
                logging.error(error_msg)
                print(f"Error processing {video_name} {duration}s. See log file for details.")
                
            finally:
                # 取消定时器
                signal.alarm(0)
        
        all_camera_params[video_basename]['total'] = np.concatenate((np.mean(camera_params_out_duration, axis=0), 
                                                                        np.std(camera_params_out_duration, axis=0)), axis=-1)
    
    # 保存所有视频的相机参数统计信息到NPZ文件
    np.savez(os.path.join(output_base_dir, 'camera_stats.npz'), **all_camera_params)
    print(f"Camera statistics saved to {os.path.join(output_base_dir, 'camera_stats.npz')}")

def calculate_camera_stats(camera_params_by_duration):
    """计算不同duration下相机参数的均值和方差"""
    # 收集所有相机参数
    all_params = []
    
    # 遍历所有duration
    for duration, cameras in camera_params_by_duration.items():
        for camera_id, camera in cameras.items():
            all_params.append(camera['params'])
    
    # 转换为numpy数组以便计算
    if all_params:
        params_array = np.array(all_params)
        mean_params = np.mean(params_array, axis=0)
        var_params = np.var(params_array, axis=0)
        
        return {
            'mean': mean_params,
            'variance': var_params,
            'count': len(all_params),
            'raw_data': camera_params_by_duration  # 可选：保存原始数据
        }
    else:
        return {
            'mean': None,
            'variance': None,
            'count': 0,
            'raw_data': camera_params_by_duration
        }

def main():
    parser = argparse.ArgumentParser(description="Extract frames from videos.")
    parser.add_argument("--base_path", required=True, help="Base path containing videos")
    parser.add_argument("--video_list", required=True, help="Path to text file with video list")
    parser.add_argument("--output_dir", required=True, help="Base output directory for frames")
    parser.add_argument('--version', type=str, choices={'gim_dkm', 'gim_lightglue'}, default='gim_dkm')
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    process_videos(
        args.base_path,
        args.video_list,
        args.output_dir,
        args.version,
        args.seed
    )

if __name__ == "__main__":
    main()
