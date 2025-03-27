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
from contextlib import contextmanager
import threading
import _thread
import shutil

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

@contextmanager
def time_limit(seconds):
    timer = threading.Timer(seconds, lambda: _thread.interrupt_main())
    timer.start()
    try:
        yield
    except KeyboardInterrupt:
        raise TimeoutException("Task timed out")
    finally:
        timer.cancel()

def init_worker():
    """Initialize worker process before running."""
    import torch
    torch.set_num_threads(1)  # 限制每个进程的 CPU 线程数

def process_single_video(video_name, base_path, output_base_dir, version, seed, gpu_id, durations, timeout):
    """Process a single video on specified GPU."""
    # Set GPU device before importing torch and other GPU related modules
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    
    # Import torch and set device after setting CUDA_VISIBLE_DEVICES
    import torch
    torch.cuda.set_device(0)
    
    # Set independent deterministic random seed for each process
    process_seed = seed + hash(video_name) % 10000
    random.seed(process_seed)
    np.random.seed(process_seed)
    torch.manual_seed(process_seed)
    torch.cuda.manual_seed(process_seed)
    
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
            reconstruction_dir = output_dir + '/' + version + '/sparse'
            video_seed = process_seed + duration
            start_time = time.time()
            
            # 合并所有需要超时控制的操作
            with time_limit(timeout):
                extract_frames(video_path, output_dir + '/images', duration, video_seed)
                colmap_reconstruction(version, root_dir=output_dir)
                
            for filename in ['images.bin', 'cameras.bin', 'points3D.bin']:
                if not Path(reconstruction_dir + '/' + filename).exists():
                    raise RuntimeError(f"{filename} reconstruction failed")

            reconstruction = pycolmap.Reconstruction(reconstruction_dir)
            camera_params_in_duration = []
            for _, camera in reconstruction.cameras.items():
                camera_params_in_duration.append(camera.params.tolist())
                camera_params_out_duration.append(camera.params.tolist())
            camera_params_in_duration = np.array(camera_params_in_duration)
            camera_params[str(duration)] = np.concatenate((np.mean(camera_params_in_duration, axis=0), 
                                                np.std(camera_params_in_duration, axis=0)), axis=-1)
                            
        except TimeoutException:
            try:
                model_path = reconstruction_dir + '/models'
                largest_path = None
                largest_num_images = 0
                for path in os.listdir(model_path):
                    if os.path.isdir(model_path + '/' + path):
                        reconstruction = pycolmap.Reconstruction(model_path + '/' + path)
                        num_images = reconstruction.num_reg_images()
                        if num_images > largest_num_images:
                            largest_path = path
                            largest_num_images = num_images
                assert largest_path is not None
                print(f'Largest model is #{largest_path}'f'with {largest_num_images} images.')

                for filename in ['images.bin', 'cameras.bin', 'points3D.bin']:
                    if Path(reconstruction_dir + '/' + filename).exists():
                        Path(reconstruction_dir + '/' + filename).unlink()
                    shutil.move(
                        str(model_path + '/' + largest_path + '/' + filename), str(reconstruction_dir))

                reconstruction = pycolmap.Reconstruction(reconstruction_dir)
                camera_params_in_duration = []
                for _, camera in reconstruction.cameras.items():
                    camera_params_in_duration.append(camera.params.tolist())
                    camera_params_out_duration.append(camera.params.tolist())
                camera_params_in_duration = np.array(camera_params_in_duration)
                camera_params[str(duration)] = np.concatenate((np.mean(camera_params_in_duration, axis=0), 
                                                    np.std(camera_params_in_duration, axis=0)), axis=-1)
            except Exception as e:
                elapsed_time = int(time.time() - start_time)
                error_msg = f"{video_name} {duration}s: Timeout after {elapsed_time} seconds"
                logging.error(error_msg)
                print(f"Error processing {video_name} {duration}s: Timeout")  
                continue
            
        except Exception as e:
            error_msg = f"{video_name} {duration}s: {str(e)}"
            logging.error(error_msg)
            print(f"Error processing {video_name} {duration}s: {str(e)}")
            continue

    # 检查是否有成功处理的数据
    if not camera_params_out_duration:
        return video_basename, None
    
    camera_params['total'] = np.concatenate((np.mean(camera_params_out_duration, axis=0), 
                                                    np.std(camera_params_out_duration, axis=0)), axis=-1)
    
    # 保存结果到临时文件
    result_path = os.path.join(output_base_dir, f"{video_basename}_result.npz")

    if camera_params:  # 只在有结果时保存
        try:
            np.savez(result_path, **camera_params)
        except Exception as e:
            print(len(camera_params.keys()))

def process_videos(base_path, video_list_path, output_base_dir, version, seed=42, durations=None, timeout=3600):
    """Process videos in parallel using multiple GPUs."""
    # Set start method for multiprocessing
    if mp.get_start_method(allow_none=True) != 'spawn':
        mp.set_start_method('spawn', force=True)
    
    # Read video list
    video_list = sorted(read_video_list(os.path.join(base_path, video_list_path)))
    
    # Create log directory and set up logging
    log_dir = Path('reconstruction_out')
    log_dir.mkdir(exist_ok=True)
    log_path = log_dir / 'reconstruction_errors.log'
    
    logging.basicConfig(
        filename=log_path,
        filemode='w',
        level=logging.ERROR,
        format='%(message)s'
    )
    
    # Get number of available GPUs and control processes per GPU
    num_gpus = torch.cuda.device_count()
    processes_per_gpu = 4  # 每张卡最多运行4个进程
    
    # 将视频列表分组，每组对应一个GPU的最大进程数
    video_groups = []
    for i in range(0, len(video_list), processes_per_gpu):
        video_groups.append(video_list[i:i + processes_per_gpu])
    
    all_results = []
    with tqdm(total=len(video_list), desc="Processing videos") as pbar:
        # 按组处理视频，确保每张卡同时最多运行processes_per_gpu个进程
        for group_idx, video_group in enumerate(video_groups):
            processes = []
            gpu_id = group_idx % num_gpus  # 循环使用GPU
            
            # 启动当前组的所有进程
            for video_name in video_group:
                args = (video_name, base_path, output_base_dir, version, seed, gpu_id, durations, timeout)
                p = mp.Process(target=process_single_video, args=args, daemon=False)
                p.start()
                processes.append(p)
            
            # 等待当前组的所有进程完成
            for p in processes:
                p.join()
                pbar.update(1)
                
            # 收集结果（需要修改process_single_video来支持结果返回，比如通过文件或队列）
            for video_name in video_group:
                video_basename = os.path.splitext(os.path.basename(video_name))[0]
                # 从临时文件或其他方式获取结果
                result_path = os.path.join(output_base_dir, f"{video_basename}_result.npz")
                if os.path.exists(result_path):
                    result = np.load(result_path)
                    all_results.append((video_basename, dict(result)))
                    # os.remove(result_path)  # 清理临时文件
    
    # Combine results
    all_camera_params = {}
    for video_basename, camera_params in all_results:
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
    parser.add_argument("--durations", type=int, nargs='+',
                      default=(30, 60, 120),
                      help="List of video segment durations in seconds (default: 30 60 120)")
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
