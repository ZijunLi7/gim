import os
import cv2
import random
import numpy as np
import json
import argparse
from tqdm import tqdm
from reconstruction import colmap_reconstruction
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
from camera_stats import analyze_camera_stats_magnitude, plot_magnitude_recall

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
    torch.set_num_threads(16)  # 限制每个进程的 CPU 线程数

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
    video_basename = video_name
    
    # Store camera parameters for different durations
    camera_params = {}
    camera_params_out_duration = []
    os.makedirs(os.path.join(output_base_dir, video_basename), exist_ok=True)
    log_path = os.path.join(output_base_dir, video_basename, "reconstruction_errors.txt")
    
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
                    raise FileNotFoundError(f"reconstruction failed")

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
                error_msg = f"{duration}s: Timeout after {elapsed_time} seconds"
                with open(log_path, 'a') as f:
                    f.write(error_msg + '\n')
                print(f"Error processing {video_name} {duration}s: Timeout")  
                continue
            
        except Exception as e:
            error_msg = f"{duration}s: {str(e)}"
            with open(log_path, 'a') as f:
                    f.write(error_msg + '\n')
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

def process_videos(base_path, video_list_path, output_base_dir, version, seed=42, durations=None, timeout=3600, prefix=None):
    output_base_dir = os.path.join(output_base_dir, prefix)
    """Process videos in parallel using multiple GPUs."""
    # Set start method for multiprocessing
    if mp.get_start_method(allow_none=True) != 'spawn':
        mp.set_start_method('spawn', force=True)
    
    # Read video list
    video_list = sorted(read_video_list(os.path.join(base_path, video_list_path)))
    
    # Get number of available GPUs and control processes per GPU
    num_gpus = torch.cuda.device_count()
    processes_per_gpu = 1  # 每张卡恰好运行4个进程
    
    # 按GPU分组视频列表
    gpu_video_groups = [[] for _ in range(num_gpus)]
    for i, video_name in enumerate(video_list):
        gpu_idx = i % num_gpus
        gpu_video_groups[gpu_idx].append(video_name)
    
    all_results = []
    
    with tqdm(total=len(video_list), desc="Processing videos") as pbar:
        # 为每个GPU创建进程跟踪器
        gpu_processes = [[] for _ in range(num_gpus)]
        gpu_queues = [[] for _ in range(num_gpus)]  # 等待处理的视频队列
        
        # 初始化每个GPU的视频队列
        for gpu_id, videos in enumerate(gpu_video_groups):
            gpu_queues[gpu_id] = list(videos)
        
        # 初始化每个GPU的前processes_per_gpu个进程
        for gpu_id in range(num_gpus):
            for _ in range(min(processes_per_gpu, len(gpu_queues[gpu_id]))):
                if gpu_queues[gpu_id]:
                    video_name = gpu_queues[gpu_id].pop(0)
                    args = (video_name, base_path, output_base_dir, version, seed, gpu_id, durations, timeout)
                    p = mp.Process(target=process_single_video, args=args, daemon=False)
                    p.start()
                    gpu_processes[gpu_id].append((p, video_name))
        
        # 持续监控所有进程，一个完成就立即启动下一个
        remaining_videos = sum(len(q) for q in gpu_queues)
        completed_videos = 0
        
        while remaining_videos > 0 or any(gpu_processes):
            for gpu_id in range(num_gpus):
                # 检查这个GPU上是否有完成的进程
                i = 0
                while i < len(gpu_processes[gpu_id]):
                    process, video_name = gpu_processes[gpu_id][i]
                    if not process.is_alive():
                        # 进程完成
                        process.join()
                        completed_videos += 1
                        pbar.update(1)
                        
                        # 收集结果
                        video_basename = os.path.splitext(os.path.basename(video_name))[0]
                        result_path = os.path.join(output_base_dir, f"{video_basename}_result.npz")
                        if os.path.exists(result_path):
                            result = np.load(result_path)
                            all_results.append((video_basename, dict(result)))
                        
                        # 移除这个进程
                        gpu_processes[gpu_id].pop(i)
                        
                        # 如果有等待的视频，启动新进程
                        if gpu_queues[gpu_id]:
                            next_video = gpu_queues[gpu_id].pop(0)
                            remaining_videos -= 1
                            args = (next_video, base_path, output_base_dir, version, seed, gpu_id, durations, timeout)
                            new_p = mp.Process(target=process_single_video, args=args, daemon=False)
                            new_p.start()
                            gpu_processes[gpu_id].append((new_p, next_video))
                    else:
                        i += 1
            
            # 短暂睡眠，避免CPU占用过高
            time.sleep(0.5)
        
        # 等待所有剩余进程完成
        for gpu_id in range(num_gpus):
            for process, video_name in gpu_processes[gpu_id]:
                process.join()
                pbar.update(1)
                
                # 收集结果
                video_basename = os.path.splitext(os.path.basename(video_name))[0]
                result_path = os.path.join(output_base_dir, f"{video_basename}_result.npz")
                if os.path.exists(result_path):
                    result = np.load(result_path)
                    all_results.append((video_basename, dict(result)))
    
    # Combine results
    all_camera_params = {}
    for video_basename, camera_params in all_results:
        if camera_params:  # Only add if we have valid results
            all_camera_params[video_basename] = camera_params
    
    # Save results
    np.savez(os.path.join(output_base_dir, 'camera_stats.npz'), **all_camera_params)
    
    # 合并所有日志文件
    combined_log_path = os.path.join(output_base_dir, 'combined_reconstruction_errors.txt')
    with open(combined_log_path, 'w', encoding='utf-8') as combined_log:
        # 遍历所有视频文件夹
        for video_dir in os.listdir(output_base_dir):
            video_path = os.path.join(output_base_dir, video_dir)
            if os.path.isdir(video_path):
                log_path = os.path.join(video_path, "reconstruction_errors.txt")
                if os.path.exists(log_path):
                    # 写入视频名称作为分隔符
                    combined_log.write(f"\n{'='*50}\n")
                    combined_log.write(f"Video: {video_dir}\n")
                    combined_log.write(f"{'='*50}\n\n")
                    
                    # 读取并写入原始日志内容
                    with open(log_path, 'r', encoding='utf-8') as f:
                        combined_log.write(f.read())
    
    durations = [str(k) for k in durations]
    durations.append('total')
    # Analyze the 5th parameter's magnitude distribution
    recall_stats, detailed_stats = analyze_camera_stats_magnitude(all_camera_params, durations, param_index=4)
    
    # Print results
    for duration, recalls in recall_stats.items():
        print(f"Duration {duration}:")
        for mag, recall in recalls.items():
            print(f"  < {mag}: {recall:.2%}")
    
    # Plot visualization charts
    plot_magnitude_recall(output_base_dir, recall_stats, detailed_stats)

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
    parser.add_argument("--prefix", type=str, default=None, help="Prefix for output directory")
    
    args = parser.parse_args()

    process_videos(
        args.base_path,
        args.video_list,
        args.output_dir,
        args.version,
        args.seed,
        args.durations,
        args.timeout,
        args.prefix
    )

if __name__ == "__main__":
    main()
