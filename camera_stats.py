import os
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import datetime

def read_video_list(video_list_path):
    """Read the list of videos from a text file."""
    with open(video_list_path, 'r') as f:
        videos = [line.strip() for line in f.readlines()]
    return videos

def read_single_camera_stats(npz_file_path):
    """
    Read saved camera parameter statistics data
    
    Args:
        npz_file_path: Path to NPZ file
    
    Returns:
        A dictionary where keys are video names and values are nested dictionaries 
        containing camera parameters for different durations
    """
    try:
        # Load NPZ file
        data = np.load(npz_file_path, allow_pickle=True)
        
        # Build structured result dictionary
        camera_stats = {}
        for video_name in data.files:
            camera_stats[video_name] = dict(data[video_name].item())
            
            # Optional: Convert internal data to NumPy arrays for easier manipulation
            for duration in camera_stats[video_name]:
                if isinstance(camera_stats[video_name][duration], np.ndarray):
                    continue
                camera_stats[video_name][duration] = np.array(camera_stats[video_name][duration])
        
        print(f"Successfully read camera parameters for {len(camera_stats)} videos from {npz_file_path}")
        return camera_stats
    
    except Exception as e:
        print(f"Error reading {npz_file_path}: {str(e)}")
        return None
    
def read_total_camera_stats(npz_file_path):
    """
    读取 video_cut.py 生成的相机参数统计数据的 NPZ 文件
    
    参数:
        npz_file_path: NPZ 文件路径
    
    返回:
        字典，其中:
        - 键是视频名称
        - 值是另一个字典，其中:
            - 键是时长（如 '30', '60', '120', 'total'）
            - 值是对应的相机参数数组（包含均值和标准差）
    """
    try:
        # 加载 NPZ 文件
        data = np.load(npz_file_path, allow_pickle=True)
        
        # 构建结果字典
        camera_stats = {}
        for video_name in data.files:
            # 获取该视频的所有时长数据
            durations_data = {}
            video_data = data[video_name].item()  # 将数组转换为字典
            for duration, values in video_data.items():
                durations_data[duration] = values
            camera_stats[video_name] = durations_data
        
        print(f"成功从 {npz_file_path} 读取了 {len(camera_stats)} 个视频的相机参数")
        return camera_stats
    
    except Exception as e:
        print(f"读取 {npz_file_path} 时出错: {str(e)}")
        return None

def analyze_camera_stats_magnitude(camera_stats, durations,param_index=4):
    """
    Analyze the magnitude distribution of a specific parameter in camera statistics
    
    Args:
        camera_stats: Camera parameter dictionary read from read_camera_stats function
        param_index: Index of the parameter to analyze, default is the 5th parameter
    
    Returns:
        A tuple containing:
        1. A dictionary with recall rates for different magnitudes at each duration
        2. A dictionary with detailed video information for different magnitudes at each duration
    """
    # Define magnitudes to check
    magnitudes = [10, 100, 1000, np.inf]
    
    # Initialize result dictionaries
    magnitude_stats = defaultdict(lambda: {mag: 0 for mag in magnitudes})
    total_counts = defaultdict(int)
    
    # Initialize detailed information dictionary to record video names and parameter values for each magnitude
    detailed_stats = defaultdict(lambda: {mag: [] for mag in magnitudes})

    for duration in durations:
        magnitude_stats[duration]['failed'] = 0 
        detailed_stats[duration]['failed'] = []
    for video_name, camera in camera_stats.items():
        for duration in durations:
            if duration not in camera.keys():
                magnitude_stats[duration]['failed'] += 1
                # Record detailed information
                detailed_stats[duration]['failed'].append((video_name, 'failed'))
            else:
                for mag in magnitudes:
                    param_value = abs(camera[duration][param_index])
                    if param_value < mag:
                        magnitude_stats[duration][mag] += 1
                        # Record detailed information
                        detailed_stats[duration][mag].append((video_name, param_value))
                        break
            total_counts[duration] += 1

    # Calculate recall rates for different magnitudes at each duration
    recall_stats = {}
    for duration, mag_counts in magnitude_stats.items():
        if total_counts[duration] > 0:
            recall_stats[duration] = {mag: count / total_counts[duration] 
                                     for mag, count in mag_counts.items()}
    
    return recall_stats, detailed_stats

def plot_magnitude_recall(output_dir, recall_stats, detailed_stats=None):
    """
    Plot bar charts of parameter magnitude recall rates for different durations and generate detailed report
    
    Args:
        output_dir: Output directory
        recall_stats: Recall statistics returned from analyze_camera_stats_magnitude function
        detailed_stats: Detailed information statistics returned from analyze_camera_stats_magnitude function
    """
    durations = recall_stats.keys()
    magnitudes = [10, 100, 1000, np.inf, 'failed']
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    bar_width = 0.2
    index = np.arange(len(durations))
    
    for i, mag in enumerate(magnitudes):
        recalls = [recall_stats[duration][mag] for duration in durations]
        if mag != 'failed':
            ax.bar(index + i * bar_width, recalls, bar_width, label=f'< {mag}')
        else:
            ax.bar(index + i * bar_width, recalls, bar_width, label=f'failed')
    
    ax.set_xlabel('duration')
    ax.set_ylabel('recall')
    ax.set_title('Recall of Different Duration')
    ax.set_xticks(index + bar_width * (len(magnitudes) - 1) / 2)
    ax.set_xticklabels(durations)
    ax.legend()
    
    plt.tight_layout()
    fig_path = os.path.join(output_dir, 'camera_stats_magnitude_recall.png')
    plt.savefig(fig_path)
    print(f"Figure saved to {fig_path}")
    
    # Generate detailed report
    if detailed_stats:
        report_path = os.path.join(output_dir, 'camera_stats_detailed_report.txt')
        with open(report_path, 'w') as f:
            f.write("Camera Parameter Magnitude Distribution Detailed Report\n")
            f.write("=" * 50 + "\n\n")
            
            # Write overall statistics
            f.write("Overall Statistics:\n")
            f.write("-" * 30 + "\n")
            for duration in durations:
                f.write(f"Duration {duration}:\n")
                for mag in magnitudes:
                    recall = recall_stats[duration][mag]
                    count = len(detailed_stats[duration][mag])
                    if mag != 'failed':
                        f.write(f"  < {mag}: {recall:.2%} ({count} videos)\n")
                    else:
                        f.write(f" {mag}: {recall:.2%} ({count} videos)\n")
                f.write("\n")
            
            # Write detailed information
            f.write("\nDetailed Video Information:\n")
            f.write("-" * 30 + "\n")
            for duration in durations:
                f.write(f"Duration {duration}:\n")
                for mag in magnitudes:
                    if mag != 'failed':
                        f.write(f"  Videos < {mag}:\n")
                    else:
                        f.write(f"  Videos failed:\n")
                    # Sort by parameter value
                    sorted_videos = sorted(detailed_stats[duration][mag], key=lambda x: x[1])
                    for video_name, param_value in sorted_videos:
                        if mag != 'failed':
                            f.write(f"    {video_name}: {param_value:.6f}\n")
                        else:
                            f.write(f"    {video_name}: failed\n")
                    f.write("\n")
                f.write("\n")
            
            f.write("=" * 50 + "\n")
            f.write(f"Report generated at: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        print(f"Detailed report saved to {report_path}")

# Usage example
if __name__ == "__main__":
    npz_file_path = "/home/lzj/lzj/matching_codes/gim/reconstruction_out/camera_stats.npz"
    scene_path = "/home/lzj/lzj/matching_codes/gim/data/100h.txt"
    output_path = "/home/lzj/lzj/matching_codes/gim/data/camera_stats.npz"
    camera_stats = read_total_camera_stats(npz_file_path)
    video_list = sorted(read_video_list(scene_path))
    durations = ['80', '90', '100']

    not_in_camera_stats = []
    failed_camera_stats = []
    single_camera_stats = []
    warning_camera_stats = []
    valid_scene={}


    for video_name in video_list:
        if video_name not in camera_stats.keys():
            not_in_camera_stats.append(video_name)
        else:
            if len(camera_stats[video_name].keys()) == 2:
                single_camera_stats.append(video_name)
            elif camera_stats[video_name]['total'][4] < 100:
                valid_scene[video_name] = camera_stats[video_name]['total'][:3]
            elif len(camera_stats[video_name].keys()) == 4:
                tuples=[[0, 1], [0, 2], [1, 2]]
                signal = False
                for tuple in tuples:
                    duration0 = durations[tuple[0]]
                    duration1 = durations[tuple[1]]
                    dist = abs(camera_stats[video_name][duration0][0] - camera_stats[video_name][duration1][0])
                    if dist < 50:
                        warning_camera_stats.append(video_name)
                        valid_scene[video_name] = camera_stats[video_name]['total'][:3]
                        valid_scene[video_name][0] = (camera_stats[video_name][duration0][0] * int(duration0) + camera_stats[video_name][duration1][0] * int(duration1)) / (int(duration0) + int(duration1))
                        signal = True
                if not signal:
                    failed_camera_stats.append(video_name)
            else:
                failed_camera_stats.append(video_name)
    np.savez(output_path, **valid_scene)

    with open("/home/lzj/lzj/matching_codes/gim/data/valid_scene.txt", 'w') as f:
        for valid_scene in valid_scene.keys():
            f.write(f"{valid_scene}\n")
    
    with open("/home/lzj/lzj/matching_codes/gim/data/abnormal_camera_stats.txt", 'w') as f:
        f.write("not in camera stats:\n")
        for scene in not_in_camera_stats:
            f.write(f"{scene}\n")
        f.write("\n")
        f.write("failed camera stats:\n")
        for scene in failed_camera_stats:
            f.write(f"{scene}\n")
        f.write("\n")
        f.write("single camera stats:\n")
        for scene in single_camera_stats:
            f.write(f"{scene}\n")
        f.write("\n")
        f.write("warning camera stats:\n")
        for scene in warning_camera_stats:
            f.write(f"{scene}\n")
        




    # Analyze the 5th parameter's magnitude distribution
    # recall_stats, detailed_stats = analyze_camera_stats_magnitude(camera_stats, ['60', '90', '120', 'total'], param_index=4)
    
    # Print results
    # for duration, recalls in recall_stats.items():
    #     print(f"Duration {duration}:")
    #     for mag, recall in recalls.items():
    #         print(f"  < {mag}: {recall:.2%}")
