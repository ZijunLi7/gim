import os
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import datetime

def read_camera_stats(npz_file_path):
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
    npz_file_path = "/home/lzj/lzj/matching_codes/gim/reconstruction_out/100h_dkm_unique_camera_id_rec_1/camera_stats.npz"
    camera_stats = read_camera_stats(npz_file_path)
    
    # Analyze the 5th parameter's magnitude distribution
    recall_stats, detailed_stats = analyze_camera_stats_magnitude(camera_stats, ['60', '90', '120', 'total'], param_index=4)
    
    # Print results
    for duration, recalls in recall_stats.items():
        print(f"Duration {duration}:")
        for mag, recall in recalls.items():
            print(f"  < {mag}: {recall:.2%}")
    
    # Plot visualization charts
    plot_magnitude_recall('reconstruction_out', recall_stats, detailed_stats)
