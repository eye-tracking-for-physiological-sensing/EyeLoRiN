import h5py
import os
import pandas as pd
import re
import cv2 as cv
import glob
from io import BytesIO
import argparse
import struct
import glob
import sys
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import cv2
import scipy.signal as signal

# Function to convert events to frame
def events_to_frame(events, width, height):
    frame = np.ones((height, width), dtype=np.uint8) * 255
    for t, x, y, polarity in events:
        frame[int(y), int(x)] = 0
    return frame

sensor_size = (640, 480, 2)
dtype = np.dtype([("t", int), ("x", int), ("y", int), ("p", int)])

def event_file_to_array(file_name):
  with h5py.File(f"./event_files/{file_name}.h5", "r") as f: #event_files/
      # original events.dtype is dtype([('t', '<u8'), ('x', '<u8'), ('y', '<u8'), ('p', '<u8')])
      # t is in us
      events = f["events"][:].astype(dtype)
      events['p'] = events['p']*2 -1  # convert polarity to -1 and 1
      event_array = np.array(events.tolist())
      event_array[:, 3][event_array[:, 3] == -1] = 0
  return event_array

# Function to extract numbers from filename for sorting
def extract_numbers(filename):
    numbers = re.findall(r'\d+', filename)
    return tuple(map(int, numbers))

# Function to combine CSV files in order
def combine_csv_files(input_dir, output_file):
    files = [f for f in os.listdir(input_dir) if f.endswith('.csv')]
    sorted_files = sorted(files, key=extract_numbers)

    print(sorted_files)

    combined_df = pd.DataFrame()

    for file in sorted_files:
        filepath = os.path.join(input_dir, file)
        df = pd.read_csv(filepath)
        combined_df = pd.concat([combined_df, df], ignore_index=True)

    combined_df.to_csv(output_file, index=False)
    print(f"Combined file saved to {output_file}")

def local_frequency_variance(data, window_size=5, fs=30):
    """Computes local frequency variance using STFT"""
    f, t, Zxx = signal.stft(data, fs=fs, nperseg=window_size)
    power_spectrum = np.abs(Zxx) ** 2
    return np.var(power_spectrum, axis=0) 

def adaptive_smoothing(predictions_csv, output_csv, method = "covariance", base_window=5, min_window=5, max_window=20, percentile=75):
    """
    Perform adaptive median smoothing based on local motion variance.
    
    Args:
        refined_x (np.array): X-coordinate predictions.
        refined_y (np.array): Y-coordinate predictions.
        base_window (int): Base window size for motion variance calculation.
        min_window (int): Minimum allowed smoothing window.
        max_window (int): Maximum allowed smoothing window.
        percentile (int): Percentile to determine adaptive window size.
    
    Returns:
        (np.array, np.array): Smoothed x and y predictions.
    """

    predictions = pd.read_csv(predictions_csv)
    refined_x = (predictions['x_smooth'].values).copy()
    refined_y = (predictions['y_smooth'].values).copy()

    if method == "raw":
      #Compute motion variance for local regions
      motion_variance = np.sqrt(pd.Series(refined_x).diff().pow(2) + pd.Series(refined_y).diff().pow(2))
      #Rolling mean to smooth variance over `base_window`
      smoothed_variance = motion_variance.rolling(base_window, center=True, min_periods=1).mean()

    if method == 'velocity':
      #velocity-based 
      motion_variance = np.sqrt(np.diff(refined_x, prepend=refined_x[0])**2 + 
                            np.diff(refined_y, prepend=refined_y[0])**2)

      # Apply rolling mean for smoother variance estimation
      smoothed_variance = pd.Series(motion_variance).rolling(base_window, center=True, min_periods=1).mean()

    if method == 'acceleration':
      #acceleration-based
      acceleration_variance = np.sqrt(np.diff(refined_x, n=2, prepend=[refined_x[0]]*2)**2 +
                                  np.diff(refined_y, n=2, prepend=[refined_y[0]]*2)**2)

      smoothed_variance = pd.Series(acceleration_variance).rolling(base_window, center=True, min_periods=1).mean()

    if method == "covariance":
        # Compute local covariance in rolling windows
        rolling_cov_x = pd.Series(refined_x).rolling(base_window).apply(lambda x: np.cov(x, rowvar=False).mean(), raw=True)
        rolling_cov_y = pd.Series(refined_y).rolling(base_window).apply(lambda y: np.cov(y, rowvar=False).mean(), raw=True)

        # Combine motion variance from x and y directions
        smoothed_variance = np.sqrt(rolling_cov_x**2 + rolling_cov_y**2)
        print(type(smoothed_variance))

    elif method == "frequency":
        # Compute frequency variance
        freq_var_x = local_frequency_variance(refined_x, base_window)
        freq_var_y = local_frequency_variance(refined_y, base_window)

        # Combine motion variance from both directions
        motion_variance = np.sqrt(freq_var_x**2 + freq_var_y**2)

        # Match length with refined_x (since STFT reduces size)
        smoothed_variance = np.pad(motion_variance, (base_window//2, len(refined_x) - len(motion_variance) - base_window//2), mode='edge')

        # Convert to Pandas Series to use rolling window operations
        smoothed_variance = pd.Series(smoothed_variance)

        # Match length with refined_x by padding at the edges
        smoothed_variance = smoothed_variance.reindex(range(len(refined_x)), method="nearest")

    else:
        raise ValueError("Invalid method!")
    
    # Clip and scale adaptive window sizes based on motion variance
    median_window = np.clip(smoothed_variance.fillna(min_window).astype(int), min_window, max_window)

    # Compute a dynamic window for each point based on the local percentile of motion variance
    adaptive_windows = median_window.rolling(base_window, center=True, min_periods=1).apply(lambda x: np.percentile(x, percentile), raw=True).astype(int)
    adaptive_windows = np.clip(adaptive_windows, min_window, max_window)
    print(adaptive_windows.max(), adaptive_windows.min())

    # Apply adaptive median filtering with varying window sizes
    smoothed_x = np.array([pd.Series(refined_x).rolling(window=w, center=True, min_periods=1).median().values[i] 
                            for i, w in enumerate(adaptive_windows)])
    smoothed_y = np.array([pd.Series(refined_y).rolling(window=w, center=True, min_periods=1).median().values[i] 
                            for i, w in enumerate(adaptive_windows)])
    
    refined_predictions = pd.DataFrame({'row_id': predictions['row_id'], 'x': smoothed_x/8, 'y': smoothed_y/8})
    refined_predictions.to_csv(output_csv, index=False)

def post_process_pupil_coordinates_optical_flow(events, predictions_csv, output_csv):
    """
    Post-processes pupil coordinates using event data, median filtering, and simplified
    optical flow approximation.

    Args:
        events (np.array): Event data with shape [N, 4] (timestamp, x, y, polarity).
        predictions_csv (str): Path to CSV file with initial predictions.
        output_csv (str): Path to output CSV file with refined predictions.
    """

    scale = 8

    predictions = pd.read_csv(predictions_csv)
    refined_x = (predictions['x'].values*scale).copy()
    refined_y = (predictions['y'].values*scale).copy()

    total_duration = events[-1, 0] - events[0, 0]
    num_predictions = len(predictions)
    time_step = total_duration / num_predictions

    median_window = 20
    refined_x = pd.Series(refined_x).rolling(window=median_window, center=True, min_periods=1).median().values
    refined_y = pd.Series(refined_y).rolling(window=median_window, center=True, min_periods=1).median().values

    prev_timestamp = events[0, 0]

    for i, row in predictions.iterrows():
        timestamp = events[0, 0] + (i+1) * time_step
        x = refined_x[i]
        y = refined_y[i]

        roi_size = 10*scale
        if i > 5:
            diff_x = np.abs(refined_x[i] - np.mean(refined_x[i-5:i]))
            diff_y = np.abs(refined_y[i] - np.mean(refined_y[i-5:i]))
            if diff_x > 2*scale or diff_y > 2*scale:
                roi_size = 15*scale
            else:
                roi_size = 8*scale

        roi_events = events[
            (events[:, 1] >= x - roi_size) & (events[:, 1] <= x + roi_size) &
            (events[:, 2] >= y - roi_size) & (events[:, 2] <= y + roi_size) &
            (events[:, 0] >= prev_timestamp) & (events[:, 0] <= timestamp)
        ]

        prev_timestamp = timestamp

        if len(roi_events) > 10*scale:
            # Simplified Optical Flow Estimation (Approximation)
            dx = 0
            dy = 0
            if len(roi_events) > 1:
                for j in range(1, len(roi_events)):
                    dx += roi_events[j, 1] - roi_events[j - 1, 1]
                    dy += roi_events[j, 2] - roi_events[j - 1, 2]

                if abs(dx) > 0 or abs(dy) > 0:
                    # Normalize and shift by 1 pixel
                    magnitude = np.sqrt(dx**2 + dy**2)
                    if magnitude > 0:
                        dx_shift = int(dx / magnitude)
                        dy_shift = int(dy / magnitude)
                        refined_x[i] += dx_shift
                        refined_y[i] += dy_shift

    refined_predictions = pd.DataFrame({'row_id': predictions['row_id'], 'x': refined_x/8, 'y': refined_y/8})
    refined_predictions.to_csv(output_csv, index=False)
    
file_names = ['1_1', '2_2', '3_1', '4_2', '5_2', '6_4', '7_5', '8_2', '8_3', '10_2', '12_4']

for file_name in file_names:
  event_array = event_file_to_array(file_name)
  post_process_pupil_coordinates_optical_flow(event_array, f'./original_bigBrains/submission_check_{file_name}.csv', f'./refined/refined_predictions_{file_name}.csv')
# combine_csv_files("./OFE", "./combined_sample_submission.csv")