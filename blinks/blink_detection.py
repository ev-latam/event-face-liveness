import os
from tqdm import tqdm
import numpy as npc
import sys
from os.path import dirname, join
sys.path.append("../..")
sys.path.append("../")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from event_reps import activity_profile
from scipy.ndimage import label
from scipy.ndimage import gaussian_filter1d
from scipy.signal import peak_widths
from scipy.signal import find_peaks
from utils import *
from iou_matching_evaluation import evaluate_iou_detection



def normalize_time_series_dt(activity, timestamps,t_min, t_max, dt=2e-3):

    activity = np.asarray(activity, dtype=np.float32)
    timestamps = np.asarray(timestamps, dtype=np.float64)
    if len(activity) < 2:
        return None

    t_new = np.arange(t_min, t_max, dt)
    activity_interp = np.interp(t_new, timestamps, activity[:-1])
    return activity_interp, t_new


def detect_blinks(balance_s, y, dt_ms, window_size, prominence_high, prominence_low):
    """
    Detect blinks from filtered activity signal using peak detection.
    
    Args:
        balance_s: normalized activity signal (A_on - A_off) after smoothing
        y: reference array (used for shape)
        dt_ms: timestep in milliseconds
        window_size: window size based on percentile duration
        prominence_high: high prominence threshold for upstroke
        prominence_low: low prominence threshold for downstroke
    
    Returns:
        blink_mask: boolean array of detected blinks
    """
    roots = np.where(np.diff(np.sign(balance_s)) != 0)[0]
    window = window_size // 2
    blink_mask = np.zeros_like(y, dtype=bool)
    
    for i, r in enumerate(roots):
        if r-window < 0 or r+window >= len(balance_s):
            continue
        roi = balance_s[r-window:r]
        peaks, props = find_peaks(
            roi,
            prominence=prominence_high,
            distance=int(60 / dt_ms),
        )

        if (len(peaks) > 0):
            distance_to_r = np.abs(peaks - r)
            closest_peak = peaks[np.argmin(distance_to_r)]
            peak_width = peak_widths(roi, [closest_peak], rel_height=1)[0][0]
            left = int(r - np.round(peak_width))
            window = window_size // 2
            roi = -balance_s[r:r+window]
            peaks, props = find_peaks(
                roi,
                prominence=prominence_low,
                distance=int(60 / dt_ms),
            )
            if (len(peaks) > 0):
                distance_to_r = np.abs(peaks - r)
                closest_peak = peaks[np.argmin(distance_to_r)]
                peak_width = peak_widths(roi, [closest_peak], rel_height=1)[0][0]
                right = int(r + np.round(peak_width))
                blink_mask[left:right] = True
            else: 
                continue
    
    return blink_mask


def main():
    
    user_train = [1,2,10,11,12,13,14,15,16,17,18,19,20]
    user_val = [21,22,23,24]
    all_users = user_train + user_val
    experiments = [1]
    base_path = '/data2/nico/RGBE_Gaze/'
    df_blinks = pd.read_csv('../annotations/blinks.csv')

    #------------------
    # PARAMETERS
    #------------------
    dt = 2e-3 # 2ms time step
    dt_ms = dt * 1e3
    tau = 10e3  # for activity profile
    #------------------
    PERCENTILE_95_MS = 440
    PERCENTILE_95_LEN = int(PERCENTILE_95_MS / dt_ms)
    PROMINENCE_HIGH = 0.35
    PROMINENCE_LOW = 0.05
    #----------------------

    all_tp =0
    all_fp =0
    all_fn =0 


    for user_name in (all_users):
        for ex_time in experiments:       
            root_event_folder = f'{base_path}raw_data/user_{user_name}/exp{ex_time}/prophesee'
            events = np.load(root_event_folder + "/left_eye.npy", allow_pickle=True)

            at_on, at_off, t_on, t_off = activity_profile(events, tau=tau)
            t_min = np.min([np.min(t_on), np.min(t_off)])
            t_max = np.max([np.max(t_on), np.max(t_off)])
            at_on, t = normalize_time_series_dt(at_on, t_on, t_min, t_max, dt=dt_ms*1e3)
            at_off, _ = normalize_time_series_dt(at_off, t_off, t_min, t_max, dt=dt_ms*1e3)

            # Get actual time duration in seconds and normalize activity
            t_min = t_min/ 1e6    
            t_max = t_max / 1e6
            max_at = np.max([np.max(at_on), np.max(at_off)])
            at_on = at_on / (max_at + 1e-6) 
            at_off = at_off / (max_at + 1e-6)
            x = np.arange(t_min, t_max, dt)

            # Compute polarity balance with smoothing
            polarity_balance = (at_on - at_off)  
            balance_s = gaussian_filter1d(polarity_balance, sigma=3)
            balance_s = balance_s / (np.max(np.abs(balance_s)) + 1e-6)
        
            print("User ", user_name)
            
            image_align_event_time = f'{base_path}processed_data/frame_align_event_timestamp/user_{user_name}_exp_{ex_time}.txt'
            with open(image_align_event_time) as f:
                timestamp = [float(i)/1e6 for i in f.readlines()]

            y = load_dense_gt(user_name, x, timestamp,'../annotations/blinks.csv')
            
            blink_mask = detect_blinks(balance_s, y, dt_ms, 
                                    PERCENTILE_95_LEN, PROMINENCE_HIGH, PROMINENCE_LOW)

            plot_features_with_gt_and_pred(
            x,
            balance_s,
            y,
            blink_mask.astype(int),
            user=user_name,
            dt=dt,
            t_start_sec=1,
            t_end_sec=15
            )

            results = evaluate_iou_detection(blink_mask, y, dt_ms,iou_thresh=0.5)
            #print(results['fn'])
            all_tp += results['tp']
            all_fp += results['fp']
            all_fn += results['fn']
            print("-------------------------------")

    # Run evaluation: global micro f1 score
    precision = all_tp / (all_tp + all_fp) if (all_tp + all_fp) > 0 else 0
    recall = all_tp / (all_tp + all_fn) if (all_tp + all_fn) > 0 else 0

    print("GLOBAL RESULTS")
    print("TP:", all_tp)
    print("FP:", all_fp)
    print("FN:", all_fn)
    print("Precision:", precision)
    print("Recall:", recall)
    print("Micro F1 score", 2*precision*recall/(precision+recall+1e-6))


if __name__ == "__main__":
    main()