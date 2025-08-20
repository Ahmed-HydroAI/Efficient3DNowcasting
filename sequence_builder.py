#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sequence preparation module for radar rainfall nowcasting.

This script extracts temporally consistent rainfall sequences of arbitrary length.
Each sequence contains frames spaced by a constant time interval (e.g., every 5 minutes),
and sequences are filtered based on minimum rainfall activity.

Author: Ahmed Abdelhalim
Date: July 2025
"""

from datetime import datetime, timedelta
import os
import numpy as np
import imageio.v3 as iio

# ----------------------------- #
#       Helper Functions        #
# ----------------------------- #

def extract_timestamp(filename):
    """
    Extract timestamp from a filename assuming the format: '..._YYYYMMDDHHMM...'.

    Parameters:
        filename (str): Filename containing a 12-digit timestamp.

    Returns:
        datetime: Parsed datetime object.
    """
    timestamp_str = filename[-30:-18]
    return datetime.strptime(timestamp_str, "%Y%m%d%H%M")


def check_consecutive(current_filename, next_filename, time_interval):
    """
    Check if two filenames correspond to consecutive frames based on time interval.

    Parameters:
        current_filename (str): Current frame filename.
        next_filename (str): Next frame filename.
        time_interval (int): Time difference in minutes between frames.

    Returns:
        bool: True if the time difference matches, False otherwise.
    """
    current_time = extract_timestamp(current_filename)
    next_time = extract_timestamp(next_filename)
    return next_time - current_time == timedelta(minutes=time_interval)


# ----------------------------- #
#     Sequence Extraction       #
# ----------------------------- #

def extract_sequences(path, file_list, num_frames=24, window=1,
                      min_nonzero_ratio=0.5, time_interval=5):
    """
    Extract valid rainfall sequences with consistent timestamps and sufficient rainfall.

    Parameters:
        path (str): Directory path where the rainfall frames are stored.
        file_list (list): List of sorted frame filenames.
        num_frames (int): Number of frames required in each sequence.
        window (int): Number of positions to slide the window after each sequence.
        min_nonzero_ratio (float): Minimum ratio of non-zero (rainfall) pixels to keep sequence.
        time_interval (int): Required time interval (in minutes) between frames.

    Returns:
        list: A list of valid file sequences (each a list of full file paths).
    """
    valid_sequences = []
    i = 0

    while i <= len(file_list) - num_frames:
        sequence_valid = True
        sequence = []
        current_time = extract_timestamp(file_list[i])
        sequence.append(os.path.join(path, file_list[i]))

        # Try to collect num_frames spaced by `time_interval`
        for j in range(1, num_frames):
            next_time_expected = current_time + timedelta(minutes=time_interval)
            found = False

            # Search ahead for the next frame matching the expected timestamp
            for k in range(i + 1, len(file_list)):
                candidate_time = extract_timestamp(file_list[k])
                if candidate_time == next_time_expected:
                    sequence.append(os.path.join(path, file_list[k]))
                    current_time = candidate_time
                    found = True
                    break
                elif candidate_time > next_time_expected:
                    break  # No match possible, list is sorted

            if not found:
                sequence_valid = False
                break

        # Check rainfall activity (non-zero pixels) across all frames
        if sequence_valid:
            for frame_path in sequence:
                image = np.array(iio.imread(frame_path), dtype=np.float32)
                if np.sum(image > 0) / (512 * 512) < min_nonzero_ratio:
                    sequence_valid = False
                    break

        if sequence_valid:
            valid_sequences.append(sequence)

        i += window

    return valid_sequences
