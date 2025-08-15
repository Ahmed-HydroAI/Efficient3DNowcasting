#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Validation metrics for rainfall nowcasting models.

Includes pixel-wise RMSE, MAE, CSI, FAR, POD, and thresholded MAE across lead times.

Author: Ahmed Abdelhalim
Date: July 2025
"""

import numpy as np

# --------------------------- #
#      Basic Error Metrics     #
# --------------------------- #

def calculate_RMSE(obs, sim):
    """
    Compute Root Mean Square Error (RMSE) per lead time.

    Args:
        obs (np.ndarray): Ground truth rainfall [T, H, W]
        sim (np.ndarray): Predicted rainfall [T, H, W]

    Returns:
        list: RMSE per timestep
    """
    return [np.sqrt(np.mean((obs[i] - sim[i])**2)) for i in range(obs.shape[0])]


def calculate_MAE(obs, sim):
    """
    Compute Mean Absolute Error (MAE) per lead time.

    Returns:
        list: MAE per timestep
    """
    return [np.mean(np.abs(obs[i] - sim[i])) for i in range(obs.shape[0])]

# --------------------------- #
#    Categorical Metrics       #
# --------------------------- #

def CSI(obs, sim, threshold):
    hits = np.sum((obs >= threshold) & (sim >= threshold))
    misses = np.sum((obs >= threshold) & (sim < threshold))
    false_alarms = np.sum((obs < threshold) & (sim >= threshold))
    denom = hits + misses + false_alarms
    return hits / denom if denom != 0 else 0


def FAR(obs, sim, threshold):
    hits = np.sum((obs >= threshold) & (sim >= threshold))
    false_alarms = np.sum((obs < threshold) & (sim >= threshold))
    denom = hits + false_alarms
    return false_alarms / denom if denom != 0 else 0


def POD(obs, sim, threshold):
    hits = np.sum((obs >= threshold) & (sim >= threshold))
    misses = np.sum((obs >= threshold) & (sim < threshold))
    denom = hits + misses
    return hits / denom if denom != 0 else 0

# --------------------------- #
#     Multi-threshold Versions #
# --------------------------- #

def calculate_CSI(obs, sim, thresholds=[0.1, 1.0, 3.0, 5.0, 8.0, 10.0, 12.0, 15.0]):
    """
    Compute CSI for multiple thresholds across all lead times.
    Returns a dict of {threshold: list of CSI per timestep}
    """
    return {str(t): [CSI(obs[i], sim[i], t) for i in range(obs.shape[0])] for t in thresholds}


def calculate_FAR(obs, sim, thresholds=[0.1, 1.0, 3.0, 5.0, 8.0, 10.0, 12.0, 15.0]):
    return {str(t): [FAR(obs[i], sim[i], t) for i in range(obs.shape[0])] for t in thresholds}


def calculate_POD(obs, sim, thresholds=[0.1, 1.0, 3.0, 5.0, 8.0, 10.0, 12.0, 15.0]):
    return {str(t): [POD(obs[i], sim[i], t) for i in range(obs.shape[0])] for t in thresholds}

# --------------------------- #
#     Thresholded MAE         #
# --------------------------- #

def calculate_MAE_thresholded(obs, sim, thresholds=[0.1, 1.0, 3.0, 5.0, 8.0, 10.0, 12.0, 15.0]):
    """
    Compute MAE only on high-intensity pixels (obs >= threshold).

    Returns:
        dict: {threshold: list of MAE per timestep}
    """
    result = {}
    for t in thresholds:
        per_frame_mae = []
        for i in range(obs.shape[0]):
            obs_masked = np.where(obs[i] >= t, obs[i], np.nan)
            sim_masked = np.where(obs[i] >= t, sim[i], np.nan)
            per_frame_mae.append(np.nanmean(np.abs(obs_masked - sim_masked)))
        result[str(t)] = per_frame_mae
    return result
