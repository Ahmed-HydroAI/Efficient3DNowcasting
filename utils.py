
"""
Utilities for Efficient3DNowcasting.

Notes
-----
- Scaling pipeline is preserved exactly as provided:
  X -> (X / 32.0) -> log10(X + 1), inverse: 10**x - 1 (no *32 on inverse).
- Functions work with either numpy arrays or torch tensors.
- Heavy plotting deps (cartopy) are imported lazily inside functions.
"""

from __future__ import annotations

import os
from typing import Optional, Sequence, Dict, Any

import numpy as np
import torch
import torch.nn as nn
path= r"/home/sv20953/PHD/PHD_part6/github"
os.chdir(path)
from Persistence_model import Persistence

# ------------------------- Scaling ------------------------- #

def Scaler(array: np.ndarray) -> np.ndarray:
    """
    Apply log10(x + 1) in numpy space. (Keeps the original behavior.)
    """
    return np.log10(array + 1)


def invScaler(array: np.ndarray) -> np.ndarray:
    """
    Inverse of Scaler: 10**x - 1. (No extra *32 here by design.)
    """
    return (10 ** array) - 1


# --------------------- Pre/Post-processing --------------------- #

def data_preprocessing(X: torch.Tensor | np.ndarray, device: torch.device) -> torch.Tensor:
    """
    Convert input to float32 tensor on device, divide by 32.0, then log-scale.

    Parameters
    ----------
    X : torch.Tensor or np.ndarray
        Expected shapes:
          - 3D: [T, H, W]
          - 4D: [C/T, H, W] or [B, T, H, W]
          - 5D: [B, C, T, H, W]
    device : torch.device

    Returns
    -------
    torch.Tensor (float32, on device)
    """
    if isinstance(X, torch.Tensor):
        X = X.to(dtype=torch.float32, device=device)
    else:
        X = torch.tensor(X, dtype=torch.float32, device=device)

    X = X / 32.0  # keep original scaling

    # Move to CPU for numpy ops, then back
    X_np = X.detach().cpu().numpy()
    X_scaled = Scaler(X_np)
    X = torch.tensor(X_scaled, dtype=torch.float32, device=device)
    return X


def data_postprocessing(nwcst: torch.Tensor | np.ndarray) -> np.ndarray:
    """
    Move to CPU numpy, inverse log scaling, and non-negative clipping.
    """
    if isinstance(nwcst, torch.Tensor):
        nwcst = nwcst.detach().cpu().numpy()
    nwcst = np.squeeze(np.array(nwcst))
    nwcst = invScaler(nwcst)
    nwcst = np.where(nwcst > 0, nwcst, 0)
    return nwcst


# --------------------- Inference helpers --------------------- #

def prediction_seq2seq_3D(
    model_instance: nn.Module,
    input_data: torch.Tensor | np.ndarray,
    device: Optional[str | torch.device] = None,
) -> np.ndarray:
    """
    One-shot multi-frame prediction for 3D models (e.g., 4→12).

    Returns
    -------
    np.ndarray
        Typically [Tout, H, W] after squeeze for B=1,C=1.
    """
    if device is None or not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        device = torch.device(device)

    model_instance = model_instance.to(device)
    model_instance.eval()

    if isinstance(input_data, np.ndarray):
        input_data = torch.tensor(input_data, dtype=torch.float32)

    # Normalize shapes to [B, C, T, H, W]
    if input_data.ndim == 3:
        input_data = input_data.unsqueeze(0).unsqueeze(0)  # [1,1,T,H,W]
    elif input_data.ndim == 4:
        input_data = input_data.unsqueeze(0)               # [1,C,T,H,W]
    # else assume already [B,C,T,H,W]

    input_data = input_data.to(device)
    input_data = data_preprocessing(input_data, device)

    with torch.no_grad():
        output = model_instance(input_data)

    output = output.cpu().detach()
    output = data_postprocessing(output)
    output = np.squeeze(output)
    return output


def prediction_recursive_3D(
    model_instance: nn.Module,
    input_data: torch.Tensor | np.ndarray,
    lead_time: int,
    device: torch.device,
) -> np.ndarray:
    """
    Autoregressive 3D prediction with 1-step-ahead models.

    Assumes model returns [B, C, 1, H, W] each step.
    """
    model_instance.to(device)
    model_instance.eval()

    if isinstance(input_data, np.ndarray):
        input_data = torch.tensor(input_data, dtype=torch.float32, device=device)
    else:
        input_data = input_data.to(device)

    # Normalize to [B, C, T, H, W]
    if input_data.ndim == 4:
        input_data = input_data.unsqueeze(0)  # [1, C, T, H, W]
    elif input_data.ndim == 3:
        input_data = input_data.unsqueeze(0).unsqueeze(0)  # [1, 1, T, H, W]

    input_data = data_preprocessing(input_data, device)

    preds = []
    with torch.no_grad():
        for _ in range(lead_time):
            pred = model_instance(input_data)  # [B, C, 1, H, W]
            preds.append(pred)
            # Slide window over time dimension (dim=2)
            input_data = torch.cat((input_data[:, :, 1:, :, :], pred), dim=2)

    nwcst = torch.cat(preds, dim=2)  # [B, C, lead_time, H, W]
    nwcst = data_postprocessing(nwcst)
    nwcst = np.squeeze(nwcst)
    return nwcst




def prediction_recursive_2D(model_instance, input_data, device, lead_time):
    # print(f"[DEBUG] Moving model to device: {device}")
    model_instance.to(device)
    model_instance.eval()

    if isinstance(input_data, np.ndarray):
        input_data = torch.tensor(input_data, dtype=torch.float32, device=device)
    else:
        input_data = input_data.to(device)

    input_data = input_data.unsqueeze(0)  # Add batch dimension
    input_data = data_preprocessing(input_data, device)

    nwcst = []

    with torch.no_grad():
        for _ in range(lead_time):
            pred = model_instance(input_data)
            pred = pred.squeeze(1)
            nwcst.append(pred)  # Keep tensors on GPU
            input_data = torch.cat((input_data[:, 1:, :, :], pred.unsqueeze(1)), dim=1)

    nwcst = torch.cat(nwcst, dim=0)  # Concatenate predictions
    nwcst = data_postprocessing(nwcst)  # Postprocess predictions
    nwcst = nwcst.squeeze()
    return nwcst

def prediction_persistence(inputs: np.ndarray) -> np.ndarray:
    """
    Simple persistence baseline using the last frame (keeps your original scaling behavior).
    """
    inputs = inputs / 32.0
    last = inputs[-1:]  # keep shape
    persistence = Persistence()
    persistence.input_data = last
    prediction = persistence.run()
    return prediction


# --------------------- Visualization --------------------- #

def plot_animations(
    obs: np.ndarray,
    pre: np.ndarray,
    path: str,
    cmap: str = "rainbow"
) -> str:
    """
    Save two GIFs (obs and pred) in British National Grid projection.

    Parameters
    ----------
    obs : array, shape [T, H, W]
    pre : array, shape [T, H, W]
    path : output directory

    Returns
    -------
    str : the same `path`
    """
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation

    # Heavy imports inside function to keep module import light
    try:
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature
    except Exception as e:
        raise ImportError(
            "plot_animations requires 'cartopy' to be installed."
        ) from e

    os.makedirs(path, exist_ok=True)

    crs_uk = ccrs.OSGB()  # British National Grid
    x_min, x_max = 150000, 662000
    y_min, y_max = 0, 512000

    fig1, ax1 = plt.subplots(subplot_kw={'projection': crs_uk})
    fig2, ax2 = plt.subplots(subplot_kw={'projection': crs_uk})

    obs_ims = []
    pre_ims = []

    for i in range(obs.shape[0]):
        time = 't + ' + str(i + 1)
        line1 = ax1.annotate(time, xy=(0.05, 0.95), xycoords='axes fraction',
                             fontsize=12, color='black', ha='left', va='top')
        line2 = ax2.annotate(time, xy=(0.05, 0.95), xycoords='axes fraction',
                             fontsize=12, color='black', ha='left', va='top')

        obs_frame = np.copy(obs[i])
        pre_frame = np.copy(pre[i])

        # Transparent below light rain threshold (unchanged)
        obs_frame[obs_frame < 0.5] = np.nan
        pre_frame[pre_frame < 0.5] = np.nan

        obs_im = ax1.imshow(
            obs_frame, vmin=0, vmax=10, animated=True, cmap=cmap,
            transform=crs_uk, extent=[x_min, x_max, y_min, y_max]
        )
        pre_im = ax2.imshow(
            pre_frame, vmin=0, vmax=10, animated=True, cmap=cmap,
            transform=crs_uk, extent=[x_min, x_max, y_min, y_max]
        )

        obs_ims.append([obs_im, line1])
        pre_ims.append([pre_im, line2])

    for ax in [ax1, ax2]:
        ax.coastlines(resolution='10m')
        ax.add_feature(cfeature.BORDERS, linestyle=':')
        ax.set_extent([x_min, x_max, y_min, y_max], crs=crs_uk)
        gl = ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)
        gl.right_labels = False
        gl.top_labels = False
        gl.xlines = False
        gl.ylines = False

    obs_ani = animation.ArtistAnimation(fig1, obs_ims, interval=150, blit=True, repeat_delay=3000)
    pre_ani = animation.ArtistAnimation(fig2, pre_ims, interval=150, blit=True, repeat_delay=3000)

    cbar1 = fig1.colorbar(obs_im, ax=ax1, orientation='vertical', fraction=0.046, pad=0.4)
    cbar1.set_label('Intensity mm/h', size=8, weight='bold')
    cbar1.ax.tick_params(labelsize=12)

    cbar2 = fig2.colorbar(pre_im, ax=ax2, orientation='vertical', fraction=0.046, pad=0.4)
    cbar2.set_label('Rainfall Intensity (mm/h)', size=12)
    cbar2.ax.tick_params(labelsize=12)

    obs_ani.save(os.path.join(path, 'obs.gif'), writer='Pillow')
    pre_ani.save(os.path.join(path, 'pre.gif'), writer='Pillow')

    plt.close(fig1)
    plt.close(fig2)
    return path


def plot_selected_lead_times_comparison(
    all_predictions_np: Dict[str, np.ndarray],
    all_labels_np: Dict[str, np.ndarray],
    selected_models: Sequence[str],
    renamed_models: Dict[str, str],
    sequence_index: int = 0,
    lead_times: Sequence[int] = (0, 5, 11),
    save_path: Optional[str] = None,
) -> None:
    """
    Plot predictions at selected lead times (3 frames per model).
    Two models per row → 6 subplots per row (T+5, T+30, T+60) × 2.
    The Obs row spans only 3 columns.
    """
    import math
    import numpy as np
    import matplotlib.pyplot as plt

    n_models = len(selected_models)
    models_per_row = 2
    n_model_rows = math.ceil(n_models / models_per_row)
    n_leads = len(lead_times)
    n_cols = models_per_row * n_leads
    n_rows = n_model_rows + 1  # +1 for Obs row

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 3, n_rows * 3))

    # Ensure axes is 2D array
    if n_rows == 1:
        axes = np.expand_dims(axes, 0)
    if n_cols == 1:
        axes = np.expand_dims(axes, 1)

    # Colormap (transparent below vmin)
    cmap = plt.cm.jet.copy()
    cmap.set_under("white", alpha=0)

    # Dynamic vmax from predictions + labels
    vmax = max([
        np.nanmax(all_predictions_np[model][sequence_index]) for model in selected_models
    ] + [np.nanmax(all_labels_np[selected_models[0]][sequence_index])])

    # 1) Observations (first row, first len(lead_times) columns)
    for j, t in enumerate(lead_times):
        ax = axes[0, j]
        obs_frame = all_labels_np[selected_models[0]][sequence_index][t]
        im = ax.imshow(obs_frame, cmap=cmap, vmin=0.1, vmax=10)
        ax.set_title(f"T+{(t+1)*5} min", fontsize=22)
        ax.set_xticks([])
        ax.set_yticks([])
    axes[0, 0].set_ylabel("Obs", fontsize=22)

    # Clear unused obs row axes, if any
    for j in range(len(lead_times), n_cols):
        axes[0, j].axis("off")

    # 2) Predictions: 2 models per row
    for idx, model in enumerate(selected_models):
        row = (idx // models_per_row) + 1
        col_offset = (idx % models_per_row) * n_leads
        for j, t in enumerate(lead_times):
            col = col_offset + j
            ax = axes[row, col]
            pred = all_predictions_np[model][sequence_index][t]
            ax.imshow(pred, cmap=cmap, vmin=0.1, vmax=10)
            ax.set_xticks([])
            ax.set_yticks([])
        axes[row, col_offset].set_ylabel(renamed_models.get(model, model), fontsize=22)

    # 3) Colorbar
    cbar_ax = fig.add_axes([0.92, 0.2, 0.015, 0.6])
    cbar = fig.colorbar(im, cax=cbar_ax, extend='min')
    cbar.set_label("Rainfall Intensity (mm/h)", fontsize=22)
    cbar.ax.tick_params(labelsize=22)

    plt.tight_layout(rect=[0, 0, 0.9, 1])
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"✅ Saved plot to {save_path}")
    plt.show()
