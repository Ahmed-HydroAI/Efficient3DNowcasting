
import time
import numpy as np
import os
import torch
import matplotlib.pyplot as plt
import imageio.v3 as iio
import collections
import gc
from tqdm import tqdm
from tabulate import tabulate
try:
    from ptflops import get_model_complexity_info
    HAVE_PTFLOPS = True
except Exception:
    HAVE_PTFLOPS = False


from metrics import *
from sequence_builder import extract_sequences
from depthwise_unet3d import DepthwiseUNet3D_4in_12out
from standard_unet2d import Standard_UNet2D
from standard_unet3d import StandardUNet3D_4in_12out, StandardUNet3D_12in_12out, StandardUNet3D_3in_1out
from group_unet3d import GroupUNet3D_4in_12out
from hybrid_unet3d import HybridUNet3D_4in_12out
from R2Plus1D_unet3d import R2Plus1DUNet3D_4in_12out
from ghost_unet3d import GhostUNet3D_4in_12out
from shift_unet3d import ShiftUNet3D_4in_12out
from utils import *
from Persistence_model import Persistence


#####Select models you want to run

model_configs = [
    

    {
        "name": "Persistence",
        "model": None,
        "weights": None,
        "in_frames": 4,
        "out_frames": 12,
        "predict_fn": lambda _, x, device=None: prediction_persistence(x),
    },
    
    {
        "name": "Standard_UNet2D",
        "model": Standard_UNet2D(),
        "weights": "/home/sv20953/unet3d_weights/input_configurations/Standard_UNet2D_3in_1out/best_model_weights.pth",
        "in_frames": 3,
        "out_frames": 12,
        "predict_fn": prediction_recursive_2D,
    },
    
    {
        "name": "StandardUNet3D_3in_1out",
        "model": StandardUNet3D_3in_1out(),
        "weights": "/home/sv20953/unet3d_weights/input_configurations/Standard_UNet3D_3in_1out/best_model_weights.pth",
        "in_frames": 3,
        "out_frames": 12,
        "predict_fn": prediction_recursive_3D,
    },
    
    {
        "name": "StandardUNet3D_4in_12out",
        "model": StandardUNet3D_4in_12out(),
        "weights": "/home/sv20953/unet3d_weights/input_configurations_1000sequences/Standard_UNet3D_4in_12out/best_model_weights.pth",
        "in_frames": 4,
        "out_frames": 12,
        "predict_fn": prediction_seq2seq_3D,
    },
    
    # {
    #     "name": "StandardUNet3D_12in_12out",
    #     "model": StandardUNet3D_12in_12out(),
    #     "weights": "/home/sv20953/unet3d_weights/input_configurations_1000sequences/Standard_UNet3D_12in_12out/best_model_weights.pth",
    #     "in_frames": 12,
    #     "out_frames": 12,
    #     "predict_fn": prediction_seq2seq_3D,
    # },
        
    
    {
        "name": "GhostUNet3D_4in_12out",
        "model": GhostUNet3D_4in_12out(),
        "weights": "/home/sv20953/unet3d_weights/input_configurations_1000sequences/Ghost_UNet3D_4in_12out/best_model_weights.pth",
        "in_frames": 4,
        "out_frames": 12,
        "predict_fn": prediction_seq2seq_3D,
    },
    {
        "name": "DepthwiseUNet3D_4in_12out",
        "model": DepthwiseUNet3D_4in_12out(),
        "weights": "/home/sv20953/unet3d_weights/input_configurations_1000sequences/Depthwise_UNet3D_4in_12out/best_model_weights.pth",
        "in_frames": 4,
        "out_frames": 12,
        "predict_fn": prediction_seq2seq_3D,
    },
    {
        "name": "GroupUNet3D_4in_12out",
        "model": GroupUNet3D_4in_12out(),
        "weights": "/home/sv20953/unet3d_weights/input_configurations_1000sequences/Group_UNet3D_4in_12out/best_model_weights.pth",
        "in_frames": 4,
        "out_frames": 12,
        "predict_fn": prediction_seq2seq_3D,
    },
    {
        "name": "HybridUNet3D_4in_12out",
        "model": HybridUNet3D_4in_12out(),
        "weights": "/home/sv20953/unet3d_weights/input_configurations_1000sequences/Hybrid_UNet3D_4in_12out/best_model_weights.pth",
        "in_frames": 4,
        "out_frames": 12,
        "predict_fn": prediction_seq2seq_3D,
    },
        
    {
        "name": "R2Plus1DUNet3D_4in_12out",
        "model": R2Plus1DUNet3D_4in_12out(),
        "weights": "/home/sv20953/unet3d_weights/input_configurations_1000sequences/R2plus1D_UNet3D_4in_12out/best_model_weights.pth",
        "in_frames": 4,
        "out_frames": 12,
        "predict_fn": prediction_seq2seq_3D,
    },
    {
        "name": "ShiftUNet3D_4in_12out",
        "model": ShiftUNet3D_4in_12out(),
        "weights": "/home/sv20953/unet3d_weights/input_configurations_1000sequences/Shift_UNet3D_4in_12out/best_model_weights.pth",
        "in_frames": 4,
        "out_frames": 12,
        "predict_fn": prediction_seq2seq_3D,
    },

    
]

thresholds = [0.1, 1.0, 3.0, 5.0, 10.0]
data_path = "/home/sv20953/PHD/2020"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Keep ONE of these two lines (not both). If you want an offset slice, keep the first:
lists = sorted(os.listdir(data_path))[5000:100000]
# lists = sorted(os.listdir(data_path))

max_total_frames = max(cfg["in_frames"] + cfg["out_frames"] for cfg in model_configs)

sequences_all = extract_sequences(
    data_path,
    lists,
    num_frames=max_total_frames,
    window=max_total_frames,
    min_nonzero_ratio=0.2
)
print("Number of sequences:", len(sequences_all))

sequences_all = sequences_all[:10]

# -----------------------------
# Utilities
# -----------------------------
def read_stack(files):
    """Read a list of file paths into a float32 array [T, H, W]."""
    return np.stack([iio.imread(f).astype(np.float32) for f in files], axis=0)

def call_predict(predict_fn, model, input_imgs, out_frames, device):
    """
    Unified entry for all models. Crucially:
      - If the predictor is 'prediction_recursive' (2D legacy), we pass lead_time.
      - If it's 'prediction_recursive_3D', we pass (lead_time, device) in that order.
      - Otherwise (seq2seq or baseline) we call (model, x, device).
    Finally, squeeze/trim/pad to [T=out_frames,H,W].
    """
    name = getattr(predict_fn, "__name__", "")

    if name == "prediction_recursive_2D":              # <-- your old 2D function
        pred = predict_fn(model, input_imgs, device, out_frames)

    elif name == "prediction_recursive_3D":
        pred = predict_fn(model, input_imgs, out_frames, device)

    
    else:
        # seq2seq_3D or persistence lambda(model ignored)
        pred = predict_fn(model, input_imgs, device)

    # normalize to numpy [T,H,W]
    if isinstance(pred, torch.Tensor):
        pred = pred.detach().cpu().numpy()
    pred = np.asarray(pred)

    # squeeze leading singleton dims
    while pred.ndim > 3 and pred.shape[0] == 1:
        pred = np.squeeze(pred, axis=0)
    if pred.ndim == 4 and pred.shape[0] == 1:
        pred = np.squeeze(pred, axis=0)

    if pred.ndim != 3:
        raise ValueError(f"Prediction has unexpected shape {pred.shape}, expected [T,H,W].")

    # enforce exact out_frames (pad with last frame or crop)
    T = pred.shape[0]
    if T < out_frames:
        pred = np.concatenate([pred, np.repeat(pred[-1:], out_frames - T, axis=0)], axis=0)
    elif T > out_frames:
        pred = pred[:out_frames]
    return pred


# -----------------------------
# Evaluation
# -----------------------------
all_predictions_np = {}
all_labels_np = {}
results_all = {}

for config in model_configs:
    name       = config["name"]
    model      = config["model"]
    weights    = config["weights"]
    in_frames  = int(config["in_frames"])
    out_frames = int(config["out_frames"])
    predict_fn = config["predict_fn"]

    print(f"\n🔹 Starting evaluation for: {name}  (in={in_frames}, out={out_frames})")

    # Load weights if provided
    if weights is not None and model is not None:
        state = torch.load(weights, map_location=device)
        if isinstance(state, (dict, collections.OrderedDict)):
            try:
                model.load_state_dict(state)
            except Exception as e:
                print(f"[{name}] Failed to load state_dict: {e}")
                continue
        else:
            model = state  # pickled whole module

    if model is not None:
        model.to(device)
        model.eval()

    preds = []
    labels_all = []
    mae_all = []
    rmse_all = []
    iou_all = []
    csi_all = {str(th): [] for th in thresholds}

    for seq in tqdm(sequences_all, desc=name):
        try:
            need = in_frames + out_frames
            if len(seq) < need:
                continue

            # Per-model slicing
            input_files  = seq[0:in_frames]
            target_files = seq[in_frames:in_frames + out_frames]

            input_imgs  = read_stack(input_files)          # [Tin, H, W]
            target_imgs = read_stack(target_files)         # [Tout, H, W]

            # Process target with same scaler/inv-scaler path
            target_proc = data_postprocessing(
                data_preprocessing(target_imgs, device)
            )
            if isinstance(target_proc, torch.Tensor):
                target_proc = target_proc.detach().cpu().numpy()
            if target_proc.ndim == 4 and target_proc.shape[0] == 1:
                target_proc = np.squeeze(target_proc, axis=0)
            if target_proc.ndim != 3:
                raise ValueError(f"Target has unexpected shape {target_proc.shape}")

            # Predict, normalized to [Tout, H, W]
            pred = call_predict(predict_fn, model, input_imgs, out_frames, device)

            if pred.shape != target_proc.shape:
                print(f"[{name}] ⚠ Shape mismatch: pred {pred.shape}, target {target_proc.shape} — skipping")
                continue

            preds.append(pred)
            labels_all.append(target_proc)

            # Per-T metrics
            mae_seq  = np.array([np.array(calculate_MAE (target_proc[t], pred[t])).mean() for t in range(out_frames)])
            rmse_seq = np.array([np.array(calculate_RMSE(target_proc[t], pred[t])).mean() for t in range(out_frames)])
            iou_seq  = np.array([np.array(calculate_IOU (target_proc[t], pred[t])).mean() for t in range(out_frames)])

            mae_all.append(mae_seq)
            rmse_all.append(rmse_seq)
            iou_all.append(iou_seq)

            csi_seq = calculate_CSI(target_proc, pred)
            for th in thresholds:
                csi_all[str(th)].append(csi_seq[str(th)])

        except Exception as e:
            print(f"[{name}] ⚠ Skipping one sequence due to error: {e}")

    print(f"🔸 Finished inference for {name}, attempting to stack predictions...")

    if not preds:
        print(f"[{name}] ⚠ No valid predictions were collected. Skipping result storage.")
        continue

    try:
        all_predictions_np[name] = np.stack(preds)      # [N, T, H, W]
        all_labels_np[name]      = np.stack(labels_all) # [N, T, H, W]
    except Exception as e:
        print(f"[{name}] ❌ np.stack failed: {e}")
        print(f"Example pred shapes: {[p.shape for p in preds[:3]]}")
        continue

    try:
        results_all[name] = {
            "mae":  np.mean(mae_all,  axis=0).tolist(),
            "rmse": np.mean(rmse_all, axis=0).tolist(),
            "iou":  np.mean(iou_all,  axis=0).tolist(),
            "csi":  {th: np.mean(csi_all[th], axis=0).tolist() for th in csi_all}
        }
        print(f"[{name}] ✅ Evaluation results stored.")
    except Exception as e:
        print(f"[{name}] ❌ Failed to aggregate metrics: {e}")

print("\nDone. Results keys:", list(results_all.keys()))
print(results_all)


###########################  Plot metrics


# ---- Final results summary ----
print("\nFinal results:")
for model_name, metrics in results_all.items():
    print(f"\nModel: {model_name}")
    print(f"  MAE: {metrics['mae']}")
    for th in thresholds:
        print(f"  CSI @ {th}: {metrics['csi'][str(th)]}")

def _x_for(y_list):
    return np.arange(1, len(y_list) + 1)

# ---- Plot MAE over Lead Time ----
plt.figure(figsize=(10, 6))
for model_name, metrics in results_all.items():
    mae_values = metrics['mae']  # list of length T (e.g., 12)
    plt.plot(_x_for(mae_values), mae_values, label=model_name)
plt.xlabel("Lead Time (frames)")
plt.ylabel("MAE")
plt.title("MAE over Lead Time")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("mae_over_lead_time.png")
plt.show()

# ---- Plot RMSE over Lead Time ----
plt.figure(figsize=(10, 6))
for model_name, metrics in results_all.items():
    rmse_values = metrics['rmse']
    plt.plot(_x_for(rmse_values), rmse_values, label=model_name)
plt.xlabel("Lead Time (frames)")
plt.ylabel("RMSE")
plt.title("RMSE over Lead Time")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("rmse_over_lead_time.png")
plt.show()

# ---- Plot IOU over Lead Time ----
plt.figure(figsize=(10, 6))
for model_name, metrics in results_all.items():
    iou_values = metrics['iou']
    plt.plot(_x_for(iou_values), iou_values, label=model_name)
plt.xlabel("Lead Time (frames)")
plt.ylabel("IOU")
plt.title("IOU over Lead Time")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("iou_over_lead_time.png")
plt.show()

# ---- Plot CSI over Lead Time for each threshold ----
for th in thresholds:
    plt.figure(figsize=(10, 6))
    for model_name, metrics in results_all.items():
        csi_values = metrics['csi'][str(th)]
        plt.plot(_x_for(csi_values), csi_values, label=model_name)
    plt.xlabel("Lead Time (frames)")
    plt.ylabel(f"CSI @ {th} mm/h")
    plt.title(f"CSI over Lead Time (Threshold {th})")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"csi_{th}_over_lead_time.png")
    plt.show()

################################################### Plot Bars
def plot_metric_bar(metric_dict, ylabel, title, filename):
    plt.figure(figsize=(10, 6))
    
    models = list(metric_dict.keys())
    values = list(metric_dict.values())
    
    bars = plt.bar(models, values)
    
    # Add value labels on top of each bar
    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{height:.3f}",
            ha='center',
            va='bottom',
            fontsize=10
        )

    plt.ylabel(ylabel)
    plt.title(title)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(filename)
    plt.show()

# ---- MAE ----
mae_avg = {model_name: np.mean(metrics['mae']) for model_name, metrics in results_all.items()}
plot_metric_bar(mae_avg, "Average MAE", "Average MAE across 12 Lead Times", "avg_mae_per_model.png")

# ---- RMSE ----
rmse_avg = {model_name: np.mean(metrics['rmse']) for model_name, metrics in results_all.items()}
plot_metric_bar(rmse_avg, "Average RMSE", "Average RMSE across 12 Lead Times", "avg_rmse_per_model.png")

# ---- IOU ----
iou_avg = {model_name: np.mean(metrics['iou']) for model_name, metrics in results_all.items()}
plot_metric_bar(iou_avg, "Average IOU", "Average IOU across 12 Lead Times", "avg_iou_per_model.png")

# ---- CSI for each threshold ----
for th in thresholds:
    csi_avg = {
        model_name: np.mean(metrics['csi'][str(th)])
        for model_name, metrics in results_all.items()
    }
    plot_metric_bar(
        csi_avg,
        f"Average CSI @ {th} mm/h",
        f"Average CSI across 12 Lead Times (Threshold {th} mm/h)",
        f"avg_csi_{th}_per_model.png"
    )



#######################################################save to file 

#Saving results in JSON format for readability and interoperability
import json

with open('/path/results_all.json', 'w') as f:
    json.dump(results_all, f, indent=4)

print("Results have been successfully saved to results_all.json")


############  Plot Predictions


# plot animation for for a defined model and for a defined sequence 1, or 2,......

prediction = all_predictions_np['GroupUNet3D_4in_12out'][1]  # [T,H,W]
label = all_labels_np['GroupUNet3D_4in_12out'][1]


# Plot predictions as images at defined lead times and for defined models   
path_plots = r"/home/sv20953"
plot_animations(label, prediction, path_plots)

plot_selected_lead_times_comparison(
    all_predictions_np=all_predictions_np,
    all_labels_np=all_labels_np,
    selected_models=[
        
        
        "Standard_UNet2D",
        "StandardUNet3D_4in_12out",
        "GhostUNet3D_4in_12out",
        "DepthwiseUNet3D_4in_12out",
        "GroupUNet3D_4in_12out",
        "HybridUNet3D_4in_12out",
        "R2Plus1DUNet3D_4in_12out",
        "ShiftUNet3D_4in_12out",
    ],
    renamed_models={
        "StandardUNet3D_4in_12out": "Standard_UNet3D",
        "GhostUNet3D_4in_12out": "Ghost_UNet3D",
        "DepthwiseUNet3D_4in_12out": "Depthwise_UNet3D",
        "GroupUNet3D_4in_12out": "Group_UNet3D",
        "HybridUNet3D_4in_12out": "Hybrid_UNet3D",
        "R2Plus1DUNet3D_4in_12out": "R2Plus1D_UNet3D",
        "ShiftUNet3D_4in_12out": "Shift_UNet3D",
        
    },
    sequence_index=1,
    lead_times=[0, 5, 11],
    save_path="/home/sv20953/plots/predictions_lead_times_5_30_60.png"
)


#################################################  calculate efficiency
# calculate efficiency such as parameters, inference time , etc




# -----------------------------
# 1) Choose which models to run
# -----------------------------
selected_models = [
    "Standard_UNet2D",
    "StandardUNet3D_4in_12out",
    "GhostUNet3D_4in_12out",
    "DepthwiseUNet3D_4in_12out",
    "GroupUNet3D_4in_12out",
    "HybridUNet3D_4in_12out",
    "R2Plus1DUNet3D_4in_12out",
    "ShiftUNet3D_4in_12out",
]  # <-- NO trailing comma

# 2) Tag -> class (import/define your classes above this file)
model_classes = {
    "Standard_UNet2D": Standard_UNet2D,
    "StandardUNet3D_4in_12out": StandardUNet3D_4in_12out,
    "GhostUNet3D_4in_12out": GhostUNet3D_4in_12out,
    "DepthwiseUNet3D_4in_12out": DepthwiseUNet3D_4in_12out,
    "GroupUNet3D_4in_12out": GroupUNet3D_4in_12out,
    "HybridUNet3D_4in_12out": HybridUNet3D_4in_12out,
    "R2Plus1DUNet3D_4in_12out": R2Plus1DUNet3D_4in_12out,
    "ShiftUNet3D_4in_12out": ShiftUNet3D_4in_12out,
}

# 3)  names for the table/prints
renamed_models = {
    "Standard_UNet2D": "Standard_UNet2D",
    "StandardUNet3D_4in_12out": "Standard_UNet3D",
    "GhostUNet3D_4in_12out": "Ghost_UNet3D",
    "DepthwiseUNet3D_4in_12out": "Depthwise_UNet3D",
    "GroupUNet3D_4in_12out": "Group_UNet3D",
    "HybridUNet3D_4in_12out": "Hybrid_UNet3D",
    "R2Plus1DUNet3D_4in_12out": "R2Plus1D_UNet3D",
    "ShiftUNet3D_4in_12out": "Shift_UNet3D",
}  # <-- NO trailing comma

# In case they were accidentally tuples somewhere else:
if isinstance(selected_models, tuple):
    selected_models = selected_models[0]
if isinstance(renamed_models, tuple):
    renamed_models = renamed_models[0]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
WARMUP, TEST = 5, 20

def bytes_to_mb(x): return x / (1024 ** 2)

def make_relu_safe(module):
    for name, child in module.named_children():
        if isinstance(child, torch.nn.ReLU) and child.inplace:
            setattr(module, name, torch.nn.ReLU(inplace=False))
        else:
            make_relu_safe(child)

def make_dummy_input_and_shape(tag, device):
    """
    Returns:
      x: torch.Tensor dummy input
      input_shape_for_flops: tuple passed to ptflops (C,H,W) or (C,D,H,W)
    """
    if "2D" in tag:  # 2D nets expect [B, C, H, W]
        C = 3  # your Standard_UNet2D default is in_channels=3
        H = W = 512
        x = torch.randn(1, C, H, W, device=device)
        shape_for_flops = (C, H, W)
    else:           # 3D nets expect [B, C, D, H, W]
        C, D, H, W = 1, 4, 512, 512
        x = torch.randn(1, C, D, H, W, device=device)
        shape_for_flops = (C, D, H, W)
    return x, shape_for_flops

results = {}
for tag in selected_models:
    if tag not in model_classes:
        print(f"⚠️  Skipping unknown model tag: {tag}")
        continue

    Net = model_classes[tag]
    name = renamed_models.get(tag, tag)
    print(f"\n🔍 Evaluating {name}")

    # ---------------- Inference profile ----------------
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    net = Net().to(device).eval()
    x, shape_for_flops = make_dummy_input_and_shape(tag, device)

    # FLOPs/params (if ptflops is available)
    if HAVE_PTFLOPS:
        try:
            flops_str, params_str = get_model_complexity_info(
                net, shape_for_flops,
                as_strings=True,
                print_per_layer_stat=False,
                verbose=False
            )
        except Exception as e:
            print(f"   ⚠️  ptflops failed: {e}")
            flops_str = "n/a"
    else:
        flops_str = "n/a"

    n_params  = sum(p.numel() for p in net.parameters())
    param_mem = bytes_to_mb(n_params * 4)  # fp32

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    with torch.no_grad():
        # warmup
        for _ in range(WARMUP):
            _ = net(x)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(TEST):
            _ = net(x)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        latency_ms = (time.time() - t0) / TEST * 1e3

    if torch.cuda.is_available():
        inf_peak_mb = bytes_to_mb(torch.cuda.max_memory_allocated())
    else:
        inf_peak_mb = float("nan")

    results[name] = dict(
        Params_M     = f"{n_params/1e6:.2f}",
        ParamMem_MB  = f"{param_mem:.1f}",
        FLOPs        = flops_str,
        Latency_ms   = f"{latency_ms:.1f}",
        InfPeak_MB   = f"{inf_peak_mb:.0f}" if torch.cuda.is_available() else "n/a",
        TrainPeak_MB = "n/a"
    )

    # --------------- Training peak (optional) ---------------
    try:
        net_t = Net().to(device).train()
        make_relu_safe(net_t)
        x_t, _ = make_dummy_input_and_shape(tag, device)
        x_t.requires_grad_(True)

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        out = net_t(x_t)
        loss = torch.nn.functional.mse_loss(out, torch.zeros_like(out))
        loss.backward()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            train_peak_mb = bytes_to_mb(torch.cuda.max_memory_allocated())
            results[name]["TrainPeak_MB"] = f"{train_peak_mb:.0f}"
        else:
            results[name]["TrainPeak_MB"] = "n/a"
    except Exception as e:
        print(f"   ⚠️  Train mem failed: {e}")
    finally:
        del net_t, x_t, out, loss
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

# ---------------- table ----------------
rows = [{"Model": k, **v} for k, v in results.items()]
print("\n📊 Complexity Summary (MB, batch = 1, 512²)")
print(tabulate(rows, headers="keys", tablefmt="github"))






