import torch
import matplotlib.pyplot as plt
import numpy as np
import os

LOG_DIR = "../Evaluating/logs/"

file_to_name = {
    "trajectories_20251113_224826.pt": "Baseline Fixed Seed",
    "trajectories_20251113_224838.pt": "Baseline Random Seed 1",
    "trajectories_20251114_082538.pt": "Baseline Random Seed 2",

    "trajectories_20251114_082603.pt": "CAPS T:0.1 S:0.01 R:0.5",
    "trajectories_20251114_105501.pt": "CAPS T:0.1 S:0.1 R:0.5",
}

# ------------------- FFT + CoM -------------------
def compute_fft(actions, fps=60):
    T, N, D = actions.shape
    freqs = torch.fft.rfftfreq(T, d=1.0/fps)
    all_fft, all_com = [], []

    for i in range(N):
        fft_env = torch.fft.rfft(actions[:, i, :], dim=0)
        amp_env = fft_env.abs() / T
        all_fft.append(amp_env)
        all_com.append((amp_env * freqs[:, None]).sum(dim=0) / amp_env.sum(dim=0))

    all_fft = torch.stack(all_fft)
    all_com = torch.stack(all_com)

    return all_fft.mean(dim=0), all_fft.std(dim=0), freqs, all_com.mean(dim=0), all_com.std(dim=0)

# ------------------- FFT PLOT -------------------
def plot_fft(runs, title, labels):
    colors = plt.cm.tab10(np.linspace(0, 1, len(runs)))

    for axis in range(len(labels)):
        plt.figure(figsize=(7,4))
        for (name, metrics), color in zip(runs, colors):
            mean = metrics["FFT_mean"][:, axis].numpy()
            std = metrics["FFT_std"][:, axis].numpy()
            freqs = metrics["FFT_freqs"].numpy()
            com_mean = metrics["FFT_CoM_mean"][axis].item()
            com_std  = metrics["FFT_CoM_std"][axis].item()
            lower, upper = mean - std, mean + std
            lower = np.maximum(lower, 0.0)
            plt.plot(freqs, mean, color=color, label=name, linewidth=1.5)
            plt.fill_between(freqs, lower, upper, color=color, alpha=0.25)

            plt.axvline(com_mean, color=color, linestyle="--", lw=2)
            plt.axvspan(com_mean-com_std, com_mean+com_std, color=color, alpha=0.15)

        plt.yscale("symlog", linthresh=1e-2)
        plt.ylim(bottom=0.0)
        plt.title(f"{title} — {labels[axis]}")
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Amplitude")
        plt.grid(True, linestyle="--", alpha=0.3)
        plt.legend()
        plt.tight_layout()

        fname = f"{title.replace(' ','_').lower()}_{labels[axis].lower()}.png"
        plt.savefig(fname, dpi=600)
        plt.close()

# ------------------- LOAD RUNS -------------------
loaded_runs = []
for filename, run_name in file_to_name.items():
    path = os.path.join(LOG_DIR, filename)
    if not os.path.exists(path):
        print(f"Missing file: {filename}")
        continue

    print(f"Loading: {filename}")
    data = torch.load(path, map_location="cpu", weights_only=True)
    actions = torch.stack([e["actions"] for e in data])

    fft_mean, fft_std, freqs, com_mean, com_std = compute_fft(actions)
    loaded_runs.append((f"{run_name}_axis", {
        "FFT_mean": fft_mean, "FFT_std": fft_std,
        "FFT_freqs": freqs, "FFT_CoM_mean": com_mean,
        "FFT_CoM_std": com_std
    }))

    actions_norm = torch.linalg.norm(actions, dim=2, keepdim=True)
    fft_mean_n, fft_std_n, freqs_n, com_mean_n, com_std_n = compute_fft(actions_norm)
    loaded_runs.append((f"{run_name}_norm", {
        "FFT_mean": fft_mean_n, "FFT_std": fft_std_n,
        "FFT_freqs": freqs_n, "FFT_CoM_mean": com_mean_n,
        "FFT_CoM_std": com_std_n
    }))

# ------------------- PLOT -------------------
axis_runs = [r for r in loaded_runs if "_axis" in r[0]]
plot_fft(axis_runs, "FFT per Axis", ["X","Y","Z"])

norm_runs = [r for r in loaded_runs if "_norm" in r[0]]
plot_fft(norm_runs, "FFT Norm", ["Norm"])