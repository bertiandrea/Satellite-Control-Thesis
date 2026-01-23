import torch
import matplotlib.pyplot as plt
import numpy as np
import os

LOG_DIR = "../Evaluating/logs/"

file_to_name = {
    "logs_delta_lambda_normalization/trajectories_XXXXXXXX_XXXXXX.pt": "Normalization sum(act[t])^2:0.1 sum(act[t]-act[t-1])^2:0.05",
    "logs_delta_lambda_normalization_noise/trajectories_XXXXXXXX_XXXXXX.pt": "Normalization sum(act[t])^2:0.1 sum(act[t]-act[t-1])^2:0.05 + Noise",

    "logs_delta_lambda_normalization_randomization/trajectories_XXXXXXXX_XXXXXX.pt": "Normalization sum(act[t])^2:0.1 sum(act[t]-act[t-1])^2:0.05 + Randomization",
    "logs_delta_lambda_normalization_randomization_noise/trajectories_XXXXXXXX_XXXXXX.pt": "Normalization sum(act[t])^2:0.1 sum(act[t]-act[t-1])^2:0.05 + Randomization + Noise",

    "logs_CAPS/trajectories_XXXXXXXX_XXXXXX.pt": "CAPS T:1e-9 S:1e-9 R:0.01",
    "logs_CAPS_noise/trajectories_XXXXXXXX_XXXXXX.pt": "CAPS T:1e-9 S:1e-9 R:0.01 + Noise",

    "logs_CAPS_randomization/trajectories_XXXXXXXX_XXXXXX.pt": "CAPS T:1e-9 S:1e-9 R:0.01 + Randomization",
    "logs_CAPS_randomization_noise/trajectories_XXXXXXXX_XXXXXX.pt": "CAPS T:1e-9 S:1e-9 R:0.01 + Randomization + Noise",
}

# ------------------- FFT + CoM -------------------
def compute_fft(actions, fps):
    T, N, D = actions.shape
    freqs = torch.fft.rfftfreq(T, d=1.0 / fps)
    fft_envs = torch.fft.rfft(actions, dim=0) # [Freq, N, D]
    amp_envs = fft_envs.abs() / T
    mean_fft = amp_envs.mean(dim=1)
    std_fft = amp_envs.std(dim=1)
    com_envs = (amp_envs * freqs[:, None, None]).sum(dim=0) / amp_envs.sum(dim=0)
    return mean_fft, std_fft, freqs, com_envs.mean(dim=0), com_envs.std(dim=0)

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