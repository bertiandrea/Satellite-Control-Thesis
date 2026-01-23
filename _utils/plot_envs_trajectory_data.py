import torch
import matplotlib.pyplot as plt
import random
import os

LOG_DIR = "../Evaluating/logs/"

file_to_name = {
    "logs_base/trajectories_XXXXXXXX_XXXXXX.pt": "Baseline",
    "logs_base_noise/trajectories_XXXXXXXX_XXXXXX.pt": "Baseline + Noise",

    "logs_lambda_normalization/trajectories_XXXXXXXX_XXXXXX.pt": "Normalization sum(act[t])^2:0.1",
    "logs_lambda_normalization_noise/trajectories_XXXXXXXX_XXXXXX.pt": "Normalization sum(act[t])^2:0.1 + Noise",

    "logs_delta_lambda_normalization/trajectories_XXXXXXXX_XXXXXX.pt": "Normalization sum(act[t])^2:0.1 sum(act[t]-act[t-1])^2:0.05",
    "logs_delta_lambda_normalization_noise/trajectories_XXXXXXXX_XXXXXX.pt": "Normalization sum(act[t])^2:0.1 sum(act[t]-act[t-1])^2:0.05 + Noise",

    "logs_delta_lambda_normalization_randomization/trajectories_XXXXXXXX_XXXXXX.pt": "Normalization sum(act[t])^2:0.1 sum(act[t]-act[t-1])^2:0.05 + Randomization",
    "logs_delta_lambda_normalization_randomization_noise/trajectories_XXXXXXXX_XXXXXX.pt": "Normalization sum(act[t])^2:0.1 sum(act[t]-act[t-1])^2:0.05 + Randomization + Noise",

    "logs_CAPS/trajectories_XXXXXXXX_XXXXXX.pt": "CAPS T:1e-9 S:1e-9 R:0.01",
    "logs_CAPS_noise/trajectories_XXXXXXXX_XXXXXX.pt": "CAPS T:1e-9 S:1e-9 R:0.01 + Noise",

    "logs_CAPS_randomization/trajectories_XXXXXXXX_XXXXXX.pt": "CAPS T:1e-9 S:1e-9 R:0.01 + Randomization",
    "logs_CAPS_randomization_noise/trajectories_XXXXXXXX_XXXXXX.pt": "CAPS T:1e-9 S:1e-9 R:0.01 + Randomization + Noise",

    "logs_explosion/trajectories_XXXXXXXX_XXXXXX.pt": "Explosion",
}

def plot_component_across_files(title, list_of_data, labels, non_negative=False, log_scale=False):
    C = list_of_data[0][2].shape[2]

    # -------- LINEAR PLOT --------
    plt.figure(figsize=(14, 3*C))
    for i, label in enumerate(labels):
        plt.subplot(C, 1, i+1)

        for run_name, steps, data in list_of_data:
            mean = data[:, :, i].mean(dim=1)
            std = data[:, :, i].std(dim=1)

            lower, upper = mean - std, mean + std
            if non_negative:
                lower = torch.clamp(lower, min=0.0)

            plt.plot(steps, mean, label=run_name)
            plt.fill_between(steps, lower, upper, alpha=0.15)

        plt.title(f"{title} – {label}")
        plt.ylabel(label)
        plt.grid(True, linestyle="--", alpha=0.4)

    plt.xlabel("Step")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{title.replace(' ','_').lower()}_all.png", dpi=600)
    plt.close()

    # -------- LOG PLOT --------
    if log_scale:
        plt.figure(figsize=(14, 3*C))
        for i, label in enumerate(labels):
            plt.subplot(C, 1, i+1)

            for run_name, steps, data in list_of_data:
                mean = data[:, :, i].mean(dim=1)
                std = data[:, :, i].std(dim=1)

                lower, upper = mean - std, mean + std
                if non_negative:
                    lower = torch.clamp(lower, min=0.0)

                plt.plot(steps, mean, label=run_name)
                plt.fill_between(steps, lower, upper, alpha=0.15)

            plt.yscale("symlog", linthresh=1e0)

            plt.title(f"{title} – {label} [log]")
            plt.ylabel(label + " [log]")
            plt.grid(True, which="both", linestyle="--", alpha=0.4)

        plt.xlabel("Step")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{title.replace(' ','_').lower()}_all_log.png", dpi=600)
        plt.close()


loaded_runs = []
for filename, run_name in file_to_name.items():
    path = os.path.join(LOG_DIR, filename)
    if not os.path.exists(path):
        print(f"WARNING: missing file {filename}")
        continue

    print(f"Caricamento log da: {path}")
    data = torch.load(path, map_location="cpu", weights_only=True)
    steps = torch.tensor([d["step"] for d in data])
    quat     = torch.stack([d["quat"]     for d in data])
    angdiff  = torch.stack([d["ang_diff"] for d in data]).unsqueeze(-1)
    angvel   = torch.stack([d["angvel"]   for d in data])
    angacc   = torch.stack([d["angacc"]   for d in data])
    actions  = torch.stack([d["actions"]  for d in data])

    loaded_runs.append((run_name, steps, quat, angdiff, angvel, angacc, actions))

def extract(loaded_runs, idx):
    return [(name, steps, data[idx]) for (name, steps, *data) in loaded_runs]

plot_component_across_files("Quaternion", extract(loaded_runs, 0), ["x","y","z","w"])
plot_component_across_files("Angular difference (deg)", extract(loaded_runs, 1), ["angle"], non_negative=True, log_scale=True)
plot_component_across_files("Angular velocity", extract(loaded_runs, 2), ["x","y","z"])
plot_component_across_files("Angular acceleration", extract(loaded_runs, 3), ["x","y","z"])
plot_component_across_files("Actions", extract(loaded_runs, 4), ["x","y","z"], log_scale=True)
