import argparse
import re
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict

def nat_key(s):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r'(\d+)', str(s))]

def plot_component_across_files(out_dir, title, list_of_data, labels, non_negative=False, log_scale=False):
    C = list_of_data[0][2].shape[2]

    list_of_data.sort(key=lambda x: nat_key(x[0]))
    
    unique_groups = sorted([x[0] for x in list_of_data], key=nat_key)
    color_map = {name: plt.cm.tab10(i % 10) for i, name in enumerate(unique_groups)}

    for scale in (["linear", "log"] if log_scale else ["linear"]):
        plt.figure(figsize=(14, 3 * C))
        for i, label in enumerate(labels):
            if i >= C: break
            plt.subplot(C, 1, i + 1)

            for run_name, steps, data in list_of_data:
                mean = data[:, :, i].mean(dim=1).numpy()
                std = data[:, :, i].std(dim=1).numpy()
                step_np = steps.numpy()

                lower, upper = mean - std, mean + std
                if non_negative:
                    lower = np.maximum(lower, 0.0)

                color = color_map[run_name]
                plt.plot(step_np, mean, label=run_name, color=color, linewidth=1.5)
                plt.fill_between(step_np, lower, upper, color=color, alpha=0.15)

            if scale == "log":
                plt.yscale("symlog", linthresh=1e0)

            plt.title(f"{title} – {label} ({scale})")
            plt.ylabel(f"{label} ({scale})")
            plt.grid(True, linestyle="--", alpha=0.4)
            plt.legend(loc='upper right', fontsize='x-small', ncol=2)

        plt.xlabel("Step")
        plt.tight_layout()
        plt.savefig(out_dir / f"{title.replace(' ', '_').lower()}_{scale}.png", dpi=300)
        plt.close()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=str, required=True)
    ap.add_argument("--outdir", type=str, default="_img/plots_trajectories")
    args = ap.parse_args()

    base_path = Path(args.input)
    paths = list(base_path.rglob("*.pt*"))

    grouped = defaultdict(list)
    for p in paths:
        gid = f"{p.parts[-4]}/{p.parts[-3]}"
        grouped[gid].append(p)

    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)

    # ---------------- LOAD DATA ----------------
    all_results = {} 
    for gid, files in grouped.items():
        print(f"--> Loading Group: {gid} ({len(files)} files)")
        try:
            list_quat, list_angdiff, list_angvel, list_angacc, list_actions = [], [], [], [], []
            list_steps = []
            for f in files:
                data = torch.load(f, map_location="cpu", weights_only=True)
                
                d_steps   = torch.tensor([d["step"] for d in data])
                d_quat    = torch.stack([d["quat"] for d in data])
                d_angdiff = torch.stack([d["ang_diff"] for d in data]).unsqueeze(-1)
                d_angvel  = torch.stack([d["angvel"] for d in data])
                d_angacc  = torch.stack([d["angacc"] for d in data])
                d_actions = torch.stack([d["actions"] for d in data])

                list_steps.append(d_steps)
                list_quat.append(d_quat)
                list_angdiff.append(d_angdiff)
                list_angvel.append(d_angvel)
                list_angacc.append(d_angacc)
                list_actions.append(d_actions)

            all_results[gid] = {
                "steps":   list_steps[0],  # niente slicing qui
                "quat":    torch.cat(list_quat, dim=1),
                "angdiff": torch.cat(list_angdiff, dim=1),
                "angvel":  torch.cat(list_angvel, dim=1),
                "angacc":  torch.cat(list_angacc, dim=1),
                "actions": torch.cat(list_actions, dim=1),
            }
        except Exception as e:
            print(f"   [!] Error loading {gid}: {e}")

    def extract(key):
        return [(gid, d["steps"], d[key]) for gid, d in all_results.items()]
    
    if not all_results:
        print("No valid data found.")
        return

    plot_component_across_files(out, "Quaternion", extract("quat"), ["x", "y", "z", "w"])
    plot_component_across_files(out, "Angular Difference", extract("angdiff"), ["angle"], non_negative=True, log_scale=True)
    plot_component_across_files(out, "Angular Velocity", extract("angvel"), ["x", "y", "z"])
    plot_component_across_files(out, "Angular Acceleration", extract("angacc"), ["x", "y", "z"])
    plot_component_across_files(out, "Actions", extract("actions"), ["x", "y", "z"], log_scale=True)

if __name__ == "__main__":
    main()