import argparse
import numpy as np
import matplotlib.pyplot as plt
import re
from pathlib import Path
from collections import defaultdict
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

def nat_key(s):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r'(\d+)', str(s))]

TAGS = ["Reward_policy/phi_mean", "Reward_policy/energy_mean", 
        "Reward_policy/du_energy_mean", "Reward_policy/max_torque_mean"]

LOG_DISPLAY = ["Reward_policy/phi_mean", "Reward_policy/energy_mean", 
        "Reward_policy/du_energy_mean", "Reward_policy/max_torque_mean"]

THRESHOLDS = {
    "Reward_policy/phi_mean": 1,
    "Reward_policy/energy_mean": 100,
    "Reward_policy/du_energy_mean": 1,
    "Reward_policy/max_torque_mean": 1,
}

def load_data(files):
    step_data = defaultdict(lambda: defaultdict(list))
    for f in files:
        try:
            ea = EventAccumulator(str(f), size_guidance={"scalars": 0})
            ea.Reload()
            for t in set(ea.Tags().get("scalars", [])) & set(TAGS):
                for e in ea.Scalars(t):
                    step_data[t][e.step].append(e.value)
        except: continue
    
    result = {}
    for t in step_data:
        steps = sorted(step_data[t].keys())
        m = [np.mean(step_data[t][s]) for s in steps]
        mi = [np.min(step_data[t][s]) for s in steps]
        ma = [np.max(step_data[t][s]) for s in steps]
        result[t] = {"x": np.array(steps), "m": np.array(m), "min": np.array(mi), "max": np.array(ma)}
    return result

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=str, required=True)
    ap.add_argument("--outdir", default="plots_reward_policy")
    args = ap.parse_args()

    base_path = Path(args.input)
    paths = list(base_path.glob("*/*/*/*tfevents*"))
    
    grouped = defaultdict(list)
    for p in paths:
        gid = f"{p.parts[-4]}/{p.parts[-3]}"
        grouped[gid].append(p)

    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)
    
    all_results = {gid: load_data(files) for gid, files in grouped.items()}

    unique_groups = sorted(list(set(gid.split('/')[0] for gid in all_results.keys())), key=nat_key)

    for t in TAGS:
        plt.figure(figsize=(12, 7))
        
        # --- PATCH LOG SCALE ---
        if t in LOG_DISPLAY:
            plt.yscale('log')
        # -----------------------

        thr = THRESHOLDS.get(t, None)
        if thr is not None:
            plt.axhline(thr, linestyle="--", linewidth=1.2, color="red", label="_nolegend_")

        legend_below_flags = []
        for i, gid in enumerate(sorted(all_results.keys(), key=nat_key)):
            if t not in all_results[gid]: continue

            group_name = gid.split('/')[0]
            group_idx = unique_groups.index(group_name)
            base_color = plt.cm.tab10(group_idx % 10) 
            runs_in_group = [g for g in sorted(all_results.keys(), key=nat_key) if g.startswith(group_name)]
            run_idx = runs_in_group.index(gid)
            alpha_val = max(0.3, 1.0 - (run_idx * 0.12))
            color = (base_color[0], base_color[1], base_color[2], alpha_val)
            
            d = all_results[gid][t]
            last_val = np.mean(d["m"][-max(1, int(len(d["m"]) * 0.05)):]) if len(d["m"]) > 0 else 0.0
            line, = plt.plot(d["x"], d["m"], label=f"{gid} ({last_val:.2f})", color=color, linewidth=1.5)
            #plt.fill_between(d["x"], d["min"], d["max"], color=line.get_color(), alpha=0.2)

            below = (thr is not None) and (last_val < thr)
            legend_below_flags.append(below)

        plt.title(t.replace('_', ' ').title() + (" (Log Scale)" if t in LOG_DISPLAY else ""))
        plt.grid(True, which="both", alpha=0.3)
        leg = plt.legend(fontsize='x-small', ncol=2)
        if leg is not None:
            texts = leg.get_texts()
            for txt, below in zip(texts, legend_below_flags):
                if below:
                    txt.set_color("black")
                    txt.set_bbox(dict(
                        facecolor="yellow",
                        edgecolor="none",
                        alpha=0.6,
                        boxstyle="round,pad=0.2"
                    ))
                else:
                    txt.set_bbox(None)

        plt.xlabel("Steps")
        plt.ylabel("Value")
        plt.tight_layout()
        plt.savefig(out / f"{t.replace('/', '_')}.png", dpi=150)
        plt.close()

if __name__ == "__main__":
    main()