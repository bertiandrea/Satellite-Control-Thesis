import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

TAGS = ["Reward_policy/phi_mean", "Reward_policy/energy_mean", 
        "Reward_policy/du_energy_mean", "Reward_policy/max_torque_mean"]

LOG_DISPLAY = [
    "Reward_policy/phi_mean",
    "Reward_policy/max_torque_mean"
]

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

    for t in TAGS:
        plt.figure(figsize=(10, 6))
        
        # --- PATCH LOG SCALE ---
        if t in LOG_DISPLAY:
            plt.yscale('log')
        # -----------------------

        colors = plt.cm.get_cmap('tab20', len(all_results))

        for i, gid in enumerate(sorted(all_results.keys())):
            if t not in all_results[gid]: continue
            
            d = all_results[gid][t]
            line, = plt.plot(d["x"], d["m"], label=gid, color=colors(i), linewidth=1.5)
            #plt.fill_between(d["x"], d["min"], d["max"], color=line.get_color(), alpha=0.2)

        plt.title(t.replace('_', ' ').title() + (" (Log Scale)" if t in LOG_DISPLAY else ""))
        plt.grid(True, which="both", alpha=0.3) # "both" per vedere la griglia logaritmica
        plt.legend(fontsize='x-small', ncol=2)
        plt.xlabel("Steps")
        plt.ylabel("Value")
        plt.tight_layout()
        plt.savefig(out / f"{t.replace('/', '_')}.png", dpi=150)
        plt.close()

if __name__ == "__main__":
    main()