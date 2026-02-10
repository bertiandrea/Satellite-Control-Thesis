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
        m = [np.median(step_data[t][s]) for s in steps]
        mi = [np.min(step_data[t][s]) for s in steps]
        ma = [np.max(step_data[t][s]) for s in steps]
        result[t] = {"x": np.array(steps), "m": np.array(m), "min": np.array(mi), "max": np.array(ma)}
    return result

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=str, required=True)
    ap.add_argument("--outdir", default="_img/plots_reward_policy")
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
            plt.yscale("symlog", linthresh=1e0)
        # -----------------------

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

        plt.title(t.replace('_', ' ').title() + (" (Log Scale)" if t in LOG_DISPLAY else ""))
        plt.grid(True, which="both", alpha=0.3)
        plt.legend(fontsize='x-small', ncol=2)
        plt.xlabel("Steps")
        plt.ylabel("Value")
        plt.tight_layout()
        plt.savefig(out / f"{t.replace('/', '_')}.png", dpi=150)
        plt.close()

    base_runs = sorted([gid for gid in all_results.keys() if "_noise" not in gid], key=nat_key)
    
    sub_h = " |  NOMINAL  |   NOISE   |   DIFF%   "
    name_w = 60
    h_top = " " * name_w
    h_mid = f"{'RUN ID':<{name_w}}"
    
    for t in TAGS:
        h_top += f" | {t.split('/')[-1][:12]:^34}"
        h_mid += sub_h

    print(f"\n{h_top}\n{h_mid}\n{'-' * len(h_mid)}")

    for base in base_runs:
        row = f"{base:<{name_w}}"
        p = base.split('/')
        noise_id = f"{p[0]}_noise/{'/'.join(p[1:])}" if len(p) >= 2 else f"{base}_noise"

        for t in TAGS:
            d_nom = all_results.get(base, {}).get(t, {})
            d_noi = all_results.get(noise_id, {}).get(t, {})
            
            v_nom = np.mean(d_nom["m"][-max(1, int(len(d_nom["m"])*0.05)):]) if "m" in d_nom else None
            v_noi = np.mean(d_noi["m"][-max(1, int(len(d_noi["m"])*0.05)):]) if "m" in d_noi else None

            f = lambda v: f"{v:9.2e}" if v is not None and abs(v) > 9999 else (f"{v:9.2f}" if v is not None else "      N/A")
            
            s_nom = f(v_nom)
            s_noi = f(v_noi)
            s_diff = "        -"
            if v_nom and v_noi:
                diff = (v_noi - v_nom) / abs(v_nom) * 100
                s_diff = f"{diff:+8.1e}%" if abs(diff) > 999 else f"{diff:+8.1f}%"

            row += f" | {s_nom} | {s_noi} | {s_diff} "
        
        print(row)
    print("=" * len(h_mid))

if __name__ == "__main__":
    main()