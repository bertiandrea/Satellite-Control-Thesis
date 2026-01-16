import argparse
import re
import shutil
from datetime import datetime, timedelta
from pathlib import Path


# Config file example:
# evaluate_config_runs_base_..._PPO_checkpoints_agent_28800_20260109_100400.json
CONFIG_RE = re.compile(
    r"^.+_checkpoints_agent_(?P<agent>\d+?)_(?P<date>\d{8})_(?P<time>\d{6})\.json$"
)

# Run dir example:
# Jan09_10-04-00_mushu_satellite_reward
RUN_RE = re.compile(
    r"^(?P<mon>[A-Za-z]{3})(?P<day>\d{2})_(?P<h>\d{2})-(?P<m>\d{2})-(?P<s>\d{2})_.+$"
)


def mv(src: Path, dst: Path, dry: bool) -> None:
    """Move file/dir to dst, creating parents. No-op if already at destination."""
    src = src.resolve()
    dst = dst.resolve()

    if src == dst:
        return

    dst.parent.mkdir(parents=True, exist_ok=True)
    if dry:
        print(f"[DRY] mv {src} -> {dst}")
    else:
        shutil.move(str(src), str(dst))
        print(f"[OK ] mv {src} -> {dst}")


def parse_config(p: Path):
    """Return (agent_dir_name, timestamp) from config filename or None."""
    m = CONFIG_RE.match(p.name)
    if not m:
        return None
    agent = int(m.group("agent"))
    ts = datetime.strptime(m.group("date") + m.group("time"), "%Y%m%d%H%M%S")
    return f"agent_{agent}", ts


def parse_run_dir_ts(dir_name: str, year: int):
    """Return datetime for a run directory name (given year) or None."""
    m = RUN_RE.match(dir_name)
    if not m:
        return None
    return datetime.strptime(
        f"{year} {m.group('mon')} {m.group('day')} {m.group('h')}:{m.group('m')}:{m.group('s')}",
        "%Y %b %d %H:%M:%S",
    )


def iter_run_dirs(runs_root: Path):
    """
    Yield run directories that match RUN_RE.
    Supports both layouts:
      - runs_root/Jan09_... (unorganized)
      - runs_root/agent_*/Jan09_... (organized)
    """
    for entry in runs_root.iterdir():
        if not entry.is_dir():
            continue
        if entry.name == "real_sat":
            continue

        # If it's an agent folder, scan one level down
        if entry.name.startswith("agent_"):
            for sub in entry.iterdir():
                if sub.is_dir() and RUN_RE.match(sub.name):
                    yield sub
            continue

        # Otherwise it might be a run dir directly under runs_root
        if RUN_RE.match(entry.name):
            yield entry


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config-dir", required=True, help="e.g. Evaluating/run_config/run_base")
    ap.add_argument("--runs-dir", required=True, help="e.g. Evaluating/runs/runs_base")
    ap.add_argument("--max-delay", type=int, default=5)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    cfg_root = Path(args.config_dir).expanduser().resolve()
    runs_root = Path(args.runs_dir).expanduser().resolve()
    max_delay = timedelta(seconds=args.max_delay)

    if not cfg_root.is_dir():
        raise SystemExit(f"config-dir not found: {cfg_root}")
    if not runs_root.is_dir():
        raise SystemExit(f"runs-dir not found: {runs_root}")

    # 1) Collect ALL config jsons under cfg_root (already-organized + new/unorganized)
    configs = []  # [(path, agent_dir_name, ts)]
    for p in sorted(cfg_root.rglob("*.json")):
        parsed = parse_config(p)
        if parsed:
            agent_dir, ts = parsed
            configs.append((p, agent_dir, ts))

    if not configs:
        print("No matching config files found.")
        return

    # 2) Build set of years present in config timestamps (usually 1)
    years = sorted({ts.year for _, _, ts in configs})

    # 3) Index run dirs by year: year -> [(ts, path)] sorted
    runs_by_year = {y: [] for y in years}
    for rd in iter_run_dirs(runs_root):
        for y in years:
            ts = parse_run_dir_ts(rd.name, y)
            if ts:
                runs_by_year[y].append((ts, rd))
                break

    for y in years:
        runs_by_year[y].sort(key=lambda x: x[0])

    used_runs = set()

    # 4) Group configs by agent directory name
    by_agent = {}
    for p, agent_dir, ts in configs:
        by_agent.setdefault(agent_dir, []).append((p, ts))

    # 5) For each agent: move configs into cfg_root/agent_*/ and runs into runs_root/agent_*/
    for agent_dir, items in by_agent.items():
        items.sort(key=lambda x: x[1])  # by ts

        cfg_agent_dir = cfg_root / agent_dir
        runs_agent_dir = runs_root / agent_dir

        for cfg_path, cfg_ts in items:
            # Move config into cfg_root/agent_*/<filename>
            mv(cfg_path, cfg_agent_dir / cfg_path.name, args.dry_run)

            # Find best matching run in [cfg_ts, cfg_ts + max_delay] not used
            candidates = runs_by_year.get(cfg_ts.year, [])
            eligible = []
            for r_ts, r_path in candidates:
                if r_path in used_runs:
                    continue
                if cfg_ts <= r_ts <= cfg_ts + max_delay:
                    eligible.append((r_ts, r_path))

            if not eligible:
                print(f"[WARN] No run within +{args.max_delay}s for {cfg_path.name}")
                continue

            # choose closest (min delta)
            r_ts, r_path = min(eligible, key=lambda x: (x[0] - cfg_ts, x[0]))
            used_runs.add(r_path)

            # Move run directory into runs_root/agent_*/<run_dir_name>
            mv(r_path, runs_agent_dir / r_path.name, args.dry_run)

    print("Done.")


if __name__ == "__main__":
    main()
