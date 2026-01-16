#!/usr/bin/env python3
import json
import re
import sys
from datetime import datetime, timedelta
from pathlib import Path

EXPECTED_SEEDS = {420, 42000, 4200000}

# Config: ..._checkpoints_agent_<agent>_YYYYMMDD_HHMMSS.json
CFG_RE = re.compile(r".+_checkpoints_agent_(\d+)_(\d{8})_(\d{6})\.json$")

# Run dir: Jan09_10-04-00_...
RUN_RE = re.compile(r"([A-Za-z]{3})(\d{2})_(\d{2})-(\d{2})-(\d{2})_.+")


def cfg_timestamp(cfg_name: str) -> datetime:
    m = CFG_RE.match(cfg_name)
    if not m:
        raise ValueError("config filename does not match expected pattern")
    date, time = m.group(2), m.group(3)
    return datetime.strptime(date + time, "%Y%m%d%H%M%S")


def run_timestamp(run_dir_name: str, year: int) -> datetime | None:
    m = RUN_RE.match(run_dir_name)
    if not m:
        return None
    mon, day, hh, mm, ss = m.group(1), m.group(2), m.group(3), m.group(4), m.group(5)
    return datetime.strptime(f"{year} {mon} {day} {hh}:{mm}:{ss}", "%Y %b %d %H:%M:%S")


def find_seed(obj) -> int | None:
    """Find first integer value under key 'seed' anywhere in JSON."""
    if isinstance(obj, dict):
        if "seed" in obj and isinstance(obj["seed"], int):
            return obj["seed"]
        for v in obj.values():
            s = find_seed(v)
            if s is not None:
                return s
    elif isinstance(obj, list):
        for it in obj:
            s = find_seed(it)
            if s is not None:
                return s
    return None


def corresponding_runs_dir(evaluating_root: Path, cfg_run_dir: Path) -> Path:
    # cfg_run_dir: Evaluating/run_config/run_base  -> runs/runs_base
    runs_dir_name = cfg_run_dir.name.replace("run_", "runs_", 1)
    return evaluating_root / "runs" / runs_dir_name


def best_match_run(cfg_ts: datetime, run_dirs: list[Path], max_delay: timedelta, used: set[Path]) -> Path | None:
    """Pick the closest run dir whose timestamp is within [cfg_ts, cfg_ts + max_delay]."""
    year = cfg_ts.year
    candidates: list[tuple[timedelta, Path]] = []

    for rd in run_dirs:
        if rd in used:
            continue
        rts = run_timestamp(rd.name, year)
        if rts is None:
            continue
        if cfg_ts <= rts <= cfg_ts + max_delay:
            candidates.append((rts - cfg_ts, rd))

    if not candidates:
        return None

    candidates.sort(key=lambda x: x[0])  # smallest delta first
    return candidates[0][1]


def audit_agent_folder(agent_cfg_dir: Path, agent_runs_dir: Path, max_delay_s: int) -> list[str]:
    """Returns list of error strings for this agent folder."""
    errors: list[str] = []
    max_delay = timedelta(seconds=max_delay_s)

    cfg_files = sorted(agent_cfg_dir.glob("*.json"))
    if len(cfg_files) != 3:
        errors.append(f"{agent_cfg_dir}: expected 3 config JSON, found {len(cfg_files)}")

    if not agent_runs_dir.is_dir():
        errors.append(f"{agent_runs_dir}: missing runs agent directory")
        return errors

    run_dirs = sorted([d for d in agent_runs_dir.iterdir() if d.is_dir()])
    if len(run_dirs) != 3:
        errors.append(f"{agent_runs_dir}: expected exactly 3 run directories, found {len(run_dirs)}")

    # Seeds check
    seeds = set()
    for cf in cfg_files:
        try:
            data = json.loads(cf.read_text())
        except Exception as e:
            errors.append(f"{cf}: cannot parse JSON ({e})")
            continue

        seed = find_seed(data)
        if seed is None:
            errors.append(f"{cf}: seed not found in JSON")
        else:
            seeds.add(seed)

    if cfg_files and seeds != EXPECTED_SEEDS:
        errors.append(f"{agent_cfg_dir}: seeds mismatch. Found={sorted(seeds)} Expected={sorted(EXPECTED_SEEDS)}")

    # Pairing check (1 config -> 1 run, within +5s)
    used_runs: set[Path] = set()
    for cf in cfg_files:
        try:
            ts = cfg_timestamp(cf.name)
        except Exception:
            errors.append(f"{cf}: filename does not match pattern '*_YYYYMMDD_HHMMSS.json'")
            continue

        match = best_match_run(ts, run_dirs, max_delay, used_runs)
        if match is None:
            errors.append(f"{cf}: no matching run within +{max_delay_s}s under {agent_runs_dir}")
        else:
            used_runs.add(match)

    return errors


def main():
    # Default root as in your project
    evaluating_root = Path("~/Satellite-Control-Thesis/Evaluating").expanduser().resolve()
    max_delay_s = 5

    # Allow optional CLI args (kept minimal)
    if len(sys.argv) >= 2:
        evaluating_root = Path(sys.argv[1]).expanduser().resolve()
    if len(sys.argv) >= 3:
        max_delay_s = int(sys.argv[2])

    run_config_root = evaluating_root / "run_config"
    if not run_config_root.is_dir():
        print(f"[FATAL] Missing: {run_config_root}")
        sys.exit(1)

    all_errors: list[str] = []

    for cfg_run_dir in sorted([d for d in run_config_root.glob("run_*") if d.is_dir()]):
        runs_root = corresponding_runs_dir(evaluating_root, cfg_run_dir)
        if not runs_root.is_dir():
            all_errors.append(f"{runs_root}: missing runs directory for {cfg_run_dir}")
            continue

        for agent_cfg_dir in sorted([a for a in cfg_run_dir.glob("agent_*") if a.is_dir()]):
            agent_runs_dir = runs_root / agent_cfg_dir.name
            errs = audit_agent_folder(agent_cfg_dir, agent_runs_dir, max_delay_s)
            all_errors.extend(errs)

    if all_errors:
        print("[FAIL] Issues found:")
        for e in all_errors:
            print("  -", e)
        sys.exit(1)

    print("[OK] All checks passed.")
    sys.exit(0)


if __name__ == "__main__":
    main()
