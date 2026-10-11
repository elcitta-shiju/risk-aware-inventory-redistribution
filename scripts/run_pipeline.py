"""Run the full pipeline end to end (each stage is also runnable on its own).

  python scripts/run_pipeline.py                 # all stages
  python scripts/run_pipeline.py --skip-rl       # reuse existing RL checkpoints
  python scripts/run_pipeline.py --from forecasting
"""

import _bootstrap  # noqa: F401

import argparse
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
STAGES = [
    ("extract", "extract_data.py"),
    ("audit", "run_data_audit.py"),
    ("preprocess", "run_preprocess.py"),
    ("forecasting", "run_forecasting.py"),
    ("risk", "run_risk.py"),
    ("rl", "run_rl.py"),
    ("evaluation", "run_evaluation.py"),
]

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--from", dest="start", choices=[s for s, _ in STAGES], default="extract")
    ap.add_argument("--skip-rl", action="store_true")
    args = ap.parse_args()
    names = [s for s, _ in STAGES]
    for name, script in STAGES[names.index(args.start):]:
        if name == "rl" and args.skip_rl:
            print("== skipping rl")
            continue
        print(f"== {name}: {script}", flush=True)
        t0 = time.time()
        r = subprocess.run([sys.executable, str(HERE / script)])
        if r.returncode != 0:
            sys.exit(f"stage '{name}' failed with exit code {r.returncode}")
        print(f"== {name} done in {time.time() - t0:.0f}s", flush=True)
