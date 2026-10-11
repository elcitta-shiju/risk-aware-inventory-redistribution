"""Stage 8: train PPO agents (point-forecast vs uncertainty-aware observations) over several seeds.

Outputs: models/rl/{variant}_seed{k}/best_model.zip, final_model.zip, validation_curve.csv, meta.json
Usage:  python scripts/run_rl.py [--timesteps N] [--seeds 0 1 2]
"""

import _bootstrap  # noqa: F401

import argparse
import json

from inventory_rl.agents.ppo import train_ppo
from inventory_rl.environment.factory import load_context
from inventory_rl.utils.config import load_config, save_json
from inventory_rl.utils.paths import MODELS_DIR

VARIANTS = {"rl_point": False, "rl_uncertainty": True}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--timesteps", type=int)
    ap.add_argument("--seeds", type=int, nargs="+")
    ap.add_argument("--variants", nargs="+", choices=list(VARIANTS))
    args = ap.parse_args()
    cfg = load_config()
    if args.timesteps:
        cfg["rl"]["total_timesteps"] = args.timesteps
    seeds = args.seeds or cfg["rl"]["seeds"]
    variants = args.variants or list(VARIANTS)
    ctx = load_context(cfg)
    for seed in seeds:
        for name in variants:
            meta = train_ppo(ctx, VARIANTS[name], seed, MODELS_DIR / "rl" / f"{name}_seed{seed}")
            print({**meta, "variant": name}, flush=True)
    # summary covers every completed run on disk (runs may be trained in separate invocations)
    metas = []
    for meta_file in sorted((MODELS_DIR / "rl").glob("*/meta.json")):
        meta = json.loads(meta_file.read_text())
        meta["variant"] = meta_file.parent.name.rsplit("_seed", 1)[0]
        metas.append(meta)
    save_json({"rl_config": cfg["rl"], "runs": metas}, MODELS_DIR / "rl" / "training_summary.json")
