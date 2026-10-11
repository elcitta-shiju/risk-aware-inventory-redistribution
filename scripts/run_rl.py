"""Stage 8: train PPO agents (point-forecast vs uncertainty-aware observations) over several seeds.

Outputs: models/rl/{variant}_seed{k}/best_model.zip, final_model.zip, validation_curve.csv, meta.json
Usage:  python scripts/run_rl.py [--timesteps N] [--seeds 0 1 2]
"""

import _bootstrap  # noqa: F401

import argparse

from inventory_rl.agents.ppo import train_ppo
from inventory_rl.environment.factory import load_context
from inventory_rl.utils.config import load_config, save_json
from inventory_rl.utils.paths import MODELS_DIR

VARIANTS = {"rl_point": False, "rl_uncertainty": True}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--timesteps", type=int)
    ap.add_argument("--seeds", type=int, nargs="+")
    args = ap.parse_args()
    cfg = load_config()
    if args.timesteps:
        cfg["rl"]["total_timesteps"] = args.timesteps
    seeds = args.seeds or cfg["rl"]["seeds"]
    ctx = load_context(cfg)
    metas = []
    for seed in seeds:
        for name, use_unc in VARIANTS.items():
            meta = train_ppo(ctx, use_unc, seed, MODELS_DIR / "rl" / f"{name}_seed{seed}")
            meta["variant"] = name
            metas.append(meta)
            print(meta, flush=True)
    save_json({"rl_config": cfg["rl"], "runs": metas}, MODELS_DIR / "rl" / "training_summary.json")
