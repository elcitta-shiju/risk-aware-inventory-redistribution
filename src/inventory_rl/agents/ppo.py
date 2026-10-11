"""Stage 8: PPO training (Stable-Baselines3) with validation-based checkpoint selection.

PPO supports continuous Box action spaces natively, which matches the per-store action design.
Training uses ONLY the training-split simulator (Poisson demand scenarios); checkpoints are
selected on the validation split; the test split is touched only by the final evaluation.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from inventory_rl.environment.factory import SimContext
from inventory_rl.agents.policy import PerStorePolicy
from inventory_rl.environment.gym_env import RedistributionEnv, action_to_transfers, build_obs, per_store_features
from inventory_rl.evaluation.experiment import evaluate_policy


class RLPolicy:
    """Adapter: trained SB3 model -> simulator policy (deterministic actions)."""

    def __init__(self, model: PPO, use_uncertainty: bool, max_target_offset: float, name: str = "RL"):
        self.model, self.use_uncertainty, self.max_offset, self.name = model, use_uncertainty, max_target_offset, name

    def __call__(self, sim) -> list:
        obs = build_obs(sim, self.use_uncertainty)
        action, _ = self.model.predict(obs, deterministic=True)
        return action_to_transfers(action, sim, self.max_offset)


class ValidationCallback(BaseCallback):
    def __init__(self, ctx: SimContext, use_uncertainty: bool, eval_freq: int, n_scen: int, out_dir: Path):
        super().__init__()
        self.ctx, self.use_unc, self.eval_freq, self.n_scen, self.out_dir = ctx, use_uncertainty, eval_freq, n_scen, out_dir
        self.best, self.history = -np.inf, []
        self.max_offset = ctx.cfg["environment"]["max_target_offset"]

    def _evaluate(self) -> float:
        pol = RLPolicy(self.model, self.use_unc, self.max_offset)
        df = evaluate_policy(self.ctx.sims("val"), pol, self.n_scen, base_seed=20_000)
        per_seed = df.groupby(["scenario", "seed"]).saving_vs_noop.sum()
        score = float(per_seed.loc["poisson"].mean())
        self.history.append({"timesteps": self.num_timesteps, "val_saving_poisson": score,
                             "val_saving_replay": float(per_seed.loc["replay"].mean()),
                             "val_units_transferred": float(df.units_transferred.sum() / df.seed.nunique())})
        if score > self.best:
            self.best = score
            self.model.save(self.out_dir / "best_model")
        return score

    def _on_training_start(self) -> None:
        self._evaluate()  # untrained policy, for reference in the learning curve

    def _on_step(self) -> bool:
        if self.num_timesteps % self.eval_freq < self.training_env.num_envs:
            self._evaluate()
        return True

    def _on_training_end(self) -> None:
        if not self.history or self.history[-1]["timesteps"] != self.num_timesteps:
            self._evaluate()


def train_ppo(ctx: SimContext, use_uncertainty: bool, seed: int, out_dir: Path) -> dict:
    rl, env_cfg = ctx.cfg["rl"], ctx.cfg["environment"]
    out_dir.mkdir(parents=True, exist_ok=True)

    def make(i: int):
        def _f():
            sim = ctx.sim("train", "poisson", seed=seed * 1000 + i)
            env = RedistributionEnv(sim, use_uncertainty, env_cfg["max_target_offset"])
            env.reset(seed=seed * 1000 + i)
            return Monitor(env)
        return _f

    venv = DummyVecEnv([make(i) for i in range(rl["n_envs"])])
    policy_kwargs = {"n_stores": len(ctx.network.stores),
                     "per_store_features": per_store_features(use_uncertainty),
                     "hidden": rl["hidden"], "log_std_init": rl["log_std_init"]}
    model = PPO(PerStorePolicy, venv, learning_rate=rl["learning_rate"], n_steps=rl["n_steps"],
                batch_size=rl["batch_size"], gamma=rl["gamma"], ent_coef=rl["ent_coef"], seed=seed,
                policy_kwargs=policy_kwargs, verbose=0, device="cpu")
    cb = ValidationCallback(ctx, use_uncertainty, rl["eval_freq_steps"], 5, out_dir)
    t0 = time.time()
    model.learn(total_timesteps=rl["total_timesteps"], callback=cb, progress_bar=False)
    model.save(out_dir / "final_model")
    pd.DataFrame(cb.history).to_csv(out_dir / "validation_curve.csv", index=False)
    rewards = [r for e in venv.envs for r in e.get_episode_rewards()]
    np.savetxt(out_dir / "train_episode_rewards.csv", np.array(rewards), delimiter=",")
    meta = {"use_uncertainty": use_uncertainty, "seed": seed, "train_seconds": round(time.time() - t0, 1),
            "best_val_saving_poisson": cb.best, "timesteps": rl["total_timesteps"]}
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    return meta


def load_policy(path: Path, use_uncertainty: bool, env_cfg: dict, name: str) -> RLPolicy:
    return RLPolicy(PPO.load(path, device="cpu"), use_uncertainty, env_cfg["max_target_offset"], name)
