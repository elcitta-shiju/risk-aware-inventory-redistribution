"""Stages 9-10: evaluate every policy on the held-out TEST split under identical scenarios.

Experiment matrix:
  1. No redistribution
  2. Rule-based (point target)        3. Rule-based (upper-bound target)   4. Rule-based (CQR target)
  5. RL with point forecasts only     6. RL with point forecasts + uncertainty
RL variants: the validation-selected checkpoint (best_model) of each seed is evaluated; results
are reported as mean +/- std across training seeds and demand scenarios.

Outputs: results/metrics/evaluation_episodes.csv, results/metrics/evaluation_summary.csv,
         results/metrics/results.json, results/reports/evaluation_report.md, figures.
"""

import _bootstrap  # noqa: F401

import json

import numpy as np
import pandas as pd
from scipy import stats

from inventory_rl.agents.ppo import load_policy
from inventory_rl.baselines.rules import RuleBasedPolicy, no_transfer
from inventory_rl.environment.factory import load_context
from inventory_rl.evaluation.experiment import evaluate_policy, summarize
from inventory_rl.evaluation.report import write_report
from inventory_rl.utils.config import load_config, save_json
from inventory_rl.utils.paths import FIGURES_DIR, METRICS_DIR, MODELS_DIR, ensure_dirs
from inventory_rl.visualization import plots

RL_VARIANTS = {"rl_point": ("RL (point forecast)", False), "rl_uncertainty": ("RL (forecast + uncertainty)", True)}


def main() -> None:
    ensure_dirs()
    cfg = load_config()
    ctx = load_context(cfg)
    n_scen = cfg["environment"]["eval_poisson_scenarios"]
    sims = ctx.sims("test")
    frames = []
    for name, pol in [("No redistribution", no_transfer), ("Rule-based (point)", RuleBasedPolicy("point")),
                      ("Rule-based (upper bound)", RuleBasedPolicy("upper")),
                      ("Rule-based (CQR bound)", RuleBasedPolicy("cqr"))]:
        frames.append(evaluate_policy(sims, pol, n_scen).assign(policy=name, train_seed=-1))
        print("evaluated", name, flush=True)

    rl_dir = MODELS_DIR / "rl"
    missing = []
    for key, (label, use_unc) in RL_VARIANTS.items():
        for seed in cfg["rl"]["seeds"]:
            path = rl_dir / f"{key}_seed{seed}" / "best_model.zip"
            if not path.exists():
                missing.append(str(path))
                continue
            pol = load_policy(path, use_unc, cfg["environment"], label)
            frames.append(evaluate_policy(sims, pol, n_scen).assign(policy=label, train_seed=seed))
            print("evaluated", label, "seed", seed, flush=True)
    if missing:
        print("WARNING: missing RL checkpoints (run scripts/run_rl.py):", missing)

    ep = pd.concat(frames, ignore_index=True)
    ep.to_csv(METRICS_DIR / "evaluation_episodes.csv", index=False)
    # one "seed" unit = (training seed, scenario seed): variability covers both sources
    ep["seed_unit"] = ep.train_seed.astype(str) + "_" + ep.seed.astype(str)
    summary = summarize(ep.drop(columns="seed").rename(columns={"seed_unit": "seed"}), ["policy"])
    order = ["No redistribution", "Rule-based (point)", "Rule-based (upper bound)", "Rule-based (CQR bound)",
             "RL (point forecast)", "RL (forecast + uncertainty)"]
    summary["policy"] = pd.Categorical(summary.policy, [o for o in order if o in set(summary.policy)], ordered=True)
    summary = summary.sort_values(["scenario", "policy"])
    summary.to_csv(METRICS_DIR / "evaluation_summary.csv", index=False)

    # per-training-seed RL results (robustness across seeds)
    rl = ep[ep.train_seed >= 0]
    per_seed = (rl.groupby(["policy", "train_seed", "scenario", "seed"]).saving_vs_noop.sum()
                .groupby(["policy", "train_seed", "scenario"]).mean().unstack("scenario").reset_index())
    per_seed.to_csv(METRICS_DIR / "rl_saving_per_training_seed.csv", index=False)
    seed_stats = rl_seed_statistics(per_seed)
    seed_stats.to_csv(METRICS_DIR / "rl_saving_across_training_seeds.csv", index=False)

    # paired comparison: RL-uncertainty vs RL-point and vs best rule, same scenarios
    paired = paired_differences(ep)
    fc_metrics = json.loads((METRICS_DIR / "forecast_metrics.json").read_text())
    int_metrics = json.loads((METRICS_DIR / "interval_metrics.json").read_text())
    results = {
        "split": "test", "n_poisson_scenarios": n_scen, "products": len(ctx.tensors["test"]),
        "stores": len(ctx.network.stores), "network_provenance": ctx.network.provenance,
        "summary": summary.assign(policy=summary.policy.astype(str)).to_dict("records"),
        "paired_differences": paired, "missing_rl_checkpoints": missing,
        "rl_across_training_seeds": seed_stats.to_dict("records"),
        "rl_seed_level_comparison": seed_level_comparison(per_seed),
        "forecast_test": fc_metrics["test"], "intervals_test": int_metrics["by_split"]["test"],
        "interval_method": int_metrics["chosen"],
    }
    save_json(results, METRICS_DIR / "results.json")

    for scen in ("poisson", "replay"):
        s = summary[summary.scenario == scen].assign(policy=lambda d: d.policy.astype(str))
        plots.bar_compare(s, "saving_vs_noop", "policy", FIGURES_DIR / f"eval_saving_{scen}.png",
                          f"Test: modeled cost saving vs no redistribution ({scen} demand)",
                          err="saving_vs_noop_std" if scen == "poisson" else None)
        plots.bar_compare(s, "fill_rate", "policy", FIGURES_DIR / f"eval_fill_rate_{scen}.png",
                          f"Test: fill rate ({scen} demand)")
    curves = []
    for key, (label, _) in RL_VARIANTS.items():
        for seed in cfg["rl"]["seeds"]:
            f = rl_dir / f"{key}_seed{seed}" / "validation_curve.csv"
            if f.exists():
                curves.append(pd.read_csv(f).assign(run=f"{label} s{seed}"))
    if curves:
        c = pd.concat(curves).pivot_table(index="timesteps", columns="run", values="val_saving_poisson").reset_index()
        plots.line_curves(c, "timesteps", [x for x in c.columns if x != "timesteps"],
                          FIGURES_DIR / "rl_validation_curves.png",
                          "PPO validation saving vs no-transfer (Poisson scenarios)", "saving")
    write_report(results, summary, per_seed)
    print(summary[["policy", "scenario", "fill_rate", "lost_units", "transfer_cost", "total_cost",
                   "saving_vs_noop", "saving_vs_noop_std", "units_transferred"]].round(2).to_string(index=False))


def rl_seed_statistics(per_seed: pd.DataFrame) -> pd.DataFrame:
    """Mean +/- std (ddof=1) across training seeds of each seed's mean test saving."""
    long = per_seed.melt(id_vars=["policy", "train_seed"], value_vars=["poisson", "replay"],
                         var_name="scenario", value_name="saving")
    g = long.groupby(["policy", "scenario"]).saving
    return pd.DataFrame({"mean": g.mean(), "std": g.std(ddof=1), "min": g.min(), "max": g.max(),
                         "n_train_seeds": g.size()}).reset_index()


def seed_level_comparison(per_seed: pd.DataFrame, a: str = "RL (forecast + uncertainty)",
                          b: str = "RL (point forecast)") -> dict:
    """Welch t-test on per-training-seed mean savings (seeds are the replicates, not scenarios)."""
    out = {}
    for scen in ("poisson", "replay"):
        xa = per_seed.loc[per_seed.policy == a, scen].to_numpy()
        xb = per_seed.loc[per_seed.policy == b, scen].to_numpy()
        if len(xa) < 2 or len(xb) < 2:
            continue
        t = stats.ttest_ind(xa, xb, equal_var=False)
        out[f"{a} minus {b} | {scen}"] = {"mean_diff": float(xa.mean() - xb.mean()), "welch_t": float(t.statistic),
                                          "p_value": float(t.pvalue), "n_seeds_a": int(len(xa)),
                                          "n_seeds_b": int(len(xb))}
    return out


def paired_differences(ep: pd.DataFrame) -> dict:
    """Mean paired difference in total saving (same scenario seed), with a 95% normal CI over scenarios."""
    tot = ep.groupby(["policy", "train_seed", "scenario", "seed"]).saving_vs_noop.sum().reset_index()
    avg = tot.groupby(["policy", "scenario", "seed"]).saving_vs_noop.mean().unstack("policy")
    out = {}
    pairs = [("RL (forecast + uncertainty)", "RL (point forecast)"),
             ("RL (forecast + uncertainty)", "Rule-based (upper bound)"),
             ("Rule-based (upper bound)", "Rule-based (point)")]
    for a, b in pairs:
        if a not in avg or b not in avg:
            continue
        for scen in ("poisson", "replay"):
            d = (avg.loc[scen, a] - avg.loc[scen, b]).dropna()
            se = d.std(ddof=1) / np.sqrt(len(d)) if len(d) > 1 else np.nan
            out[f"{a} minus {b} | {scen}"] = {"mean": float(d.mean()), "ci95_low": float(d.mean() - 1.96 * se),
                                              "ci95_high": float(d.mean() + 1.96 * se), "n_scenarios": int(len(d))}
    return out


if __name__ == "__main__":
    main()
