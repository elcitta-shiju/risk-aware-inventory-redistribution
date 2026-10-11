"""Reusable matplotlib figures (saved to results/figures)."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BLUE, ORANGE, GREY, RED, GREEN = "#2a6f97", "#e07a1f", "#8d99ae", "#c0392b", "#2e8b57"


def _save(fig, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def forecast_vs_actual(fc: pd.DataFrame, path: Path, title: str, lo: str = "lo", hi: str = "hi") -> Path:
    """Aggregate (sum over series) actual vs forecast with summed interval bounds (indicative only)."""
    agg = fc.groupby("week_start")[["y", "yhat", lo, hi]].sum()
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.fill_between(agg.index, agg[lo], agg[hi], color=BLUE, alpha=0.15, label="sum of interval bounds")
    ax.plot(agg.index, agg.y, color="black", lw=1.5, label="actual")
    ax.plot(agg.index, agg.yhat, color=BLUE, lw=1.5, label="forecast")
    for w, lab in _split_marks(fc):
        ax.axvline(w, color=GREY, ls="--", lw=1)
        ax.text(w, ax.get_ylim()[1] * 0.95, lab, color=GREY, fontsize=8)
    ax.set(title=title, xlabel="Week", ylabel="Units")
    ax.legend(loc="upper left", fontsize=8)
    return _save(fig, path)


def _split_marks(df: pd.DataFrame):
    if "split" not in df:
        return []
    return [(df.loc[df.split == s, "week_start"].min(), s) for s in ("val", "test") if (df.split == s).any()]


def series_panel(fc: pd.DataFrame, series: list[tuple[str, str]], path: Path, lo="lo", hi="hi") -> Path:
    n = len(series)
    fig, axes = plt.subplots(n, 1, figsize=(10, 2.3 * n), sharex=True)
    for ax, (store, grp) in zip(np.atleast_1d(axes), series):
        s = fc[(fc.store == store) & (fc.group == grp)].sort_values("week_start")
        ax.fill_between(s.week_start, s[lo], s[hi], step="mid", color=BLUE, alpha=0.2)
        ax.step(s.week_start, s.y, where="mid", color="black", lw=1.2, label="actual")
        ax.plot(s.week_start, s.yhat, color=BLUE, lw=1.2, label="forecast")
        ax.set_title(f"{store} / {grp}", fontsize=9)
    np.atleast_1d(axes)[0].legend(fontsize=8)
    return _save(fig, path)


def bar_compare(df: pd.DataFrame, value: str, label: str, path: Path, title: str, err: str | None = None) -> Path:
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(df[label], df[value], yerr=df[err] if err else None, color=BLUE, capsize=4)
    ax.set(title=title, ylabel=value)
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
    return _save(fig, path)


def coverage_by_group(cov: pd.DataFrame, nominal: float, path: Path, title: str) -> Path:
    fig, ax = plt.subplots(figsize=(10, 4))
    cov = cov.sort_values("coverage")
    ax.bar(range(len(cov)), cov.coverage, color=BLUE)
    ax.axhline(nominal, color=RED, ls="--", label=f"nominal {nominal:.0%}")
    ax.set_xticks(range(len(cov)), cov.index, rotation=90, fontsize=6)
    ax.set(title=title, ylabel="Empirical coverage", ylim=(0, 1.05))
    ax.legend()
    return _save(fig, path)


def line_curves(df: pd.DataFrame, x: str, ys: list[str], path: Path, title: str, ylabel: str) -> Path:
    fig, ax = plt.subplots(figsize=(9, 4))
    for y in ys:
        ax.plot(df[x], df[y], label=y)
    ax.set(title=title, xlabel=x, ylabel=ylabel)
    ax.legend(fontsize=8)
    return _save(fig, path)


def sensitivity_panels(res: pd.DataFrame, base: dict, policies: list[str], path: Path) -> Path:
    """One panel per cost parameter: mean saving vs parameter value (log x), error bars = std across
    training seeds where available (RL)."""
    params = [p for p in res.parameter.unique() if p != "base"]
    ncols = 2
    nrows = -(-len(params) // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(11, 4 * nrows), squeeze=False)
    colors = [GREY, BLUE, GREEN, ORANGE, RED]
    for ax, param in zip(axes.ravel(), params):
        sub = res[(res.parameter == param) | (res.parameter == "base")].copy()
        sub["value"] = sub.value.fillna(float(base[param]))
        for pol, c in zip(policies, colors):
            d = sub[sub.policy == pol].sort_values("value")
            if d.empty:
                continue
            ax.errorbar(d.value, d.saving_mean, yerr=d.saving_std_train_seeds.fillna(0), color=c, marker="o",
                        ms=3, capsize=3, label=pol)
        ax.axvline(float(base[param]), color=GREY, ls=":", lw=1)
        ax.axhline(0, color="black", lw=0.6)
        ax.set_xscale("log")
        ticks = sorted(sub.value.unique())
        ax.set_xticks(ticks, [f"{t:g}" for t in ticks])
        ax.minorticks_off()
        ax.set(title=param, xlabel="value (dotted = base)", ylabel="saving vs no redistribution")
    for ax in axes.ravel()[len(params):]:
        ax.set_visible(False)
    axes[0][0].legend(fontsize=7)
    return _save(fig, path)
