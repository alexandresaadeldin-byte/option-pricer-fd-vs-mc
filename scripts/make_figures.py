"""Regenerate the figures and the results table of the README: python scripts/make_figures.py"""

from dataclasses import replace
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np

from pricing import (AmericanOption, BlackScholes, EuropeanOption, MCConfig, PDEConfig,
                     price_pde, reference_price)
from pricing.convergence import exercise_dates_study, fit_slope, mc_convergence, pde_convergence
from pricing.mc.engine import VR_TECHNIQUES

OUT = Path(__file__).resolve().parents[1] / "docs" / "figures"
OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"figure.dpi": 130, "axes.grid": True, "grid.alpha": 0.3, "font.size": 10})

EURO_MODEL = BlackScholes(s0=100, r=0.05, sigma=0.2)
EURO_CALL = EuropeanOption(100, 1.0, "call")
AMER_MODEL = BlackScholes(s0=36, r=0.06, sigma=0.2)
AMER_PUT = AmericanOption(40, 1.0, "put")
lines = []


def save(fig, name):
    fig.tight_layout()
    fig.savefig(OUT / name)
    plt.close(fig)


def variance_reduction():
    n_values = np.unique(np.logspace(3, 6, 10).astype(int))
    fig, ax = plt.subplots(figsize=(7, 4.5))
    table = []
    base = None
    for technique in VR_TECHNIQUES:
        df = mc_convergence(EURO_MODEL, EURO_CALL, MCConfig(variance_reduction=technique), n_values, repeats=3)
        ax.loglog(df["n_paths"], df["ci_half_width"], marker="o", ms=3,
                  label=f"{technique} (slope {fit_slope(df['n_paths'], df['ci_half_width']):.2f})")
        last = df.iloc[-1]
        efficiency = last["stderr"] ** 2 * last["elapsed"]
        base = efficiency if technique == "none" else base
        table.append((technique, last["stderr"], 1e3 * last["elapsed"], base / efficiency))
    ax.set(xlabel="Number of payoff evaluations N", ylabel="95% CI half-width",
           title="European call: Monte Carlo variance reduction")
    ax.legend(fontsize=8)
    save(fig, "mc_variance_reduction.png")
    crude_se = table[0][1]
    lines.append("| Technique | Std error at N = 10⁶ | Variance reduction | Time (ms) | Efficiency gain |")
    lines.append("|---|---|---|---|---|")
    for technique, se, ms, gain in table:
        lines.append(f"| {technique} | {se:.5f} | ×{(crude_se / se) ** 2:.1f} | {ms:.0f} | ×{gain:.1f} |")


def rannacher():
    option = EURO_CALL
    exact = reference_price(EURO_MODEL, option)
    n_values = [100, 200, 400, 800, 1600]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for steps, label in [(2, "Crank–Nicolson + Rannacher"), (0, "Pure Crank–Nicolson")]:
        df = pde_convergence(EURO_MODEL, option, PDEConfig(n_space=100, n_time=12, rannacher_steps=steps),
                             n_values, repeats=1)
        axes[0].loglog(df["n_nodes"], df["gamma_error"], marker="o",
                       label=f"{label} (slope {fit_slope(df['n_nodes'], df['gamma_error']):.2f})")
        res = price_pde(EURO_MODEL, option, PDEConfig(n_space=400, n_time=20, rannacher_steps=steps))
        s = res.extra["s"][1:-1]
        mask = (s > 70) & (s < 140)
        axes[1].plot(s[mask], res.extra["gamma_profile"][mask], label=label)
    axes[0].set(xlabel="Grid nodes", ylabel="|gamma error|", title="Gamma convergence, N_t = N_x / 8")
    axes[0].legend(fontsize=8)
    from pricing.models.black_scholes import bs_gamma
    s = np.linspace(70, 140, 400)
    axes[1].plot(s, bs_gamma(s, 100, 1.0, 0.05, 0.2, 0.0), "k--", label="Exact")
    axes[1].set(xlabel="S", ylabel="Gamma", title="Gamma profile, N_x = 400, N_t = 20")
    axes[1].legend(fontsize=8)
    save(fig, "pde_rannacher.png")
    lines.append(f"\nEuropean call closed form: {exact.price:.6f}")


def cost_european():
    n_values = np.unique(np.logspace(3, 6.5, 10).astype(int))
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for technique in ["none", "control_variate", "stratification"]:
        df = mc_convergence(EURO_MODEL, EURO_CALL, MCConfig(variance_reduction=technique), n_values, repeats=3)
        ax.loglog(df["elapsed"], df["ci_half_width"], marker="o", ms=3,
                  label=f"MC {technique} (slope {fit_slope(df['elapsed'], df['ci_half_width']):.2f})")
    df = pde_convergence(EURO_MODEL, EURO_CALL, PDEConfig(n_space=50, n_time=12),
                         [50, 100, 200, 400, 800, 1600, 3200], repeats=5)
    ax.loglog(df["elapsed"], df["error"], marker="s", color="k",
              label=f"PDE Crank–Nicolson (slope {fit_slope(df['elapsed'], df['error']):.2f})")
    ax.set(xlabel="CPU time (s)", ylabel="Price error (MC: 95% CI half-width)",
           title="European call: accuracy vs CPU time")
    ax.legend(fontsize=8)
    save(fig, "cost_european.png")


def american_lcp():
    n_values = [200, 400, 800, 1600, 3200]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for solver, label in [("brennan_schwartz", "Brennan–Schwartz"), ("psor", "PSOR")]:
        df = pde_convergence(AMER_MODEL, AMER_PUT, PDEConfig(n_space=200, n_time=50, lcp_solver=solver),
                             n_values, repeats=3)
        axes[0].loglog(df["n_nodes"], df["error"], marker="o",
                       label=f"{label} (slope {fit_slope(df['n_nodes'], df['error']):.2f})")
        axes[1].loglog(df["n_nodes"], df["elapsed"], marker="o", label=label)
        lines.append(f"{label}: errors {', '.join(f'{e:.1e}' for e in df['error'])}; "
                     f"times (ms) {', '.join(f'{1e3 * t:.0f}' for t in df['elapsed'])}")
    axes[0].set(xlabel="Grid nodes", ylabel="|price error|", title="American put: convergence")
    axes[1].set(xlabel="Grid nodes", ylabel="CPU time (s)", title="American put: LCP solver cost")
    for ax in axes:
        ax.legend(fontsize=8)
        ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    save(fig, "american_lcp.png")


def american_mc():
    ref = reference_price(AMER_MODEL, AMER_PUT)
    df = exercise_dates_study(AMER_MODEL, AMER_PUT, MCConfig(n_paths=200_000), [2, 5, 10, 25, 50, 100])
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].errorbar(df["n_steps"], df["bias"], yerr=1.96 * df["stderr"], marker="o", capsize=3)
    axes[0].axhline(0, color="k", ls="--")
    axes[0].set(xscale="log", xlabel="Exercise dates", ylabel="LSM price − reference",
                title="Longstaff–Schwartz: Bermudan bias")
    res = price_pde(AMER_MODEL, AMER_PUT, PDEConfig(n_space=1600, n_time=400))
    t, s_star = res.extra["exercise_boundary_t"], res.extra["exercise_boundary_s"]
    order = np.argsort(t)
    axes[1].plot(t[order], s_star[order], label="Exercise boundary S*(t)")
    axes[1].axhline(40, color="k", ls="--", label="Strike")
    axes[1].fill_between(t[order], 0, s_star[order], alpha=0.15, label="Exercise region")
    axes[1].set(ylim=(30, 41), xlabel="t (years)", ylabel="S*", title="American put: early-exercise boundary")
    axes[1].legend(fontsize=8)
    save(fig, "american_mc_boundary.png")
    lines.append(f"\nAmerican put reference (S0=36): {ref.price:.6f}")
    for _, row in df.iterrows():
        lines.append(f"LSM {int(row.n_steps)} dates: {row.price:.4f} ± {1.96 * row.stderr:.4f} (bias {row.bias:+.4f})")


if __name__ == "__main__":
    variance_reduction()
    rannacher()
    cost_european()
    american_lcp()
    american_mc()
    (OUT / "results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Figures and results written to {OUT}")
