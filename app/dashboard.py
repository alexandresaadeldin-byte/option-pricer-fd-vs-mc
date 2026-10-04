from dataclasses import replace

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from pricing import BlackScholes, EuropeanOption, MCConfig, PDEConfig, price_mc, price_pde
from pricing.convergence import fit_slope, mc_convergence, pde_convergence
from pricing.mc.engine import VR_TECHNIQUES
from pricing.models.black_scholes import bs_delta, bs_gamma

st.set_page_config(page_title="MC vs PDE pricer", layout="wide")


@st.cache_data(show_spinner=False)
def run_pricers(model_kw, option_kw, mc_kw, pde_kw):
    model, option = BlackScholes(**model_kw), EuropeanOption(**option_kw)
    return (
        model.closed_form(option),
        price_mc(model, option, MCConfig(**mc_kw)),
        price_pde(model, option, PDEConfig(**pde_kw)),
    )


@st.cache_data(show_spinner=False)
def run_mc_study(model_kw, option_kw, mc_kw, techniques, n_max):
    model, option = BlackScholes(**model_kw), EuropeanOption(**option_kw)
    n_values = np.unique(np.logspace(3, np.log10(n_max), 8).astype(int))
    base = MCConfig(**mc_kw)
    return {t: mc_convergence(model, option, replace(base, variance_reduction=t), n_values, repeats=1)
            for t in techniques}


@st.cache_data(show_spinner=False)
def run_pde_study(model_kw, option_kw, pde_kw, levels):
    model, option = BlackScholes(**model_kw), EuropeanOption(**option_kw)
    base = PDEConfig(**pde_kw)
    n_values = [max(8, base.n_space // 2 ** (levels - 1)) * 2**i for i in range(levels)]
    no_rannacher = replace(base, rannacher_steps=0)
    return {
        "Rannacher": pde_convergence(model, option, base, n_values),
        "Pure CN": pde_convergence(model, option, no_rannacher, n_values),
    }


@st.cache_data(show_spinner=False)
def run_pde_profiles(model_kw, option_kw, pde_kw):
    model, option = BlackScholes(**model_kw), EuropeanOption(**option_kw)
    base = PDEConfig(**pde_kw)
    return {
        "Rannacher": price_pde(model, option, base),
        "Pure CN": price_pde(model, option, replace(base, rannacher_steps=0)),
    }


def loglog(title, x_title, y_title):
    fig = go.Figure()
    fig.update_layout(title=title, xaxis_title=x_title, yaxis_title=y_title,
                      xaxis_type="log", yaxis_type="log", height=450)
    return fig


def sidebar():
    with st.sidebar:
        st.header("Model — Black–Scholes")
        model_kw = {
            "s0": st.number_input("Spot S0", value=100.0, min_value=0.01),
            "r": st.number_input("Rate r", value=0.05, format="%.4f"),
            "q": st.number_input("Dividend yield q", value=0.0, format="%.4f"),
            "sigma": st.number_input("Volatility σ", value=0.2, min_value=0.001, format="%.4f"),
        }
        st.header("Product — European")
        option_kw = {
            "kind": st.selectbox("Type", ["call", "put"]),
            "strike": st.number_input("Strike K", value=100.0, min_value=0.01),
            "maturity": st.number_input("Maturity T (years)", value=1.0, min_value=0.01),
        }
        st.header("Monte Carlo")
        technique = st.selectbox("Variance reduction", VR_TECHNIQUES)
        mc_kw = {
            "n_paths": int(st.number_input("Paths N", value=100_000, min_value=1_000, step=10_000)),
            "variance_reduction": technique,
            "delta_method": st.selectbox("Delta estimator", ["pathwise", "likelihood_ratio"]),
            "gamma_method": st.selectbox("Gamma estimator", ["likelihood_ratio", "pathwise_lr", "pathwise"]),
            "seed": int(st.number_input("Seed", value=42, step=1)),
        }
        with st.expander("Technique parameters"):
            auto_drift = st.checkbox("Automatic IS drift (μ = −d₂)", value=True)
            drift = st.number_input("IS drift μ", value=0.0)
            mc_kw["is_drift"] = None if auto_drift else drift
            mc_kw["n_strata"] = int(st.number_input("Strata L", value=100, min_value=1))
            mc_kw["conditioning_time"] = st.slider("Conditioning time θ", 0.05, 0.95, 0.5)
        st.header("Finite differences")
        pde_kw = {
            "n_space": int(st.number_input("Space intervals N_x", value=400, min_value=8)),
            "n_time": int(st.number_input("Time steps N_t", value=50, min_value=1)),
            "width": st.number_input("Domain half-width (std devs)", value=5.0, min_value=1.0),
            "rannacher_steps": int(st.number_input("Rannacher steps", value=2, min_value=0)),
            "theta": st.slider("θ (0.5 = Crank–Nicolson)", 0.0, 1.0, 0.5),
        }
        st.header("Convergence studies")
        study = {
            "techniques": st.multiselect("MC techniques to compare", VR_TECHNIQUES,
                                         default=list(dict.fromkeys(["none", technique]))),
            "n_max": st.select_slider("MC: largest N", options=[10**4, 10**5, 10**6], value=10**5),
            "levels": st.slider("PDE: refinement levels", 3, 7, 5),
        }
        solve = st.button("SOLVE", type="primary", width="stretch")
    return model_kw, option_kw, mc_kw, pde_kw, study, solve


def summary_table(exact, mc, pde):
    rows = []
    for name, res in [("Closed form", exact), ("Monte Carlo", mc), ("Finite differences", pde)]:
        rows.append({
            "Method": name,
            "Price": res.price,
            "Price error": abs(res.price - exact.price),
            "95% CI ±": 1.96 * res.price_stderr if res.price_stderr else None,
            "Delta": res.delta,
            "Delta error": abs(res.delta - exact.delta),
            "Gamma": res.gamma,
            "Gamma error": abs(res.gamma - exact.gamma),
            "Time (ms)": 1e3 * res.elapsed,
        })
    st.dataframe(pd.DataFrame(rows).set_index("Method"), width="stretch")


def mc_tab(studies):
    fig = loglog("Monte Carlo: CI half-width (lines) and actual error (markers)", "N (payoff evaluations)", "Price error")
    rows = []
    base_efficiency = None
    for technique, df in studies.items():
        slope = fit_slope(df["n_paths"], df["ci_half_width"])
        fig.add_trace(go.Scatter(x=df["n_paths"], y=df["ci_half_width"], mode="lines",
                                 name=f"{technique} (slope {slope:.2f})", legendgroup=technique))
        fig.add_trace(go.Scatter(x=df["n_paths"], y=df["error"], mode="markers", showlegend=False,
                                 legendgroup=technique))
        last = df.iloc[-1]
        efficiency = last["stderr"] ** 2 * last["elapsed"]
        if technique == "none":
            base_efficiency = efficiency
        rows.append({"Technique": technique, "Variance per path": last["stderr"] ** 2 * last["n_paths"],
                     "Time (ms)": 1e3 * last["elapsed"], "Variance × time": efficiency})
    st.plotly_chart(fig, width="stretch")
    table = pd.DataFrame(rows).set_index("Technique")
    if base_efficiency:
        table["Efficiency gain vs none"] = base_efficiency / table["Variance × time"]
    st.caption("Efficiency measured at the largest N. Higher gain = same accuracy for less CPU time.")
    st.dataframe(table, width="stretch")


def pde_tab(studies):
    cols = st.columns(3)
    for col, (quantity, label) in zip(cols, [("error", "Price"), ("delta_error", "Delta"), ("gamma_error", "Gamma")]):
        fig = loglog(f"{label} error", "Grid nodes", "Absolute error")
        for scheme, df in studies.items():
            slope = fit_slope(df["n_nodes"], df[quantity])
            fig.add_trace(go.Scatter(x=df["n_nodes"], y=df[quantity], mode="lines+markers",
                                     name=f"{scheme} (slope {slope:.2f})"))
        col.plotly_chart(fig, width="stretch")
    st.caption("Space and time are refined together, keeping N_t / N_x fixed. Theory: slope −2 for Crank–Nicolson.")


def cost_tab(mc_studies, pde_studies):
    fig = loglog("Accuracy vs CPU time", "CPU time (s)", "Price error")
    for technique, df in mc_studies.items():
        fig.add_trace(go.Scatter(x=df["elapsed"], y=df["ci_half_width"], mode="lines+markers",
                                 name=f"MC {technique} (CI half-width)"))
    df = pde_studies["Rannacher"]
    fig.add_trace(go.Scatter(x=df["elapsed"], y=df["error"], mode="lines+markers",
                             name="PDE Crank–Nicolson + Rannacher"))
    st.plotly_chart(fig, width="stretch")
    st.caption("Theory: MC error ∝ time^(−1/2), 1D Crank–Nicolson error ∝ time^(−1).")


def profile_tab(profiles, model_kw, option_kw):
    k = option_kw["strike"]
    cols = st.columns(2)
    for col, (key, exact_fn, label) in zip(cols, [
        ("delta_profile", lambda s: bs_delta(s, k, option_kw["maturity"], model_kw["r"], model_kw["sigma"],
                                             model_kw["q"], option_kw["kind"] == "call"), "Delta"),
        ("gamma_profile", lambda s: bs_gamma(s, k, option_kw["maturity"], model_kw["r"], model_kw["sigma"],
                                             model_kw["q"]), "Gamma"),
    ]):
        fig = go.Figure()
        fig.update_layout(title=f"{label}(S) on the grid", xaxis_title="S", height=450)
        for scheme, res in profiles.items():
            s = res.extra["s"][1:-1]
            mask = (s > 0.6 * k) & (s < 1.5 * k)
            fig.add_trace(go.Scatter(x=s[mask], y=res.extra[key][mask], mode="lines", name=scheme))
        s = profiles["Rannacher"].extra["s"][1:-1]
        mask = (s > 0.6 * k) & (s < 1.5 * k)
        fig.add_trace(go.Scatter(x=s[mask], y=exact_fn(s[mask]), mode="lines", name="Exact",
                                 line={"dash": "dash"}))
        col.plotly_chart(fig, width="stretch")
    st.caption("Pure Crank–Nicolson does not damp the payoff kink when N_t is small relative to N_x: "
               "gamma oscillates around the strike. Rannacher start-up removes the oscillations.")


def main():
    st.title("Option pricer — Monte Carlo vs finite differences")
    model_kw, option_kw, mc_kw, pde_kw, study, solve = sidebar()
    if solve:
        st.session_state["params"] = (model_kw, option_kw, mc_kw, pde_kw, study)
    if "params" not in st.session_state:
        st.info("Set the parameters in the sidebar, then click **SOLVE**.")
        return
    model_kw, option_kw, mc_kw, pde_kw, study = st.session_state["params"]
    try:
        with st.spinner("Pricing..."):
            exact, mc, pde = run_pricers(model_kw, option_kw, mc_kw, pde_kw)
            mc_studies = run_mc_study(model_kw, option_kw, mc_kw, tuple(study["techniques"]), study["n_max"])
            pde_studies = run_pde_study(model_kw, option_kw, pde_kw, study["levels"])
            profiles = run_pde_profiles(model_kw, option_kw, pde_kw)
    except ValueError as err:
        st.error(str(err))
        return

    summary_table(exact, mc, pde)
    tabs = st.tabs(["MC convergence", "PDE convergence", "MC vs PDE", "PDE Greek profiles"])
    with tabs[0]:
        mc_tab(mc_studies)
    with tabs[1]:
        pde_tab(pde_studies)
    with tabs[2]:
        cost_tab(mc_studies, pde_studies)
    with tabs[3]:
        profile_tab(profiles, model_kw, option_kw)


main()
