from dataclasses import replace

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from pricing import (AmericanOption, BlackScholes, EuropeanOption, MCConfig, PDEConfig,
                     price_mc, price_pde, reference_price)
from pricing.convergence import exercise_dates_study, fit_slope, mc_convergence, pde_convergence
from pricing.mc.engine import VR_TECHNIQUES
from pricing.models.black_scholes import bs_delta, bs_gamma

st.set_page_config(page_title="MC vs PDE pricer", layout="wide")

STYLES = {"European": EuropeanOption, "American": AmericanOption}


def build(model_kw, option_kw):
    kw = dict(option_kw)
    option_cls = STYLES[kw.pop("style")]
    return BlackScholes(**model_kw), option_cls(**kw)


def is_american(option_kw):
    return option_kw["style"] == "American"


# ---------------------------------------------------------------------------- computations (cached)

@st.cache_data(show_spinner=False)
def run_pricers(model_kw, option_kw, mc_kw, pde_kw):
    model, option = build(model_kw, option_kw)
    return (
        reference_price(model, option),
        price_mc(model, option, MCConfig(**mc_kw)),
        price_pde(model, option, PDEConfig(**pde_kw)),
    )


@st.cache_data(show_spinner=False)
def run_mc_study(model_kw, option_kw, mc_kw, techniques, n_max):
    model, option = build(model_kw, option_kw)
    n_values = np.unique(np.logspace(3, np.log10(n_max), 8).astype(int))
    base = MCConfig(**mc_kw)
    return {t: mc_convergence(model, option, replace(base, variance_reduction=t), n_values, repeats=3)
            for t in techniques}


def _pde_levels(base, levels):
    return [max(8, base.n_space // 2 ** (levels - 1)) * 2**i for i in range(levels)]


@st.cache_data(show_spinner=False)
def run_pde_study(model_kw, option_kw, pde_kw, levels):
    model, option = build(model_kw, option_kw)
    base = PDEConfig(**pde_kw)
    n_values = _pde_levels(base, levels)
    if is_american(option_kw):
        return {
            "Brennan-Schwartz": pde_convergence(model, option, replace(base, lcp_solver="brennan_schwartz"), n_values),
            "PSOR": pde_convergence(model, option, replace(base, lcp_solver="psor"), n_values),
        }
    return {
        "Rannacher": pde_convergence(model, option, base, n_values),
        "Pure CN": pde_convergence(model, option, replace(base, rannacher_steps=0), n_values),
    }


@st.cache_data(show_spinner=False)
def run_steps_study(model_kw, option_kw, mc_kw, steps_values):
    model, option = build(model_kw, option_kw)
    return exercise_dates_study(model, option, MCConfig(**mc_kw), steps_values)


@st.cache_data(show_spinner=False)
def run_pde_profiles(model_kw, option_kw, pde_kw):
    model, option = build(model_kw, option_kw)
    base = PDEConfig(**pde_kw)
    if is_american(option_kw):
        european = EuropeanOption(option.strike, option.maturity, option.kind)
        return {"American": price_pde(model, option, base), "European": price_pde(model, european, base)}
    return {
        "Rannacher": price_pde(model, option, base),
        "Pure CN": price_pde(model, option, replace(base, rannacher_steps=0)),
    }


# ---------------------------------------------------------------------------- sidebar

def sidebar():
    with st.sidebar:
        st.header("Product")
        option_kw = {
            "style": st.selectbox("Exercise", list(STYLES)),
            "kind": st.selectbox("Type", ["call", "put"], index=1),
            "strike": st.number_input("Strike K", value=100.0, min_value=0.01),
            "maturity": st.number_input("Maturity T (years)", value=1.0, min_value=0.01),
        }
        american = is_american(option_kw)

        st.header("Model — Black–Scholes")
        model_kw = {
            "s0": st.number_input("Spot S0", value=100.0, min_value=0.01),
            "r": st.number_input("Rate r", value=0.05, format="%.4f"),
            "q": st.number_input("Dividend yield q", value=0.0, format="%.4f"),
            "sigma": st.number_input("Volatility σ", value=0.2, min_value=0.001, format="%.4f"),
        }

        st.header("Monte Carlo")
        mc_kw = {"n_paths": int(st.number_input("Paths N", value=100_000, min_value=1_000, step=10_000))}
        if american:
            mc_kw["n_steps"] = int(st.number_input("Exercise dates (Longstaff–Schwartz)", value=50, min_value=1))
            mc_kw["basis_degree"] = int(st.number_input("Regression basis degree", value=3, min_value=1, max_value=8))
        else:
            technique = st.selectbox("Variance reduction", VR_TECHNIQUES)
            mc_kw.update({
                "variance_reduction": technique,
                "delta_method": st.selectbox("Delta estimator", ["pathwise", "likelihood_ratio"]),
                "gamma_method": st.selectbox("Gamma estimator", ["likelihood_ratio", "pathwise_lr", "pathwise"]),
            })
            with st.expander("Technique parameters"):
                auto_drift = st.checkbox("Automatic IS drift (μ = −d₂)", value=True)
                drift = st.number_input("IS drift μ", value=0.0)
                mc_kw["is_drift"] = None if auto_drift else drift
                mc_kw["n_strata"] = int(st.number_input("Strata L", value=100, min_value=1))
                mc_kw["conditioning_time"] = st.slider("Conditioning time θ", 0.05, 0.95, 0.5)
        mc_kw["seed"] = int(st.number_input("Seed", value=42, step=1))

        st.header("Finite differences")
        pde_kw = {
            "n_space": int(st.number_input("Space intervals N_x", value=400, min_value=8)),
            "n_time": int(st.number_input("Time steps N_t", value=50, min_value=1)),
            "width": st.number_input("Domain half-width (std devs)", value=5.0, min_value=1.0),
            "rannacher_steps": int(st.number_input("Rannacher steps", value=2, min_value=0)),
            "theta": st.slider("θ (0.5 = Crank–Nicolson)", 0.0, 1.0, 0.5),
        }
        if american:
            pde_kw["lcp_solver"] = st.selectbox("LCP solver", ["brennan_schwartz", "psor"])
            pde_kw["omega"] = st.slider("PSOR relaxation ω", 1.0, 1.95, 1.6)

        st.header("Convergence studies")
        study = {"n_max": st.select_slider("MC: largest N", options=[10**4, 10**5, 10**6], value=10**5),
                 "levels": st.slider("PDE: refinement levels", 3, 7, 5)}
        if american:
            study["techniques"] = ("none",)
        else:
            study["techniques"] = tuple(st.multiselect("MC techniques to compare", VR_TECHNIQUES,
                                                       default=list(dict.fromkeys(["none", technique]))))
        solve = st.button("SOLVE", type="primary", width="stretch")
    return model_kw, option_kw, mc_kw, pde_kw, study, solve


# ---------------------------------------------------------------------------- views

def loglog(title, x_title, y_title):
    fig = go.Figure()
    fig.update_layout(title=title, xaxis_title=x_title, yaxis_title=y_title,
                      xaxis_type="log", yaxis_type="log", height=450)
    return fig


def summary_table(reference, mc, pde, american):
    label = "Reference (PDE 8000×4000)" if american else "Closed form"
    rows = []
    for name, res in [(label, reference), ("Monte Carlo", mc), ("Finite differences", pde)]:
        rows.append({
            "Method": name,
            "Price": res.price,
            "Price error": abs(res.price - reference.price),
            "95% CI ±": 1.96 * res.price_stderr if res.price_stderr else None,
            "Delta": res.delta,
            "Delta error": abs(res.delta - reference.delta),
            "Gamma": res.gamma,
            "Gamma error": abs(res.gamma - reference.gamma),
            "Time (ms)": 1e3 * res.elapsed,
        })
    st.dataframe(pd.DataFrame(rows).set_index("Method"), width="stretch")
    if american:
        st.caption("No closed form for American options: the reference is a Brennan–Schwartz solution on an "
                   "8000 × 4000 grid (error ~1e-5). Longstaff–Schwartz gives no gamma.")


def mc_tab(studies):
    fig = loglog("Monte Carlo: CI half-width (lines) and actual error (markers)", "N (paths)", "Price error")
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


def steps_tab(df):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["n_steps"], y=df["bias"], mode="lines+markers", name="LSM price − reference",
                             error_y={"type": "data", "array": 1.96 * df["stderr"]}))
    fig.add_hline(y=0, line_dash="dash")
    fig.update_layout(title="Longstaff–Schwartz: bias vs number of exercise dates", xaxis_title="Exercise dates",
                      yaxis_title="Price − reference", xaxis_type="log", height=450)
    st.plotly_chart(fig, width="stretch")
    st.caption("With few exercise dates the option is Bermudan, worth less than the American: the estimator is "
               "biased downwards. The regressed exercise rule is also sub-optimal, another downward bias.")


def pde_tab(studies, american):
    cols = st.columns(3)
    for col, (quantity, label) in zip(cols, [("error", "Price"), ("delta_error", "Delta"), ("gamma_error", "Gamma")]):
        fig = loglog(f"{label} error", "Grid nodes", "Absolute error")
        for scheme, df in studies.items():
            slope = fit_slope(df["n_nodes"], df[quantity])
            fig.add_trace(go.Scatter(x=df["n_nodes"], y=df[quantity], mode="lines+markers",
                                     name=f"{scheme} (slope {slope:.2f})"))
        col.plotly_chart(fig, width="stretch")
    if american:
        fig = loglog("LCP solver cost", "Grid nodes", "CPU time (s)")
        for scheme, df in studies.items():
            fig.add_trace(go.Scatter(x=df["n_nodes"], y=df["elapsed"], mode="lines+markers", name=scheme))
        st.plotly_chart(fig, width="stretch")
        st.caption("Both solvers give the same solution; Brennan–Schwartz is direct (O(N) per step), PSOR iterates "
                   "and needs more sweeps as the grid is refined.")
    else:
        st.caption("Space and time are refined together, keeping N_t / N_x fixed. Theory: slope −2 for "
                   "Crank–Nicolson.")


def cost_tab(mc_studies, pde_studies, american):
    fig = loglog("Accuracy vs CPU time", "CPU time (s)", "Price error")
    for technique, df in mc_studies.items():
        fig.add_trace(go.Scatter(x=df["elapsed"], y=df["ci_half_width"], mode="lines+markers",
                                 name=f"MC {technique} (CI half-width)"))
    name = "Brennan-Schwartz" if american else "Rannacher"
    df = pde_studies[name]
    fig.add_trace(go.Scatter(x=df["elapsed"], y=df["error"], mode="lines+markers", name=f"PDE ({name})"))
    st.plotly_chart(fig, width="stretch")
    st.caption("Theory: MC error ∝ time^(−1/2), 1D Crank–Nicolson error ∝ time^(−1).")


def boundary_tab(pde_result, option_kw):
    t = pde_result.extra["exercise_boundary_t"]
    s_star = pde_result.extra["exercise_boundary_s"]
    order = np.argsort(t)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=t[order], y=s_star[order], mode="lines", name="Exercise boundary S*(t)"))
    fig.add_hline(y=option_kw["strike"], line_dash="dash", annotation_text="Strike")
    fig.update_layout(title="Early-exercise boundary (finite differences)", xaxis_title="t (years)",
                      yaxis_title="S*", height=450)
    st.plotly_chart(fig, width="stretch")
    region = "below" if option_kw["kind"] == "put" else "above"
    st.caption(f"Exercising is optimal when the spot is {region} the boundary. It tends to the strike at maturity.")


def profile_tab(profiles, model_kw, option_kw):
    k = option_kw["strike"]
    american = is_american(option_kw)
    cols = st.columns(2)
    for col, (key, label) in zip(cols, [("delta_profile", "Delta"), ("gamma_profile", "Gamma")]):
        fig = go.Figure()
        fig.update_layout(title=f"{label}(S) on the grid", xaxis_title="S", height=450)
        for scheme, res in profiles.items():
            s = res.extra["s"][1:-1]
            mask = (s > 0.6 * k) & (s < 1.5 * k)
            fig.add_trace(go.Scatter(x=s[mask], y=res.extra[key][mask], mode="lines", name=scheme))
        if not american:
            s = next(iter(profiles.values())).extra["s"][1:-1]
            mask = (s > 0.6 * k) & (s < 1.5 * k)
            args = (s[mask], k, option_kw["maturity"], model_kw["r"], model_kw["sigma"], model_kw["q"])
            exact = bs_delta(*args, option_kw["kind"] == "call") if key == "delta_profile" else bs_gamma(*args)
            fig.add_trace(go.Scatter(x=s[mask], y=exact, mode="lines", name="Exact", line={"dash": "dash"}))
        col.plotly_chart(fig, width="stretch")
    if american:
        st.caption("In the exercise region the American price equals the payoff: delta = −1 for a put and gamma = 0. "
                   "Gamma jumps at the exercise boundary.")
    else:
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
    american = is_american(option_kw)
    try:
        with st.spinner("Pricing..."):
            reference, mc, pde = run_pricers(model_kw, option_kw, mc_kw, pde_kw)
            mc_studies = run_mc_study(model_kw, option_kw, mc_kw, study["techniques"], study["n_max"])
            pde_studies = run_pde_study(model_kw, option_kw, pde_kw, study["levels"])
            profiles = run_pde_profiles(model_kw, option_kw, pde_kw)
            steps = run_steps_study(model_kw, option_kw, mc_kw, (2, 5, 10, 25, 50, 100)) if american else None
    except ValueError as err:
        st.error(str(err))
        return

    summary_table(reference, mc, pde, american)
    if american:
        tabs = st.tabs(["MC convergence", "Exercise dates", "PDE convergence", "MC vs PDE",
                        "Exercise boundary", "PDE Greek profiles"])
        with tabs[0]:
            mc_tab(mc_studies)
        with tabs[1]:
            steps_tab(steps)
        with tabs[2]:
            pde_tab(pde_studies, american)
        with tabs[3]:
            cost_tab(mc_studies, pde_studies, american)
        with tabs[4]:
            boundary_tab(pde, option_kw)
        with tabs[5]:
            profile_tab(profiles, model_kw, option_kw)
    else:
        tabs = st.tabs(["MC convergence", "PDE convergence", "MC vs PDE", "PDE Greek profiles"])
        with tabs[0]:
            mc_tab(mc_studies)
        with tabs[1]:
            pde_tab(pde_studies, american)
        with tabs[2]:
            cost_tab(mc_studies, pde_studies, american)
        with tabs[3]:
            profile_tab(profiles, model_kw, option_kw)


main()
