"""
VolSurface Pro: Options Surface & Greeks Analytics
====================================================

An institutional-grade, single-file Streamlit application for building
implied-volatility surfaces, analytical Greeks, and skew/term-structure
diagnostics from live US-listed options chains.

Run with:
    streamlit run app.py

Author: Quantitative Analytics & Automation Portfolio
"""

from __future__ import annotations

import datetime as dt
import warnings
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf
from scipy import interpolate
from scipy.optimize import brentq
from scipy.stats import norm

warnings.filterwarnings("ignore")

# ======================================================================
# 1. PAGE CONFIG & GLOBAL STYLING
# ======================================================================

st.set_page_config(
    page_title="VolSurface Pro | Options Analytics",
    page_icon=":material/monitoring:",
    layout="wide",
    initial_sidebar_state="expanded",
)

CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Roboto+Mono:wght@400;500;600&display=swap');
@import url('https://fonts.googleapis.com/css2?family=Material+Symbols+Rounded:opsz,wght,FILL,GRAD@20..48,100..700,0..1,-50..200&display=block');

.icon-glyph {
    font-family: 'Material Symbols Rounded';
    font-weight: normal;
    font-style: normal;
    line-height: 1;
    letter-spacing: normal;
    text-transform: none;
    display: inline-block;
    white-space: nowrap;
    word-wrap: normal;
    direction: ltr;
    -webkit-font-smoothing: antialiased;
}

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background-color: #0b0d13;
    color: #e6e6e6;
}

section[data-testid="stSidebar"] {
    background-color: #0f1119;
    border-right: 1px solid #1f2430;
}

section[data-testid="stSidebar"] .stMarkdown p, section[data-testid="stSidebar"] label {
    font-size: 0.86rem;
}

/* KPI cards */
div[data-testid="stMetric"] {
    background: linear-gradient(180deg, #171b26 0%, #12151e 100%);
    border: 1px solid #232838;
    border-radius: 12px;
    padding: 16px 18px 12px 18px;
    transition: border-color 0.15s ease;
}
div[data-testid="stMetric"]:hover {
    border-color: #2e3648;
}

div[data-testid="stMetricLabel"] {
    color: #7c8496 !important;
    font-size: 0.74rem !important;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    font-weight: 600 !important;
}

div[data-testid="stMetricValue"] {
    color: #00FFA3 !important;
    font-family: 'Roboto Mono', monospace;
    font-weight: 600 !important;
    font-size: 1.65rem !important;
}

div[data-testid="stMetricDelta"] {
    font-family: 'Roboto Mono', monospace;
    font-size: 0.85rem;
}

/* Tabs */
div[data-testid="stTabs"] button[data-baseweb="tab"] {
    font-family: 'Inter', sans-serif;
    font-weight: 600;
    font-size: 0.92rem;
    color: #7c8496;
    padding: 10px 18px;
}
div[data-testid="stTabs"] button[data-baseweb="tab"][aria-selected="true"] {
    color: #00D4FF !important;
    border-bottom-color: #00D4FF !important;
}
div[data-testid="stTabs"] [data-baseweb="tab-highlight"] {
    background-color: #00D4FF !important;
}
div[data-testid="stTabs"] [data-baseweb="tab-border"] {
    background-color: #1f2430 !important;
}

/* Headings */
h1 {
    color: #f4f6fb !important;
    font-weight: 700 !important;
    letter-spacing: -0.01em;
}
h2, h3 {
    color: #f4f6fb !important;
    font-weight: 600 !important;
}

p, .stCaption, div[data-testid="stCaptionContainer"] {
    color: #9aa2b4;
}

hr {
    border-color: #1f2430;
}

.app-header {
    display: flex;
    align-items: center;
    gap: 14px;
    margin-bottom: 2px;
}
.app-header .icon-glyph {
    font-size: 2.2rem;
    color: #00FFA3;
}
.app-subtitle {
    color: #7c8496;
    font-size: 1.0rem;
    font-weight: 400;
    margin-top: -4px;
}

.ticker-context {
    display: flex;
    gap: 22px;
    flex-wrap: wrap;
    color: #7c8496;
    font-size: 0.86rem;
    font-family: 'Roboto Mono', monospace;
    padding: 2px 0 4px 0;
}
.ticker-context b { color: #d7dbe6; font-weight: 600; }

.footer-badge {
    text-align: center;
    padding: 18px 0 6px 0;
    color: #4a5164;
    font-size: 0.82rem;
    border-top: 1px solid #1f2430;
    margin-top: 24px;
}
.footer-badge a {
    color: #00D4FF;
    text-decoration: none;
}

/* Buttons */
[data-testid="stSidebar"] button {
    border-radius: 8px !important;
    border: 1px solid #262b38 !important;
}
[data-testid="stSidebar"] button:hover {
    border-color: #ff2600 !important;
    color: #ff2600 !important;
}
[data-testid="stSidebar"] button[kind="primary"],
[data-testid="stSidebar"] [data-testid="stBaseButton-primary"],
[data-testid="stSidebar"] button[kind="primary"] p,
[data-testid="stSidebar"] button[kind="primary"] span,
[data-testid="stSidebar"] [data-testid="stBaseButton-primary"] p,
[data-testid="stSidebar"] [data-testid="stBaseButton-primary"] span {
    color: #ffffff !important;
}
[data-testid="stSidebar"] button[kind="primary"]:hover,
[data-testid="stSidebar"] [data-testid="stBaseButton-primary"]:hover,
[data-testid="stSidebar"] button[kind="primary"]:hover p,
[data-testid="stSidebar"] button[kind="primary"]:hover span,
[data-testid="stSidebar"] [data-testid="stBaseButton-primary"]:hover p,
[data-testid="stSidebar"] [data-testid="stBaseButton-primary"]:hover span {
    color: #ffffff !important;
}

/* Expander */
div[data-testid="stExpander"] {
    border: 1px solid #1f2430;
    border-radius: 10px;
    background-color: #0f1119;
}

/* DataFrame */
div[data-testid="stDataFrame"] {
    border: 1px solid #1f2430;
    border-radius: 10px;
}

code, .stCodeBlock {
    font-family: 'Roboto Mono', monospace;
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

PLOTLY_DARK_LAYOUT = dict(
    paper_bgcolor="#0b0d13",
    plot_bgcolor="#0b0d13",
    font=dict(family="Inter, Roboto Mono, monospace", color="#e6e6e6"),
    margin=dict(l=10, r=10, t=50, b=10),
)

NEON_GREEN = "#00FFA3"
ELECTRIC_BLUE = "#00D4FF"
WARN_RED = "#FF5C7A"


# ======================================================================
# 2. BLACK-SCHOLES-MERTON MATHEMATICAL ENGINE
# ======================================================================

class BlackScholesEngine:
    """
    Vector-optimized Black-Scholes-Merton pricing and Greeks engine.

    All methods accept scalars or numpy arrays (broadcastable) and are
    defensively guarded against T<=0, sigma<=0, and division-by-zero
    edge cases that occur at expiry or with degenerate volatility
    inputs. Where inputs are degenerate, methods fall back to the
    option's intrinsic value / limiting Greek rather than raising or
    returning NaN.

    Parameters (shared across methods)
    -----------------------------------
    S : float or ndarray
        Spot price of the underlying.
    K : float or ndarray
        Strike price.
    T : float or ndarray
        Time to expiration in years.
    r : float or ndarray
        Continuously-compounded risk-free rate (decimal, e.g. 0.045).
    sigma : float or ndarray
        Annualized volatility (decimal, e.g. 0.20 for 20%).
    """

    _EPS_T = 1e-6
    _EPS_SIGMA = 1e-6

    # ------------------------------------------------------------------
    @staticmethod
    def _safe_inputs(S, K, T, sigma):
        """Clip degenerate T / sigma to small positive epsilons.

        Time complexity: O(n) for array inputs of length n.
        """
        S = np.asarray(S, dtype=float)
        K = np.asarray(K, dtype=float)
        T = np.maximum(np.asarray(T, dtype=float), BlackScholesEngine._EPS_T)
        sigma = np.maximum(np.asarray(sigma, dtype=float), BlackScholesEngine._EPS_SIGMA)
        return S, K, T, sigma

    # ------------------------------------------------------------------
    @staticmethod
    def _d1_d2(S, K, T, r, sigma):
        """Compute d1 and d2 terms of the BSM formula.

        d1 = [ln(S/K) + (r + sigma^2/2) * T] / (sigma * sqrt(T))
        d2 = d1 - sigma * sqrt(T)

        Time complexity: O(n) for array inputs of length n.
        """
        S, K, T, sigma = BlackScholesEngine._safe_inputs(S, K, T, sigma)
        sqrtT = np.sqrt(T)
        with np.errstate(divide="ignore", invalid="ignore"):
            d1 = (np.log(S / np.maximum(K, 1e-12)) + (r + 0.5 * sigma ** 2) * T) / (sigma * sqrtT)
            d2 = d1 - sigma * sqrtT
        d1 = np.nan_to_num(d1, nan=0.0, posinf=1e6, neginf=-1e6)
        d2 = np.nan_to_num(d2, nan=0.0, posinf=1e6, neginf=-1e6)
        return d1, d2

    # ------------------------------------------------------------------
    @classmethod
    def price(cls, S, K, T, r, sigma, option_type: str = "call"):
        """
        Black-Scholes-Merton price for a European call or put.

        Call: C = S*N(d1) - K*exp(-rT)*N(d2)
        Put:  P = K*exp(-rT)*N(-d2) - S*N(-d1)

        At or below expiry (T <= 0), returns the exact intrinsic value
        rather than a formula evaluated at a clipped epsilon, so
        boundary contracts price cleanly with no NaN/ZeroDivisionError.

        Time complexity: O(n).
        """
        S_raw = np.asarray(S, dtype=float)
        K_raw = np.asarray(K, dtype=float)
        T_raw = np.asarray(T, dtype=float)
        S, K, T, sigma = cls._safe_inputs(S, K, T, sigma)
        d1, d2 = cls._d1_d2(S, K, T, r, sigma)
        disc = np.exp(-r * T)
        if option_type.lower().startswith("c"):
            val = S * norm.cdf(d1) - K * disc * norm.cdf(d2)
            intrinsic = np.maximum(S_raw - K_raw, 0.0)
        else:
            val = K * disc * norm.cdf(-d2) - S * norm.cdf(-d1)
            intrinsic = np.maximum(K_raw - S_raw, 0.0)
        val = np.maximum(val, 0.0)
        at_expiry = T_raw <= cls._EPS_T
        return np.where(at_expiry, intrinsic, val)

    # ------------------------------------------------------------------
    @classmethod
    def delta(cls, S, K, T, r, sigma, option_type: str = "call"):
        """Delta: dV/dS. Call = N(d1); Put = N(d1) - 1. O(n)."""
        S, K, T, sigma = cls._safe_inputs(S, K, T, sigma)
        d1, _ = cls._d1_d2(S, K, T, r, sigma)
        if option_type.lower().startswith("c"):
            return norm.cdf(d1)
        return norm.cdf(d1) - 1.0

    # ------------------------------------------------------------------
    @classmethod
    def gamma(cls, S, K, T, r, sigma):
        """Gamma: d^2V/dS^2, identical for calls and puts. O(n)."""
        S, K, T, sigma = cls._safe_inputs(S, K, T, sigma)
        d1, _ = cls._d1_d2(S, K, T, r, sigma)
        denom = S * sigma * np.sqrt(T)
        denom = np.where(denom == 0, 1e-12, denom)
        return norm.pdf(d1) / denom

    # ------------------------------------------------------------------
    @classmethod
    def vega(cls, S, K, T, r, sigma, per_one_pct: bool = True):
        """Vega: dV/dsigma. Divided by 100 to express per 1% vol move. O(n)."""
        S, K, T, sigma = cls._safe_inputs(S, K, T, sigma)
        d1, _ = cls._d1_d2(S, K, T, r, sigma)
        raw = S * norm.pdf(d1) * np.sqrt(T)
        return raw / 100.0 if per_one_pct else raw

    # ------------------------------------------------------------------
    @classmethod
    def theta(cls, S, K, T, r, sigma, option_type: str = "call", per_day: bool = True):
        """
        Theta: -dV/dT (time decay). Returns annualized theta by default,
        or the 1-calendar-day figure (annualized / 365) when per_day=True.
        O(n).
        """
        S, K, T, sigma = cls._safe_inputs(S, K, T, sigma)
        d1, d2 = cls._d1_d2(S, K, T, r, sigma)
        sqrtT = np.sqrt(T)
        term1 = -(S * norm.pdf(d1) * sigma) / (2 * sqrtT)
        disc = np.exp(-r * T)
        if option_type.lower().startswith("c"):
            term2 = -r * K * disc * norm.cdf(d2)
            annual_theta = term1 + term2
        else:
            term2 = r * K * disc * norm.cdf(-d2)
            annual_theta = term1 + term2
        return annual_theta / 365.0 if per_day else annual_theta

    # ------------------------------------------------------------------
    @classmethod
    def rho(cls, S, K, T, r, sigma, option_type: str = "call", per_one_pct: bool = True):
        """Rho: dV/dr, expressed per 1% change in rates by default. O(n)."""
        S, K, T, sigma = cls._safe_inputs(S, K, T, sigma)
        _, d2 = cls._d1_d2(S, K, T, r, sigma)
        disc = np.exp(-r * T)
        if option_type.lower().startswith("c"):
            raw = K * T * disc * norm.cdf(d2)
        else:
            raw = -K * T * disc * norm.cdf(-d2)
        return raw / 100.0 if per_one_pct else raw

    # ------------------------------------------------------------------
    @classmethod
    def greeks_bundle(cls, S, K, T, r, sigma, option_type: str = "call") -> dict:
        """Convenience wrapper returning all Greeks in one dict. O(n)."""
        return {
            "delta": cls.delta(S, K, T, r, sigma, option_type),
            "gamma": cls.gamma(S, K, T, r, sigma),
            "vega": cls.vega(S, K, T, r, sigma),
            "theta": cls.theta(S, K, T, r, sigma, option_type),
            "rho": cls.rho(S, K, T, r, sigma, option_type),
        }

    # ------------------------------------------------------------------
    @classmethod
    def implied_vol(
        cls,
        market_price: float,
        S: float,
        K: float,
        T: float,
        r: float,
        option_type: str = "call",
        lower: float = 0.001,
        upper: float = 5.0,
    ) -> Optional[float]:
        """
        Solve for implied volatility via Brent's method on the root:
            BS_Price(sigma) - Market_Mid_Price = 0

        Returns None if a sign change is not found in [lower, upper]
        (i.e. the quote violates no-arbitrage bounds implied by the
        bracket) or if the solver otherwise fails, so callers can
        gracefully fall back to a broker-reported IV.

        Time complexity: O(log(1/tol)) per contract (Brent's method).
        """
        if market_price is None or market_price <= 0 or T <= 0:
            return None

        def objective(sigma):
            return cls.price(S, K, T, r, sigma, option_type) - market_price

        try:
            f_lower, f_upper = objective(lower), objective(upper)
            if np.sign(f_lower) == np.sign(f_upper):
                return None
            iv = brentq(objective, lower, upper, xtol=1e-6, maxiter=200)
            return float(iv)
        except (ValueError, RuntimeError):
            return None


# ======================================================================
# 3. DATA PIPELINE (DEFENSIVE INGESTION)
# ======================================================================

RISK_FREE_PROXY_TICKER = "^IRX"  # 13-week T-bill discount rate


@st.cache_data(ttl=300, show_spinner=False)
def fetch_risk_free_rate() -> float:
    """
    Pull the latest 13-week US T-bill yield (^IRX) as a risk-free rate
    proxy. Falls back to 4.5% if the fetch fails for any reason.
    """
    try:
        tbill = yf.Ticker(RISK_FREE_PROXY_TICKER)
        hist = tbill.history(period="5d")
        if hist.empty:
            return 4.5
        latest = float(hist["Close"].dropna().iloc[-1])
        if latest <= 0 or latest > 20:
            return 4.5
        return round(latest, 2)
    except Exception:
        return 4.5


@st.cache_data(ttl=300, show_spinner=False)
def fetch_spot_and_change(ticker: str) -> Tuple[Optional[float], Optional[float]]:
    """
    Fetch the latest spot price and daily percent change for `ticker`.
    Returns (None, None) if the ticker is invalid or data is unavailable.
    """
    try:
        tk = yf.Ticker(ticker)
        hist = tk.history(period="5d")
        if hist.empty or len(hist) < 1:
            return None, None
        last_close = float(hist["Close"].iloc[-1])
        if len(hist) >= 2:
            prev_close = float(hist["Close"].iloc[-2])
            pct_change = (last_close - prev_close) / prev_close * 100.0
        else:
            pct_change = 0.0
        return last_close, pct_change
    except Exception:
        return None, None


@st.cache_data(ttl=300, show_spinner=False)
def fetch_ticker_info(ticker: str) -> dict:
    """
    Fetch descriptive context (name, exchange, sector, 52-week range) for
    the header strip. Best-effort — returns an empty dict on any failure
    so the app degrades to showing only the ticker symbol.
    """
    try:
        info = yf.Ticker(ticker).info or {}
        return {
            "name": info.get("longName") or info.get("shortName"),
            "exchange": info.get("exchange"),
            "sector": info.get("sector"),
            "week52_low": info.get("fiftyTwoWeekLow"),
            "week52_high": info.get("fiftyTwoWeekHigh"),
        }
    except Exception:
        return {}


@st.cache_data(ttl=300, show_spinner=False)
def fetch_expirations(ticker: str) -> Tuple[str, ...]:
    """Return the tuple of available option expiration date strings."""
    try:
        tk = yf.Ticker(ticker)
        return tuple(tk.options)
    except Exception:
        return tuple()


@st.cache_data(ttl=300, show_spinner=False)
def fetch_option_chain_for_expiry(ticker: str, expiry: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Fetch raw call and put chains for a single expiry. May return empty frames."""
    try:
        tk = yf.Ticker(ticker)
        chain = tk.option_chain(expiry)
        return chain.calls.copy(), chain.puts.copy()
    except Exception:
        return pd.DataFrame(), pd.DataFrame()


def _new_funnel() -> dict:
    """Zeroed funnel-diagnostics dict, one stage per cleaning step."""
    return {"raw": 0, "bid_ask_ok": 0, "spread_ok": 0, "liquidity_ok": 0, "iv_ok": 0}


def _clean_and_enrich(
    df: pd.DataFrame,
    option_type: str,
    spot: float,
    expiry: str,
    r: float,
    require_liquidity: bool = True,
) -> Tuple[pd.DataFrame, dict]:
    """
    Apply defensive data-cleaning rules and enrich with Mid price,
    Moneyness, DTE, Tenor, IV (recomputed via Brent fallback where
    needed), and full Greeks.

    Returns (df, funnel) where funnel is a dict of row counts surviving
    each cleaning stage (raw -> bid/ask valid -> spread ok -> liquidity
    ok -> IV resolved), so a caller can diagnose exactly which stage
    zeroed out a leg (a common real-world cause: `yfinance` sometimes
    reports openInterest/volume as 0 for an entire leg on a given day).
    Returns an empty frame if nothing survives.
    """
    funnel = _new_funnel()
    if df.empty:
        return df, funnel
    funnel["raw"] = len(df)

    df = df.copy()
    df["optionType"] = option_type

    for col in ("bid", "ask", "impliedVolatility", "openInterest", "volume", "strike"):
        if col not in df.columns:
            df[col] = np.nan

    df["bid"] = pd.to_numeric(df["bid"], errors="coerce").fillna(0.0)
    df["ask"] = pd.to_numeric(df["ask"], errors="coerce").fillna(0.0)
    df["openInterest"] = pd.to_numeric(df["openInterest"], errors="coerce").fillna(0)
    df["volume"] = pd.to_numeric(df["volume"], errors="coerce").fillna(0)
    df["strike"] = pd.to_numeric(df["strike"], errors="coerce")

    df["mid"] = (df["bid"] + df["ask"]) / 2.0

    # --- Data cleaning rules ---------------------------------------
    mask = (
        (df["bid"] > 0)
        & (df["ask"] > 0)
        & (df["ask"] >= df["bid"])
        & (df["strike"].notna())
        & (df["strike"] > 0)
    )
    df = df[mask].copy()
    funnel["bid_ask_ok"] = len(df)
    if df.empty:
        return df, funnel

    spread_ratio = (df["ask"] - df["bid"]) / df["mid"].replace(0, np.nan)
    df = df[spread_ratio.fillna(1.0) <= 0.50].copy()
    funnel["spread_ok"] = len(df)
    if df.empty:
        return df, funnel

    if require_liquidity:
        liquidity_mask = (df["openInterest"] > 0) | (df["volume"] > 0)
        df = df[liquidity_mask].copy()
    funnel["liquidity_ok"] = len(df)
    if df.empty:
        return df, funnel

    # --- Tenor / DTE --------------------------------------------------
    try:
        exp_date = dt.datetime.strptime(expiry, "%Y-%m-%d").date()
    except ValueError:
        return pd.DataFrame(), funnel
    today = dt.date.today()
    dte = (exp_date - today).days
    tenor_years = dte / 365.25
    if tenor_years < (1.0 / 365.25):
        return pd.DataFrame(), funnel

    df["expiry"] = expiry
    df["dte"] = dte
    df["tenor"] = tenor_years
    df["moneyness"] = df["strike"] / spot

    # --- Implied Volatility: recompute via Brent, fallback to yfinance
    def _resolve_iv(row):
        solved = BlackScholesEngine.implied_vol(
            market_price=row["mid"],
            S=spot,
            K=row["strike"],
            T=tenor_years,
            r=r,
            option_type=option_type,
        )
        if solved is not None and 0.01 < solved <= 3.0:
            return solved
        fallback = row.get("impliedVolatility", np.nan)
        if pd.notna(fallback) and 0.01 < fallback <= 3.0:
            return float(fallback)
        return np.nan

    df["iv"] = df.apply(_resolve_iv, axis=1)
    df = df[df["iv"].notna() & (df["iv"] > 0.01) & (df["iv"] <= 3.0)].copy()
    funnel["iv_ok"] = len(df)
    if df.empty:
        return df, funnel

    # --- Greeks ---------------------------------------------------
    greeks = BlackScholesEngine.greeks_bundle(
        S=spot, K=df["strike"].values, T=df["tenor"].values, r=r,
        sigma=df["iv"].values, option_type=option_type,
    )
    for name, values in greeks.items():
        df[name] = values

    keep_cols = [
        "expiry", "optionType", "strike", "bid", "ask", "mid",
        "openInterest", "volume", "iv", "dte", "tenor", "moneyness",
        "delta", "gamma", "vega", "theta", "rho",
    ]
    return df[keep_cols].reset_index(drop=True), funnel


@st.cache_data(ttl=300, show_spinner=False)
def build_clean_chain(
    ticker: str,
    max_dte: int,
    r_pct: float,
    option_scope: str,
    require_liquidity: bool = True,
) -> Tuple[pd.DataFrame, int, float, dict]:
    """
    Fetch and clean the full options chain for `ticker` up to `max_dte`
    days out. Returns (clean_df, raw_contract_count, spot_price,
    diagnostics), where diagnostics is a {"call": funnel, "put": funnel}
    dict of cleaning-stage row counts, summed across all expirations, so
    a caller can see exactly where one leg (e.g. calls) got filtered out.

    Time complexity: O(E * C) where E = number of expirations within
    the tenor horizon and C = average contracts per expiry.
    """
    spot, _ = fetch_spot_and_change(ticker)
    diagnostics = {"call": _new_funnel(), "put": _new_funnel()}
    if spot is None:
        return pd.DataFrame(), 0, 0.0, diagnostics

    r = r_pct / 100.0
    expirations = fetch_expirations(ticker)
    if not expirations:
        return pd.DataFrame(), 0, spot, diagnostics

    today = dt.date.today()
    valid_expirations = []
    for exp in expirations:
        try:
            exp_date = dt.datetime.strptime(exp, "%Y-%m-%d").date()
        except ValueError:
            continue
        dte = (exp_date - today).days
        if 0 < dte <= max_dte:
            valid_expirations.append(exp)

    frames = []
    raw_count = 0
    for exp in valid_expirations:
        calls_raw, puts_raw = fetch_option_chain_for_expiry(ticker, exp)
        raw_count += len(calls_raw) + len(puts_raw)

        if option_scope in ("Calls Only", "Both"):
            cleaned_calls, funnel_c = _clean_and_enrich(
                calls_raw, "call", spot, exp, r, require_liquidity
            )
            for k, v in funnel_c.items():
                diagnostics["call"][k] += v
            if not cleaned_calls.empty:
                frames.append(cleaned_calls)
        if option_scope in ("Puts Only", "Both"):
            cleaned_puts, funnel_p = _clean_and_enrich(
                puts_raw, "put", spot, exp, r, require_liquidity
            )
            for k, v in funnel_p.items():
                diagnostics["put"][k] += v
            if not cleaned_puts.empty:
                frames.append(cleaned_puts)

    if not frames:
        return pd.DataFrame(), raw_count, spot, diagnostics

    full_df = pd.concat(frames, ignore_index=True)
    return full_df, raw_count, spot, diagnostics


# ======================================================================
# 4. VOLATILITY SURFACE INTERPOLATION
# ======================================================================

def build_iv_surface_grid(
    df: pd.DataFrame,
    x_mode: str,
    y_mode: str,
    grid_res: int = 50,
):
    """
    Construct a regular (grid_res x grid_res) grid over the chosen
    X/Y axes and interpolate IV using cubic griddata, with a nearest-
    neighbor fallback to eliminate boundary NaNs.

    Returns (X_grid, Y_grid, Z_grid) or (None, None, None) if there
    is insufficient data (< 4 non-collinear points).
    """
    if df.empty or len(df) < 4:
        return None, None, None

    x_vals = df["moneyness"].values if x_mode == "Moneyness (K/S)" else df["strike"].values
    y_vals = df["tenor"].values if y_mode == "Tenor (Years)" else df["dte"].values
    z_vals = df["iv"].values * 100.0  # display in %

    x_lin = np.linspace(x_vals.min(), x_vals.max(), grid_res)
    y_lin = np.linspace(y_vals.min(), y_vals.max(), grid_res)
    X, Y = np.meshgrid(x_lin, y_lin)

    points = np.column_stack([x_vals, y_vals])

    try:
        Z_cubic = interpolate.griddata(points, z_vals, (X, Y), method="cubic")
        Z_nearest = interpolate.griddata(points, z_vals, (X, Y), method="nearest")
        Z = np.where(np.isnan(Z_cubic), Z_nearest, Z_cubic)
        Z = np.clip(Z, 0.5, 300.0)
        return X, Y, Z
    except Exception:
        return None, None, None


def compute_atm_iv(df: pd.DataFrame, target_dte: int = 30) -> Optional[float]:
    """
    Estimate ATM implied volatility nearest to `target_dte` days by
    selecting contracts closest to moneyness == 1.0 within the nearest
    available expiry cluster.
    """
    if df.empty:
        return None
    df = df.copy()
    df["dte_dist"] = (df["dte"] - target_dte).abs()
    nearest_dte = df.loc[df["dte_dist"].idxmin(), "dte"]
    cluster = df[df["dte"] == nearest_dte].copy()
    if cluster.empty:
        return None
    cluster["moneyness_dist"] = (cluster["moneyness"] - 1.0).abs()
    atm_row = cluster.loc[cluster["moneyness_dist"].idxmin()]
    return float(atm_row["iv"]) * 100.0


def compute_skew_index(df: pd.DataFrame, target_dte: int = 30) -> Optional[float]:
    """
    Volatility Skew Index: 25-delta Put IV minus 25-delta Call IV,
    using the nearest available expiry to `target_dte`. Approximates
    the 25-delta point as the contract whose |delta| is closest to
    0.25 within calls and puts respectively.
    """
    if df.empty:
        return None
    df = df.copy()
    df["dte_dist"] = (df["dte"] - target_dte).abs()
    nearest_dte = df.loc[df["dte_dist"].idxmin(), "dte"]
    cluster = df[df["dte"] == nearest_dte]

    calls = cluster[cluster["optionType"] == "call"].copy()
    puts = cluster[cluster["optionType"] == "put"].copy()
    if calls.empty or puts.empty:
        return None

    calls["delta_dist"] = (calls["delta"] - 0.25).abs()
    puts["delta_dist"] = (puts["delta"] + 0.25).abs()

    call_25d = calls.loc[calls["delta_dist"].idxmin()]
    put_25d = puts.loc[puts["delta_dist"].idxmin()]

    return float(put_25d["iv"] - call_25d["iv"]) * 100.0


# ======================================================================
# 5. SIDEBAR CONTROLS
# ======================================================================

st.sidebar.markdown("## :material/tune: Controls")
st.sidebar.markdown("---")

quick_pick = st.sidebar.radio(
    "Quick Select",
    options=["Custom", "SPY", "AAPL", "NVDA", "TSLA", "QQQ"],
    horizontal=True,
    index=1,
)
default_ticker = "SPY" if quick_pick == "Custom" else quick_pick
ticker_input = st.sidebar.text_input("Ticker Symbol", value=default_ticker).strip().upper()

fetch_clicked = st.sidebar.button(
    "Instant Fetch", icon=":material/sync:", use_container_width=True, type="primary"
)

st.sidebar.markdown("---")

live_rf = fetch_risk_free_rate()
risk_free_rate = st.sidebar.slider(
    "Risk-Free Rate r (%)",
    min_value=0.0, max_value=10.0, value=float(live_rf), step=0.1,
    help="Seeded from the 13-week T-bill yield (^IRX).",
)

option_scope = st.sidebar.selectbox(
    "Option Type",
    options=["Both", "Calls Only", "Puts Only"],
    index=0,
)

max_dte = st.sidebar.slider(
    "Tenor Horizon (Days to Expiration)",
    min_value=7, max_value=365, value=180, step=1,
)

require_liquidity = st.sidebar.checkbox(
    "Require Open Interest / Volume > 0",
    value=True,
    help="Disable to include quoted contracts even when yfinance reports 0 OI/volume for a leg.",
)

st.sidebar.markdown("---")

render_mode = st.sidebar.radio(
    "Surface Rendering",
    options=["2D Contour", "3D Surface (WebGL)"],
    index=0,
)
x_axis_mode = st.sidebar.selectbox("Surface X-Axis", ["Strike ($)", "Moneyness (K/S)"], index=1)
y_axis_mode = st.sidebar.selectbox("Surface Y-Axis", ["Tenor (Years)", "DTE (Days)"], index=1)
show_scatter_overlay = st.sidebar.checkbox("Overlay Market Contracts", value=True)

st.sidebar.markdown("---")
if st.sidebar.button("Clear Cache", icon=":material/delete_sweep:", use_container_width=True):
    st.cache_data.clear()
    st.sidebar.success("Cache cleared.", icon=":material/check_circle:")

st.sidebar.markdown("---")
st.sidebar.caption(":material/database: Yahoo Finance via `yfinance` · delayed quotes")


# ======================================================================
# 6. MAIN APP HEADER
# ======================================================================

st.markdown(
    """
    <div class="app-header">
        <span class="icon-glyph">monitoring</span>
        <div>
            <h1 style="margin-bottom:0;">VolSurface Pro</h1>
        </div>
    </div>
    <div class="app-subtitle">Options Surface &amp; Greeks Analytics — implied-volatility surfaces, skew diagnostics, and risk decomposition</div>
    """,
    unsafe_allow_html=True,
)
st.markdown("---")

if not ticker_input:
    st.warning("Enter a ticker symbol in the sidebar to begin.", icon=":material/search:")
    st.stop()

with st.spinner(f"Fetching and cleaning options chain for {ticker_input}..."):
    clean_df, raw_count, spot_price, pipeline_diag = build_clean_chain(
        ticker=ticker_input,
        max_dte=max_dte,
        r_pct=risk_free_rate,
        option_scope=option_scope,
        require_liquidity=require_liquidity,
    )

if spot_price == 0.0 or spot_price is None:
    st.error(f"No market data found for '{ticker_input}'. Check the ticker and try again.", icon=":material/error:")
    st.stop()

spot, daily_change = fetch_spot_and_change(ticker_input)
ticker_info = fetch_ticker_info(ticker_input)

if ticker_info.get("name"):
    context_bits = [f"<b>{ticker_info['name']}</b>"]
    if ticker_info.get("exchange"):
        context_bits.append(ticker_info["exchange"])
    if ticker_info.get("sector"):
        context_bits.append(ticker_info["sector"])
    if ticker_info.get("week52_low") and ticker_info.get("week52_high"):
        context_bits.append(f"52W Range ${ticker_info['week52_low']:.2f} – ${ticker_info['week52_high']:.2f}")
    st.markdown(
        f'<div class="ticker-context">{" &nbsp;·&nbsp; ".join(context_bits)}</div>',
        unsafe_allow_html=True,
    )

# --- Data pipeline diagnostics: per-leg filter funnel.
with st.expander("Data Pipeline Diagnostics — per-leg filter funnel", icon=":material/troubleshoot:"):
    funnel_df = pd.DataFrame(pipeline_diag).T.rename(columns={
        "raw": "Raw", "bid_ask_ok": "Valid Bid/Ask", "spread_ok": "Spread ≤ 50%",
        "liquidity_ok": "OI or Vol > 0", "iv_ok": "IV Resolved",
    })
    funnel_df.index = funnel_df.index.str.capitalize()
    st.dataframe(funnel_df, use_container_width=True)
    st.caption(
        "A leg dropping to 0 at **OI or Vol > 0** means yfinance reported zero "
        "open interest/volume across that entire leg for this ticker/day. "
        "Disable **Require Open Interest / Volume > 0** in the sidebar to "
        "include those contracts (bid/ask and spread checks still apply)."
    )

if clean_df.empty:
    st.warning(
        f"No clean contracts for '{ticker_input}' within a {max_dte}-day horizon. "
        "Widen the tenor horizon, switch tickers, or disable the liquidity filter.",
        icon=":material/filter_alt_off:",
    )
    st.stop()


# ======================================================================
# 7. KPI METRIC CARDS
# ======================================================================

atm_iv_30d = compute_atm_iv(clean_df, target_dte=30)
skew_idx = compute_skew_index(clean_df, target_dte=30)
filtered_count = len(clean_df)
n_expirations = clean_df["expiry"].nunique()
avg_spread_pct = ((clean_df["ask"] - clean_df["bid"]) / clean_df["mid"]).mean() * 100
total_oi = int(clean_df["openInterest"].sum())
call_oi = clean_df.loc[clean_df["optionType"] == "call", "openInterest"].sum()
put_oi = clean_df.loc[clean_df["optionType"] == "put", "openInterest"].sum()
put_call_oi_ratio = (put_oi / call_oi) if call_oi > 0 else None

kpi_cols = st.columns(4)

with kpi_cols[0]:
    st.metric(
        label=f"{ticker_input} Spot Price",
        value=f"${spot:,.2f}" if spot is not None else "N/A",
        delta=f"{daily_change:+.2f}%" if daily_change is not None else None,
        icon=":material/payments:",
    )

with kpi_cols[1]:
    st.metric(
        label="ATM 30D Implied Vol",
        value=f"{atm_iv_30d:.2f}%" if atm_iv_30d is not None else "N/A",
        icon=":material/candlestick_chart:",
    )

with kpi_cols[2]:
    st.metric(
        label="25Δ Skew (Put − Call)",
        value=f"{skew_idx:+.2f} pts" if skew_idx is not None else "N/A",
        help="Positive = put-skew (crash premium). Negative = call-skew.",
        icon=":material/balance:",
    )

with kpi_cols[3]:
    st.metric(
        label="Contracts Analyzed",
        value=f"{filtered_count:,} / {raw_count:,}",
        help="Filtered vs. total raw contracts fetched.",
        icon=":material/filter_alt:",
    )

kpi_cols2 = st.columns(4)

with kpi_cols2[0]:
    st.metric(
        label="Expirations Covered",
        value=f"{n_expirations}",
        icon=":material/event_available:",
    )

with kpi_cols2[1]:
    st.metric(
        label="Avg Bid-Ask Spread",
        value=f"{avg_spread_pct:.2f}%",
        icon=":material/swap_horiz:",
    )

with kpi_cols2[2]:
    st.metric(
        label="Put/Call OI Ratio",
        value=f"{put_call_oi_ratio:.2f}" if put_call_oi_ratio is not None else "N/A",
        icon=":material/pie_chart:",
    )

with kpi_cols2[3]:
    st.metric(
        label="Total Open Interest",
        value=f"{total_oi:,}",
        icon=":material/inventory_2:",
    )

st.markdown("---")


# ======================================================================
# 8. TABS
# ======================================================================

tab1, tab2, tab3, tab4 = st.tabs([
    ":material/view_in_ar: 3D Volatility Surface",
    ":material/show_chart: Smile & Term Structure",
    ":material/grid_view: Greeks Risk Matrix",
    ":material/table_chart: Raw Data & Export",
])

# ----------------------------------------------------------------------
# TAB 1: 3D VOLATILITY SURFACE
# ----------------------------------------------------------------------
with tab1:
    X, Y, Z = build_iv_surface_grid(clean_df, x_axis_mode, y_axis_mode, grid_res=50)

    if X is None:
        st.warning("Insufficient data points to interpolate a surface — widen the tenor horizon.", icon=":material/warning:")
    elif render_mode.startswith("3D"):
        fig_surface = go.Figure()

        fig_surface.add_trace(
            go.Surface(
                x=X, y=Y, z=Z,
                colorscale="Viridis",
                opacity=0.92,
                contours={
                    "z": {"show": True, "usecolormap": True, "project_z": True}
                },
                colorbar=dict(title="IV (%)", tickfont=dict(color="#e6e6e6")),
                hovertemplate=(
                    f"{x_axis_mode}: %{{x:.3f}}<br>"
                    f"{y_axis_mode}: %{{y:.2f}}<br>"
                    "IV: %{z:.2f}%<extra></extra>"
                ),
            )
        )

        if show_scatter_overlay:
            x_scatter = clean_df["moneyness"] if x_axis_mode == "Moneyness (K/S)" else clean_df["strike"]
            y_scatter = clean_df["tenor"] if y_axis_mode == "Tenor (Years)" else clean_df["dte"]
            z_scatter = clean_df["iv"] * 100.0

            hover_text = [
                f"Strike: ${row.strike:.2f}<br>DTE: {row.dte}<br>IV: {row.iv*100:.2f}%<br>Δ: {row.delta:.3f}"
                for row in clean_df.itertuples()
            ]

            fig_surface.add_trace(
                go.Scatter3d(
                    x=x_scatter, y=y_scatter, z=z_scatter,
                    mode="markers",
                    marker=dict(size=3, color=ELECTRIC_BLUE, opacity=0.7),
                    name="Market Contracts",
                    hovertext=hover_text,
                    hoverinfo="text",
                )
            )

        fig_surface.update_layout(
            **PLOTLY_DARK_LAYOUT,
            scene=dict(
                xaxis=dict(title=x_axis_mode, backgroundcolor="#0b0d13", gridcolor="#1f2430", color="#e6e6e6"),
                yaxis=dict(title=y_axis_mode, backgroundcolor="#0b0d13", gridcolor="#1f2430", color="#e6e6e6"),
                zaxis=dict(title="Implied Vol (%)", backgroundcolor="#0b0d13", gridcolor="#1f2430", color="#e6e6e6"),
                camera=dict(eye=dict(x=1.5, y=-1.5, z=0.8)),
            ),
            height=700,
            title=f"{ticker_input} Implied Volatility Surface",
        )
        st.plotly_chart(fig_surface, use_container_width=True)
        st.caption("50×50 grid · `scipy.interpolate.griddata` (cubic, nearest-neighbor fallback)")
    else:
        # --- 2D Contour: same interpolated grid, rendered with Plotly's
        # --- SVG/canvas Contour trace instead of a WebGL Surface.
        fig_contour = go.Figure()

        fig_contour.add_trace(
            go.Contour(
                x=X[0, :], y=Y[:, 0], z=Z,
                colorscale="Viridis",
                contours=dict(coloring="heatmap", showlabels=True, labelfont=dict(size=10, color="#0b0d13")),
                colorbar=dict(title="IV (%)", tickfont=dict(color="#e6e6e6")),
                hovertemplate=(
                    f"{x_axis_mode}: %{{x:.3f}}<br>"
                    f"{y_axis_mode}: %{{y:.2f}}<br>"
                    "IV: %{z:.2f}%<extra></extra>"
                ),
            )
        )

        if show_scatter_overlay:
            x_scatter = clean_df["moneyness"] if x_axis_mode == "Moneyness (K/S)" else clean_df["strike"]
            y_scatter = clean_df["tenor"] if y_axis_mode == "Tenor (Years)" else clean_df["dte"]

            hover_text = [
                f"Strike: ${row.strike:.2f}<br>DTE: {row.dte}<br>IV: {row.iv*100:.2f}%<br>Δ: {row.delta:.3f}"
                for row in clean_df.itertuples()
            ]

            fig_contour.add_trace(
                go.Scatter(
                    x=x_scatter, y=y_scatter,
                    mode="markers",
                    marker=dict(size=6, color="white", line=dict(width=1, color=ELECTRIC_BLUE), opacity=0.85),
                    name="Market Contracts",
                    hovertext=hover_text,
                    hoverinfo="text",
                )
            )

        fig_contour.update_layout(
            **PLOTLY_DARK_LAYOUT,
            xaxis=dict(title=x_axis_mode, gridcolor="#1f2430"),
            yaxis=dict(title=y_axis_mode, gridcolor="#1f2430"),
            height=700,
            title=f"{ticker_input} Implied Volatility Surface — 2D Contour",
        )
        st.plotly_chart(fig_contour, use_container_width=True)
        st.caption("50×50 grid · `scipy.interpolate.griddata` (cubic, nearest-neighbor fallback)")

# ----------------------------------------------------------------------
# TAB 2: SMILE & TERM STRUCTURE
# ----------------------------------------------------------------------
with tab2:
    col_smile, col_term = st.columns(2)

    with col_smile:
        st.subheader("Volatility Smile")
        available_expiries = sorted(clean_df["expiry"].unique())
        selected_expiry = st.selectbox("Expiration Date", available_expiries, key="smile_expiry")

        smile_df = clean_df[clean_df["expiry"] == selected_expiry].sort_values("strike")
        smile_calls = smile_df[smile_df["optionType"] == "call"]
        smile_puts = smile_df[smile_df["optionType"] == "put"]

        fig_smile = go.Figure()
        if not smile_calls.empty:
            fig_smile.add_trace(go.Scatter(
                x=smile_calls["strike"], y=smile_calls["iv"] * 100,
                mode="markers+lines", name="Calls",
                line=dict(color=NEON_GREEN, width=1.5),
                marker=dict(size=6),
                error_y=dict(
                    type="data",
                    array=(smile_calls["ask"] - smile_calls["bid"]) / smile_calls["mid"].replace(0, np.nan) * 5,
                    visible=True, thickness=1, width=2, color="rgba(0,255,163,0.35)",
                ),
            ))
        if not smile_puts.empty:
            fig_smile.add_trace(go.Scatter(
                x=smile_puts["strike"], y=smile_puts["iv"] * 100,
                mode="markers+lines", name="Puts",
                line=dict(color=ELECTRIC_BLUE, width=1.5),
                marker=dict(size=6),
                error_y=dict(
                    type="data",
                    array=(smile_puts["ask"] - smile_puts["bid"]) / smile_puts["mid"].replace(0, np.nan) * 5,
                    visible=True, thickness=1, width=2, color="rgba(0,212,255,0.35)",
                ),
            ))
        fig_smile.add_vline(x=spot, line_dash="dash", line_color="#8b93a7", annotation_text="Spot")
        fig_smile.update_layout(
            **PLOTLY_DARK_LAYOUT,
            xaxis_title="Strike Price ($)",
            yaxis_title="Implied Volatility (%)",
            height=460,
            legend=dict(orientation="h", y=1.08),
        )
        st.plotly_chart(fig_smile, use_container_width=True)

    with col_term:
        st.subheader("Term Structure (ATM IV)")
        term_records = []
        for exp, group in clean_df.groupby("expiry"):
            group = group.copy()
            group["m_dist"] = (group["moneyness"] - 1.0).abs()
            atm_row = group.loc[group["m_dist"].idxmin()]
            term_records.append({"expiry": exp, "dte": atm_row["dte"], "atm_iv": atm_row["iv"] * 100})
        term_df = pd.DataFrame(term_records).sort_values("dte")

        fig_term = go.Figure()
        fig_term.add_trace(go.Scatter(
            x=term_df["dte"], y=term_df["atm_iv"],
            mode="lines+markers",
            line=dict(color=NEON_GREEN, width=2.5),
            marker=dict(size=8, color=ELECTRIC_BLUE),
            fill="tozeroy",
            fillcolor="rgba(0,255,163,0.08)",
        ))
        fig_term.update_layout(
            **PLOTLY_DARK_LAYOUT,
            xaxis_title="Days to Expiration",
            yaxis_title="ATM Implied Vol (%)",
            height=460,
        )
        st.plotly_chart(fig_term, use_container_width=True)

        if len(term_df) >= 2:
            slope = term_df["atm_iv"].iloc[-1] - term_df["atm_iv"].iloc[0]
            regime = "Contango (upward-sloping)" if slope > 0 else "Backwardation (downward-sloping)"
            st.caption(f"Term structure regime: **{regime}** (Δ{slope:+.2f} pts front-to-back).")

# ----------------------------------------------------------------------
# TAB 3: GREEKS RISK MATRIX
# ----------------------------------------------------------------------
with tab3:
    st.subheader("Greeks Exposure Heatmaps")
    greek_choice = st.radio(
        "Select Greek", ["Gamma", "Vega", "Theta"], horizontal=True, key="greek_matrix_choice"
    )
    greek_col = greek_choice.lower()

    pivot_df = clean_df.pivot_table(
        index="dte", columns="strike", values=greek_col, aggfunc="mean"
    ).sort_index()

    if pivot_df.empty:
        st.warning("Not enough data to build a Greeks matrix for the current selection.")
    else:
        fig_heat = go.Figure(
            data=go.Heatmap(
                z=pivot_df.values,
                x=pivot_df.columns,
                y=pivot_df.index,
                colorscale="Plasma" if greek_choice != "Theta" else "Electric",
                colorbar=dict(title=greek_choice),
                hovertemplate="Strike: %{x}<br>DTE: %{y}<br>" + f"{greek_choice}: " + "%{z:.4f}<extra></extra>",
            )
        )
        fig_heat.update_layout(
            **PLOTLY_DARK_LAYOUT,
            xaxis_title="Strike Price ($)",
            yaxis_title="Days to Expiration",
            height=560,
            title=f"{ticker_input} {greek_choice} Exposure — Strike × Expiration",
        )
        st.plotly_chart(fig_heat, use_container_width=True)
        st.caption(
            "Gamma and Vega are pooled (mean) across calls/puts at each grid cell; "
            "Theta is shown as 1-day calendar decay."
        )

# ----------------------------------------------------------------------
# TAB 4: RAW DATA & EXPORT
# ----------------------------------------------------------------------
with tab4:
    st.subheader("Options Chain")

    display_df = clean_df.copy()
    display_df["iv"] = (display_df["iv"] * 100).round(2)
    display_df["moneyness"] = display_df["moneyness"].round(4)
    for col in ["delta", "gamma", "vega", "theta", "rho"]:
        display_df[col] = display_df[col].round(5)
    display_df = display_df.rename(columns={
        "iv": "IV (%)", "dte": "DTE", "tenor": "Tenor (Yrs)", "moneyness": "Moneyness",
        "optionType": "Type", "openInterest": "OI",
    })

    st.dataframe(
        display_df.sort_values(["expiry", "strike"]),
        use_container_width=True,
        height=520,
    )

    csv_bytes = clean_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="⬇️ Download Options Chain as CSV",
        data=csv_bytes,
        file_name=f"{ticker_input}_clean_options_chain_{dt.date.today().isoformat()}.csv",
        mime="text/csv",
        use_container_width=True,
    )

    st.caption(
        f"{filtered_count:,} liquid, arbitrage-screened contracts kept out of "
        f"{raw_count:,} raw contracts fetched across "
        f"{clean_df['expiry'].nunique()} expirations."
    )


# ======================================================================
# 9. FOOTER
# ======================================================================

st.markdown(
    """
    <div class="footer-badge">
        <strong>Quantitative Analytics &amp; Automation Portfolio - Sudhir</strong>
        &nbsp;|&nbsp;
        <a href="https://github.com/spelsudhir/volsurface-pro" target="_blank">GitHub</a>
    </div>
    """,
    unsafe_allow_html=True,
)
