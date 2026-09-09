# ⚡ VolSurface Pro — Options Surface & Greeks Analytics Engine

[![Python Version](https://img.shields.io/badge/python-3.10%2B-blue.svg?style=flat-square)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/framework-Streamlit-FF4B4B.svg?style=flat-square)](https://streamlit.io/)
[![Math Engine](https://img.shields.io/badge/engine-NumPy%20%7C%20SciPy-013243.svg?style=flat-square)](https://scipy.org/)
[![Visualization](https://img.shields.io/badge/charts-Plotly%20Graph%20Objects-7E57C2.svg?style=flat-square)](https://plotly.com/)
[![License](https://img.shields.io/badge/license-MIT-green.svg?style=flat-square)](LICENSE)

**VolSurface Pro** is an institutional-style quantitative derivatives analytics workstation built with Python and Streamlit. It ingests live US equity and ETF option chains, performs defensive market-data sanitization, calculates implied volatility using robust numerical inversion, constructs OTM-based volatility surfaces, and generates analytical Black-Scholes-Merton Greeks.

The application combines a **quantitative derivatives engine** with a high-contrast **Bloomberg/Linear-inspired terminal interface**, providing an interactive environment for volatility analysis, risk visualization, and options-chain diagnostics.

> **Educational / Research Use Only:** VolSurface Pro is intended for quantitative research, portfolio demonstration, and educational purposes. It is not investment advice and should not be used as the sole basis for trading or investment decisions.

<img width="1440" height="900" alt="Screenshot 2026-09-09 at 5 38 14 PM" src="https://github.com/user-attachments/assets/84315d1a-0dc3-474c-ba02-c40386164f48" />

<img width="1440" height="900" alt="Screenshot 2026-09-09 at 5 37 55 PM" src="https://github.com/user-attachments/assets/59ecffb0-9481-4cc7-8f10-09d68501442a" />

<img width="1440" height="900" alt="Screenshot 2026-09-09 at 5 37 48 PM" src="https://github.com/user-attachments/assets/6c9126c3-0312-4176-a29f-9ad4e6bdb03e" />

---

## 📌 Overview

Volatility surfaces derived directly from retail or delayed option-chain data can exhibit severe numerical artifacts. Deep-ITM contracts, wide bid-ask spreads, stale quotes, sparse strikes, and inconsistent market prices can produce unstable implied-volatility inversions and misleading risk surfaces.

VolSurface Pro addresses these issues through a **defensive multi-stage analytics pipeline**:

```text
Live Option Chain
       │
       ▼
Market Data Sanitization
       │
       ├── Bid / Ask Validation
       ├── Dynamic Spread Filtering
       ├── Volume / OI Liquidity Checks
       └── No-Arbitrage Boundary Screening
       │
       ▼
OTM Volatility Universe
       │
       ├── OTM Puts  → K < S
       └── OTM Calls → K ≥ S
       │
       ▼
Robust IV Inversion
       │
       ├── Bounded Brent Solver
       └── Put-Call Parity Synthetic Quotes
       │
       ▼
Regularized Volatility Surface
       │
       ├── 50 × 50 Grid
       ├── Cubic Interpolation
       └── Nearest-Neighbor Boundary Fill
       │
       ▼
Risk Analytics & Visualization
       │
       ├── 3D Volatility Surface
       ├── 2D Volatility Smile
       ├── ATM Term Structure
       ├── Greek Risk Heatmaps
       └── Filtered Option Chain
```

---

# 🚀 Key Features

## 1. Vectorized Black-Scholes-Merton Engine

The application implements a vectorized analytical BSM engine using NumPy and SciPy.

Supported outputs include:

* Option price
* Implied volatility
* Delta
* Gamma
* Vega
* Theta
* Rho

The engine is designed to operate efficiently on entire option-chain arrays rather than requiring independent Python-level calculations for every observation.

---

## 2. Robust Implied Volatility Inversion

Implied volatility is recovered numerically using Brent's root-finding algorithm:

```text
scipy.optimize.brentq
```

The solver is bounded within a configurable numerical volatility domain to prevent pathological solutions.

The production pipeline additionally applies theoretical option-price bounds before attempting inversion:

### Call

$$
C \geq \max(0,S-Ke^{-rT})
$$

### Put

$$
P \geq \max(0,Ke^{-rT}-S)
$$

Quotes that violate these fundamental boundaries are excluded before entering the inversion stage.

This prevents impossible or numerically unstable market observations from contaminating the volatility surface.

---

# 📐 Volatility Surface Construction

## 3. OTM-Blended Smile Calibration

VolSurface Pro deliberately avoids fitting volatility directly from deep-ITM options.

For a spot price $S$:

### OTM Puts

$$
K < S
$$

### OTM Calls

$$
K \geq S
$$

This OTM-blended construction is considerably more robust for scraped or retail option-chain data because OTM contracts generally provide cleaner volatility information than deep-ITM contracts whose prices are dominated by intrinsic value.

This approach specifically mitigates:

* Deep-ITM IV explosions
* Vertical IV walls
* Inverted-V smiles
* Tiny time-value inversion errors
* Wide-spread ITM artifacts

---

# 🔄 Put-Call Parity Synthetic Quotes

Single-leg analysis introduces an additional problem: a Calls-Only or Puts-Only chain does not necessarily contain a liquid OTM quote at every strike.

VolSurface Pro therefore uses European put-call parity to construct an equivalent OTM quote where necessary.

The parity relationship is:

$$
C - P = S - Ke^{-rT}
$$

Therefore:

#### Synthetic Put

$$
P_{\text{synthetic}} = C - S + Ke^{-rT}
$$

#### Synthetic Call

$$
C_{\text{synthetic}} = P + S - Ke^{-rT}
$$

This allows the application to avoid directly inverting unstable deep-ITM quotes while retaining useful strike information.

---

# 💧 Dynamic Liquidity Filtering

## 4. Adaptive Bid-Ask Screening

A rigid percentage-only spread filter can eliminate a disproportionate amount of useful OTM wing information.

Instead, VolSurface Pro uses an adaptive absolute/relative spread rule:

$$
\text{Spread}
\leq
\max(0.25,\;0.40\times\text{Mid})
$$

This has two important effects:

* Protects against excessively wide spreads on liquid contracts.
* Preserves useful low-dollar OTM options whose absolute spread can be small despite a high percentage spread.

### Liquidity Condition

A contract is considered sufficiently active when:

$$
(\text{Volume}>0)
\quad\lor\quad
(\text{Open Interest}>0)
$$

This prevents the pipeline from unnecessarily discarding contracts that have either current trading activity **or** established open interest.

---

# 🛡️ Defensive Arbitrage Screening

Before fitting volatility, market quotes pass through basic no-arbitrage checks.

## Call Lower Bound

$$
C\geq\max(0,S-Ke^{-rT})
$$

## Put Lower Bound

$$
P\geq\max(0,Ke^{-rT}-S)
$$

## Call Strike Monotonicity

For:

$$
K_1<K_2
$$

the corresponding call prices should satisfy:

$$
C(K_1)\geq C(K_2)
$$

The application sorts calls by strike and removes observations that materially violate the expected decreasing price relationship.

This acts as a defensive data-quality layer rather than attempting to perform a full global arbitrage-free calibration.

---

# 📊 ATM Term Structure

## 5. Robust ATM IV Extraction

Selecting the single closest strike to spot can introduce severe term-structure noise when that particular contract is stale or illiquid.

Instead, VolSurface Pro identifies near-ATM contracts satisfying:

$$
0.97\leq\frac{K}{S}\leq1.03
$$

The resulting observations are summarized using a liquidity-aware median approach based on available volume and open interest.

If no strike falls inside the ATM band, the application identifies the immediate strikes surrounding spot:

$$
K_{ATM-}\leq S\leq K_{ATM+}
$$

and performs linear interpolation between their implied volatilities.

This produces a significantly more stable representation of the ATM term structure.

---

# 🧮 Mathematical Reference

## Black-Scholes-Merton Pricing

For:

* $S$ = underlying spot price
* $K$ = strike price
* $T$ = time to expiration in years
* $r$ = continuously compounded risk-free rate
* $\sigma$ = implied volatility

the model calculates:

$$
d_1=
\frac{
\ln(S/K)+(r+\sigma^2/2)T
}{
\sigma\sqrt{T}
}
$$

$$
d_2=d_1-\sigma\sqrt{T}
$$

The Black–Scholes European option formulas are:

#### European Call

$$
C = S\Phi(d_1) - Ke^{-rT}\Phi(d_2)
$$

#### European Put

$$
P = Ke^{-rT}\Phi(-d_2) - S\Phi(-d_1)
$$

---

# 📐 Analytical Greeks

| Greek     | Formula                                                         | Normalization            |
| --------- | --------------------------------------------------------------- | ------------------------ |
| **Delta** | Call: $\Phi(d_1)$ / Put: $\Phi(d_1)-1$                          | Per unit underlying move |
| **Gamma** | $\frac{\phi(d_1)}{S\sigma\sqrt{T}}$                             | Per unit underlying²     |
| **Vega**  | $S\sqrt{T}\phi(d_1)$                                            | Per 1% volatility change |
| **Theta** | $-\frac{S\phi(d_1)\sigma}{2\sqrt{T}}\mp rKe^{-rT}\Phi(\pm d_2)$ | Per calendar day         |
| **Rho**   | Call: $KTe^{-rT}\Phi(d_2)$ / Put: $-KTe^{-rT}\Phi(-d_2)$        | Per 1% rate change       |

Vega and Rho are normalized by dividing by 100, while Theta is converted from annualized units to a daily measure.

---

# 🧊 Continuous Greek Risk Matrices

Raw option chains contain irregular strike and expiry coordinates.

Directly pivoting these observations into a matrix therefore produces sparse grids with large blank regions.

VolSurface Pro regularizes the raw Greek observations onto a uniform:

$$
50\times50
$$

**Strike × DTE** coordinate grid.

The interpolation pipeline is:

```text
Raw Greek Observations
          │
          ▼
Duplicate Coordinate Aggregation
          │
          ▼
SciPy griddata(method="cubic")
          │
          ├── Valid → retain cubic value
          │
          └── NaN / failure
                    │
                    ▼
          Nearest-Neighbor Fill
```

This produces continuous interactive heatmaps while preserving the original market observations as the interpolation inputs.

The same regularization approach is used for the volatility surface.

---

# 📈 Interactive Analytics

The application provides four primary analytical views.

## Tab 1 — Volatility Surface

Interactive Plotly 3D visualization containing:

* Regularized IV surface
* Raw market observations
* Strike or moneyness axis
* DTE or tenor axis
* Interactive rotation and zoom

The 3D visualization combines:

```python
plotly.graph_objects.Surface
plotly.graph_objects.Scatter3d
```

to distinguish the interpolated surface from the underlying market observations.

---

## Tab 2 — Smile & Term Structure

### Volatility Smile

Displays implied volatility against:

* Strike, or
* Moneyness

for the selected expiry.

### ATM Term Structure

Displays ATM IV against DTE, allowing users to identify:

* Contango
* Backwardation
* Term-structure steepness
* Short-dated volatility dislocations

---

## Tab 3 — Greeks Risk Matrix

Interactive heatmaps are available for:

* Delta
* Gamma
* Vega
* Theta
* Rho

Each matrix is generated from the regularized 50 × 50 Strike/DTE surface.

---

## Tab 4 — Option Chain

Provides a filtered market-data view containing relevant option-chain information and calculated analytics.

The application also provides one-click CSV export for downstream analysis.

---

# 🖥️ Terminal-Style Interface

The UI follows a high-contrast quantitative-terminal aesthetic inspired by Bloomberg and modern Linear-style interfaces.

Design characteristics include:

* Dark background
* Monospaced numerical typography
* High-contrast KPI cards
* Neon green analytical highlights
* Electric-blue interactive elements
* Compact sidebar controls
* Dense but readable quantitative tables
* Plotly dark-mode visualizations

The objective is to maximize information density without sacrificing usability.

---

# 🔬 Analytics Pipeline

The complete production workflow can be summarized as:

```text
┌──────────────────────────────────────────────────────────────┐
│                     LIVE INGESTION                           │
│                                                              │
│  yfinance → US Equity / ETF Option Chains                    │
│  Spot + Expirations + Calls + Puts                           │
└─────────────────────────────┬────────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────┐
│               DEFENSIVE DATA SANITIZATION                    │
│                                                              │
│  • Bid / Ask validation                                      │
│  • Dynamic spread threshold                                  │
│  • Volume OR Open Interest liquidity                         │
│  • IV sanity bounds                                          │
│  • Lower-bound arbitrage checks                              │
│  • Call strike monotonicity                                  │
└─────────────────────────────┬────────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────┐
│                   OTM FIT UNIVERSE                           │
│                                                              │
│  K < S  → OTM Puts                                           │
│  K ≥ S  → OTM Calls                                          │
│                                                              │
│  Single-leg modes → Put-Call Parity synthetic quotes         │
└─────────────────────────────┬────────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────┐
│                 QUANTITATIVE MATH ENGINE                     │
│                                                              │
│  • BSM pricing                                               │
│  • Brent IV inversion                                        │
│  • Delta / Gamma / Vega / Theta / Rho                        │
│  • ATM IV extraction                                         │
│  • Term structure                                            │
└─────────────────────────────┬────────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────┐
│                 SURFACE REGULARIZATION                       │
│                                                              │
│  • Duplicate aggregation                                     │
│  • 50 × 50 grid                                              │
│  • Cubic griddata interpolation                              │
│  • Nearest-neighbor boundary completion                      │
│  • IV clipping / sanity controls                             │
└─────────────────────────────┬────────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────┐
│                   INTERACTIVE WORKSTATION                    │
│                                                              │
│  3D IV Surface │ Smile │ Term Structure │ Greeks │ Chain     │
│                                                              │
│                    CSV Export                                │
└──────────────────────────────────────────────────────────────┘
```

---

# 🧱 Project Structure

```text
volsurface-pro/
│
├── .streamlit/
│   └── config.toml
│       └── Streamlit theme & server configuration
│
├── app.py
│   └── Quantitative engine, data pipeline,
│       analytics and Streamlit UI
│
├── requirements.txt
│   └── Runtime dependencies
│
├── LICENSE
│   └── MIT License
│
└── README.md
    └── Technical documentation
```

---

# 📦 Requirements

VolSurface Pro requires:

* Python 3.10+
* Streamlit
* NumPy
* SciPy
* Pandas
* Plotly
* yfinance

Example `requirements.txt`:

```text
streamlit>=1.35.0
numpy>=1.24.0
scipy>=1.10.0
pandas>=2.0.0
plotly>=5.18.0
yfinance>=0.2.40
```

---

# 🚀 Installation

## 1. Clone the repository

```bash
git clone https://github.com/spelsudhir/volsurface-pro.git
cd volsurface-pro
```

## 2. Create a virtual environment

### macOS / Linux

```bash
python -m venv .venv
source .venv/bin/activate
```

### Windows

```powershell
python -m venv .venv
.venv\Scripts\activate
```

## 3. Install dependencies

```bash
pip install -r requirements.txt
```

## 4. Launch VolSurface Pro

```bash
streamlit run app.py
```

The application will start a local Streamlit server and open the quantitative workstation in your browser.

---

# ⚙️ Design Principles

VolSurface Pro is built around several quantitative engineering principles:

### Defensive First

Market data is assumed to be noisy, incomplete, stale, or internally inconsistent until proven otherwise.

### OTM Before ITM

OTM contracts are prioritized for volatility extraction because deep-ITM prices contain relatively little time-value information.

### Bound Every Numerical Procedure

Numerical inversion is constrained by explicit theoretical price and volatility boundaries.

### Preserve Information

Liquidity filters should remove bad observations without unnecessarily starving the analytical universe.

### Regularize Before Visualization

Irregular market observations are transformed into continuous coordinate grids before rendering surfaces and risk matrices.

### Separate Market Data from Model Data

Raw quotes remain distinguishable from model-derived values, synthetic parity quotes, and interpolated observations.

---

# 🧪 Numerical Stability Controls

The production engine incorporates several defensive mechanisms:

| Control                     | Purpose                                      |
| --------------------------- | -------------------------------------------- |
| Dynamic spread filter       | Retains informative low-dollar OTM contracts |
| Volume OR OI                | Prevents unnecessary contract starvation     |
| OTM-only fitting            | Avoids deep-ITM inversion artifacts          |
| Put-call parity             | Bridges missing single-leg OTM information   |
| Price lower bounds          | Rejects theoretically impossible quotes      |
| Call monotonicity           | Detects cross-strike inconsistencies         |
| Bounded Brent solver        | Prevents unconstrained IV explosions         |
| IV sanity bounds            | Restricts pathological volatility values     |
| Duplicate aggregation       | Stabilizes interpolation geometry            |
| Cubic interpolation         | Produces smooth surfaces                     |
| Nearest-neighbor completion | Prevents sparse blank regions                |
| 50 × 50 regularization      | Standardizes surface/risk visualization      |

---

# ⚠️ Data & Model Considerations

VolSurface Pro uses option-chain information obtained through `yfinance`. Consequently, analytical quality depends on the underlying market-data source.

Potential limitations include:

* Delayed or stale quotes
* Missing bid/ask values
* Incomplete open-interest data
* Irregular strike spacing
* Corporate actions
* Dividend assumptions
* American-style exercise for many equity options
* Simplified risk-free-rate treatment
* Market microstructure effects
* Bid/ask midpoint bias

The BSM framework used by the application assumes European-style exercise and does not constitute a complete production derivatives-pricing stack.

The resulting volatility surface should therefore be interpreted as an **analytical market-data representation**, rather than a guaranteed exchange-consistent arbitrage-free surface.

---

# 🔐 License

This project is distributed under the **MIT License**.

See [`LICENSE`](LICENSE) for the complete license text.

---

# 👨‍💻 Developer & Advisory

Developed as an institutional-style portfolio demonstration of:

* Quantitative financial engineering
* Derivatives analytics
* Numerical optimization
* Market-data engineering
* Python architecture
* Interactive quantitative visualization

---

**Developer:** Sudhir

**Project:** VolSurface Pro

**GitHub:** `@spelsudhir`

---

## 📚 Disclaimer

VolSurface Pro is provided for **educational, research, and portfolio demonstration purposes only**.

It is not investment, financial, trading, tax, or legal advice. Market data may be delayed, incomplete, inaccurate, or unavailable. Model outputs are dependent on assumptions and data quality and should not be interpreted as guarantees of market behavior or trading performance.

**Use responsibly.**
