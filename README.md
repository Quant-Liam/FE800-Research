# Reduced-Form Credit Risk Modeling for Corporate Bonds

This repository contains the code for an FE800 research project on **reduced-form modeling of credit risk for corporate bonds**, with a primary case study on **United Airlines (UAL)** during and after the COVID-19 credit stress period.

The project reworks traditional bond valuation by explicitly incorporating **survival probabilities** and **hazard rates** into pricing, then connects that framework to **CDS basis analysis** and a simple **hedged portfolio construction workflow**. The implementation is paired with a Streamlit interface that surfaces the model outputs visually.

## Project Motivation

Traditional corporate bond pricing can miss the effect of issuer default risk, especially during periods of financial stress. This project follows the reduced-form framework developed in the Arthur M. Berd, Roy Mashal, and Peili Wang credit term structure papers and applies it to stressed airline credit.

The core idea is simple:

- estimate issuer default intensity from market-observed spreads
- convert that into survival probabilities over the life of the bond
- use those survival probabilities to price risky bond cash flows and recovery cash flows
- calibrate an **Option-Adjusted Spread to Fit (OASF)** to compare model value against market value
- compare bond-implied CDS spreads against market CDS spreads to identify basis dislocations

## Research Goals

The project focuses on three main questions:

1. Can hazard rates inferred from bonds and CDS markets be used consistently in a reduced-form framework?
2. Which interpolation methods best capture the survival curve across the bond's life?
3. Can the pricing outputs support an arbitrage-oriented, hedged bond/CDS portfolio construction process?

## What This Repository Implements

### 1. Bond Pricing Framework

The bond pricing workflow is implemented in [`core/pricing/bond_pricing.py`](./core/pricing/bond_pricing.py) and [`dashboard/logic_bond.py`](./dashboard/logic_bond.py).

It performs the following steps:

- loads UAL bond price history
- converts observed spreads into hazard-rate proxies under a constant 40% recovery assumption
- computes survival probabilities to maturity
- fits an **exponential spline** survival curve
- fits a **Nelson-Siegel** curve for the risk-free term structure
- generates the bond cash flow schedule
- prices the bond as the sum of:
  - discounted cash flows weighted by survival probability
  - discounted recovery cash flows weighted by default probability
- calibrates an **OASF** using Brent's root-finding method

For the valuation date used in the app (`2024-05-10`), the current implementation produces:

- **Model price (risk price):** `103.94`
- **Observed market price:** `126.83`
- **OASF:** `-846.51 bps`

Under the report's interpretation, this indicates the bond is trading **rich/overpriced** relative to the reduced-form model, which supports a **short bond / long CDS** relative-value view.

### 2. CDS Pricing and Basis Framework

The CDS framework is implemented in [`core/pricing/cds_pricing.py`](./core/pricing/cds_pricing.py) and [`dashboard/logic_cds.py`](./dashboard/logic_cds.py).

This part of the project:

- loads UAL CDS market data for the **1Y, 5Y, and 10Y** tenors
- loads a risk-free curve used for discounting
- bootstraps piecewise hazard rates from CDS spreads
- constructs survival curves across payment dates
- computes **bond-implied CDS spreads**
- measures the **CDS-bond basis** as:

`bond-implied CDS spread - market CDS spread`

The Streamlit app plots the historical basis for the 1Y, 5Y, and 10Y tenors so the user can inspect where spreads deviate from theoretical parity.

### 3. Portfolio Risk Outputs

Portfolio metrics are implemented in [`core/pricing/portfolio_construction.py`](./core/pricing/portfolio_construction.py) and [`dashboard/logic_portfolio.py`](./dashboard/logic_portfolio.py).

The portfolio section computes:

- **Market value**
- **Interest-rate duration**
- **Hazard-rate duration**
- **Value on default (VOD)**

Using the current UAL example in the app:

- **Market value on $1,000,000 notional:** `1,268,200`
- **Interest-rate duration:** `2.3598`
- **Hazard duration:** `1.5182`
- **Value on default:** `0.6846`

These metrics are used to frame hedging decisions around spread risk, hazard risk, and jump-to-default exposure.

## Application Interface

The repository includes a Streamlit dashboard in [`app.py`](./app.py) that brings the research workflow into a single interface. The application displays:

- a UAL bond selector placeholder
- the fitted survival curve
- bond pricing outputs
- CDS basis curves across tenors
- portfolio construction outputs

The dashboard logic is split across:

- [`dashboard/logic_bond.py`](./dashboard/logic_bond.py)
- [`dashboard/logic_cds.py`](./dashboard/logic_cds.py)
- [`dashboard/logic_portfolio.py`](./dashboard/logic_portfolio.py)
- [`dashboard/components.py`](./dashboard/components.py)

## Data Used

The repository currently includes local datasets under [`data/raw`](./data/raw):

- **UAL bond pricing data**
- **UAL CDS data** for 1Y, 5Y, and 10Y maturities
- **risk-free curve data**
- intermediate hazard/survival outputs used in the pricing workflow

The report also discusses testing on **Delta Airlines (DAL)**, but the operational dashboard in this repository is centered on the **UAL** example.

## Repository Structure

```text
FE800-Research/
├── app.py                          # Streamlit application entry point
├── core/
│   ├── data/                       # Data loaders and helper utilities
│   └── pricing/                    # Bond, CDS, hazard, and portfolio models
├── dashboard/                      # App logic and plotting components
├── data/raw/                       # Local bond, CDS, and curve datasets
├── CompletedPOC/                   # Earlier proof-of-concept notebooks/scripts
├── Testingfiles/                   # Draft notebooks and exploratory work
└── db/                             # Database initialization stub
```

## Methods Summary

The implementation follows the same flow described in the final report:

1. Estimate hazard rates from observed market spreads.
2. Convert hazard rates into survival probabilities.
3. Fit the survival term structure using an exponential spline.
4. Fit the interest-rate curve with Nelson-Siegel.
5. Price the bond under survival-weighted cash flows and recovery flows.
6. Calibrate OASF to compare the reduced-form value to the observed market price.
7. Bootstrap CDS hazard rates and compute CDS-bond basis across maturities.
8. Translate valuation outputs into portfolio hedge metrics.

## How To Run

### Requirements

This project uses:

- `python`
- `streamlit`
- `pandas`
- `numpy`
- `scipy`
- `matplotlib`

Install the libraries in your environment, then run:

```bash
cd /Users/liamrodgers/Desktop/Python/FE800/FE800-Research
streamlit run app.py
```

## Current Notes and Limitations

- The app is currently configured around a single issuer workflow for **UAL**.
- Several paths in the app logic are hard-coded to local project files.
- The dropdown in the interface is a placeholder with `UAL` as the active option.
- The repository includes exploratory notebooks and proof-of-concept work in addition to the main app code.
- `Dockerfile` and `docker-compose.yml` are present but currently unconfigured.

## Key Takeaway

This project shows how a **reduced-form credit framework** can produce an intrinsic, risk-adjusted corporate bond value that differs materially from observed market prices during stress periods. By combining bond pricing, CDS basis analysis, and portfolio hedge metrics, the repository turns the research report into a working prototype for identifying **relative-value and arbitrage opportunities** in credit markets.

## References

- Berd, Arthur M., Roy Mashal, and Peili Wang. *Defining, Estimating and Using Credit Term Structures. Part 1: Consistent Valuation Measures.*
- Berd, Arthur M., Roy Mashal, and Peili Wang. *Defining, Estimating and Using Credit Term Structures. Part 2: Consistent Risk Measures.*
- Berd, Arthur M., Roy Mashal, and Peili Wang. *Defining, Estimating and Using Credit Term Structures. Part 3: Consistent CDS-Bond Basis.*
