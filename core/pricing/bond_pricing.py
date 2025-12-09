import numpy as np
import pandas as pd
from datetime import datetime
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d
from scipy.optimize import brentq
from typing import Dict, Any
from core.pricing.hazard_models import calculate_survival_probabilities



def exp_spline(t, a1, a2, l1, l2, l3): 
    """
    Survival curve:
        Q(t) = beta1 * exp(-l1 t) + beta2 * exp(-l2 t) + beta3 * exp(-l3 t)
    with:
        beta_k >= 0 and beta1 + beta2 + beta3 = 1

    We get (beta1, beta2, beta3) from unconstrained (a1, a2) via a softmax-like map.
    """
    # map (a1, a2) -> (beta1, beta2, beta3) on the simplex
    #Ensures that the beta parameters are constrained to sum to 1
    e1, e2 = np.exp(a1), np.exp(a2) 
    Z = 1.0 + e1 + e2
    beta1 = e1 / Z
    beta2 = e2 / Z
    beta3 = 1.0 / Z
    return beta1 * np.exp(-l1 * t) + beta2 * np.exp(-l2 * t) + beta3 * np.exp(-l3 * t) #Equation for splining


def fit_exponential_spline(time_to_maturity, survival_prob, p0=None):
    """
    Fit exponential spline to survival probability data.
    
    Parameters:
    -----------
    time_to_maturity : array-like
        Time to maturity values
    survival_prob : array-like
        Survival probability values
    p0 : list, optional
        Initial parameter guesses [a1, a2, l1, l2, l3]
    
    Returns:
    --------
    tuple
        (fitted_params, covariance_matrix)
    """
    if p0 is None:
        p0 = [0.0, 0.0, 0.2, 0.05, 0.01]
    
    params, covariance = curve_fit(
        exp_spline, time_to_maturity, survival_prob, p0=p0,
        bounds=(
            [-np.inf, -np.inf, 0.001, 0.001, 0.001],
            [ np.inf,  np.inf, 5.0,   1.0,   0.5]
        )
    )
    
    return params, covariance

def nelson_siegel(t, beta0, beta1, beta2, lamb):
    t = np.maximum(t, 1e-6)
    return beta0 + beta1 * (1 - np.exp(-lamb * t)) / (lamb * t) + beta2 * ((1 - np.exp(-lamb * t)) / (lamb * t) - np.exp(-lamb * t))


def generate_bond_schedule(par, coupon_rate, coupon_freq, first_coupon_date, maturity_date):
    """
    Generate bond cashflow schedule with coupon payments and principal.
    
    Parameters:
    -----------
    par : float
        Par value of the bond
    coupon_rate : float
        Annual coupon rate (e.g., 0.04 for 4%)
    coupon_freq : int
        Coupon frequency per year (e.g., 2 for semi-annual)
    first_coupon_date : datetime
        Date of first coupon payment
    maturity_date : datetime
        Bond maturity date
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with columns: date, cashflow, coupon, principal
    """
    coupon_amount = par * (coupon_rate / coupon_freq)
    
    # Generate payment dates
    freq_map = {1: '12MS', 2: '6MS', 4: '3MS', 12: 'MS'}
    freq_str = freq_map.get(coupon_freq, '6MS')
    
    dates = pd.date_range(start=first_coupon_date, end=maturity_date, freq=freq_str).tolist()
    if dates[-1] < maturity_date:
        dates.append(maturity_date)
    
    schedule = []
    for d in dates:
        cashflow = coupon_amount
        principal = 0
        if d == maturity_date:
            cashflow += par
            principal = par
        schedule.append({
            "date": d,
            "cashflow": cashflow,
            "coupon": coupon_amount,
            "principal": principal
        })
    
    return pd.DataFrame(schedule)

def create_survival_curve(df, time_col='Time_to_Maturity', q_col='Q_fit_spline'):
    """
    Create interpolated survival probability curve from fitted data.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame containing survival data
    time_col : str
        Column name for time to maturity
    q_col : str
        Column name for survival probabilities
    
    Returns:
    --------
    callable
        Interpolation function Q(t)
    """
    df_Q = (
        df[[time_col, q_col]]
        .dropna()
        .sort_values(time_col)
        .drop_duplicates(time_col)
    )
    
    # Enforce monotone decreasing survival
    df_Q[q_col] = df_Q[q_col].cummin()
    
    Q_curve = interp1d(
        df_Q[time_col],
        df_Q[q_col],
        kind='linear',
        fill_value='extrapolate'
    )
    
    return Q_curve

# Tenors (in years) and yields (in decimals)
###############################################
##Function
def fit_yield_curve(tenors, rates, p0=None):
    """
    Fit Nelson-Siegel model to Treasury yield curve.
    
    Parameters:
    -----------
    tenors : array-like
        Tenor values in years
    rates : array-like
        Yield rates (decimal form)
    p0 : list, optional
        Initial parameter guesses [beta0, beta1, beta2, lambda]
    
    Returns:
    --------
    array
        Fitted Nelson-Siegel parameters
    """
    if p0 is None:
        p0 = [0.01, -0.01, 0.005, 1.0]
    
    params, _ = curve_fit(
        nelson_siegel,
        tenors,
        rates,
        p0=p0
    )
    
    return params


def calculate_discount_factors(time_values, yield_curve_params):
    """
    Calculate discount factors from Nelson-Siegel yield curve.
    
    Parameters:
    -----------
    time_values : array-like
        Time points in years
    yield_curve_params : array-like
        Nelson-Siegel parameters [beta0, beta1, beta2, lambda]
    
    Returns:
    --------
    array
        Discount factors Z(t) = exp(-r(t) * t)
    """
    rates = nelson_siegel(time_values, *yield_curve_params)
    discount_factors = np.exp(-rates * time_values)
    return discount_factors

def price_bond_with_default_risk(df_schedule, discount_factors, survival_probs, recovery_rate, par):
    """
    Price a bond incorporating default risk and recovery.
    
    Parameters:
    -----------
    df_schedule : pd.DataFrame
        Bond payment schedule with 'cashflow' column
    discount_factors : array-like
        Risk-free discount factors Z(t)
    survival_probs : array-like
        Survival probabilities Q(t)
    recovery_rate : float
        Recovery rate on default (e.g., 0.40)
    par : float
        Par value of the bond
    
    Returns:
    --------
    pd.DataFrame
        Schedule with PV calculations
    float
        Total bond present value
    """
    df = df_schedule.copy()
    df['discount_factor'] = discount_factors
    df['Q_fit_spline'] = survival_probs
    df['PV'] = 0.0
    
    # Add Q_previous column (Q(0,t_{i-1}))
    df['Q_previous'] = df['Q_fit_spline'].shift(1).fillna(1.0)
    
    for i in range(len(df)):
        z = df['discount_factor'].iloc[i]
        cash = df['cashflow'].iloc[i]
        Q_current = df['Q_fit_spline'].iloc[i]
        Q_previous = df['Q_previous'].iloc[i]
        
        # PV of cashflow if no default
        pv_cash = z * cash * Q_current
        
        # PV of recovery if default occurs in this period
        pv_recovery = z * recovery_rate * par * (Q_previous - Q_current)
        
        df['PV'].iloc[i] = pv_cash + pv_recovery
    
    total_pv = df['PV'].sum()
    
    return df, total_pv

#Option Adjusted Spread Calibration
def price_with_spread(df_schedule, spread, recovery_rate, par):
    """
    Price the bond under a constant OAS 'spread' (decimal).
    Uses existing columns: t, discount_factor, cashflow, Q_fit_spline, Q_previous.
    """
    t = df_schedule['t'].values
    Z0 = df_schedule['discount_factor'].values
    C = df_schedule['cashflow'].values
    Q = df_schedule['Q_fit_spline'].values
    Q_prev = df_schedule['Q_previous'].values

    Z_adj = Z0 * np.exp(-spread * t)

    pv_cash = Z_adj * C * Q
    pv_rec = Z_adj * recovery_rate * par * (Q_prev - Q)

    return (pv_cash + pv_rec).sum()


def calibrate_oas(df_schedule, market_price, recovery_rate, par, search_range=(-0.20, 0.20)):
    df_cf = df_schedule.copy()

    def objective(s):
        return price_with_spread(df_cf, s, recovery_rate, par) - market_price
    
    oas_decimal = brentq(objective, search_range[0], search_range[1])
    
    return {
        'oas_decimal': oas_decimal,
        'oas_bps': oas_decimal * 1e4
    }

