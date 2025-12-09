import numpy as np
from core.pricing.bond_pricing import price_with_spread

#Calculate Durations (Hazard and Interest Rate)

#Interest Rate Duration

def duration_oas(df_schedule, oas_decimal, recovery_rate, par, bump_bp=1.0):
    """
    OAS / interest-rate duration via bump-and-reprice.
    oas_decimal: calibrated OAS (decimal)
    bump_bp: bump in basis points (default 1 bp)
    """
    bump = bump_bp / 1e4  # 1 bp = 0.0001

    P0  = price_with_spread(df_schedule, oas_decimal,       recovery_rate, par)
    P_up = price_with_spread(df_schedule, oas_decimal + bump, recovery_rate, par)
    P_dn = price_with_spread(df_schedule, oas_decimal - bump, recovery_rate, par)

    # survival-based interest duration: - dP / (P dOAS)
    D_r = -(P_up - P_dn) / (2 * P0 * bump)
    return D_r, P0

#Hazard Rate Duration
def price_with_spread_and_hazard_bump(df_schedule, spread, recovery_rate, par, hazard_bump):
    """
    Price the bond with:
      - constant OAS 'spread' (decimal)
      - parallel hazard bump 'hazard_bump' (decimal per year)
    """
    t = df_schedule['t'].values
    Z0 = df_schedule['discount_factor'].values
    C = df_schedule['cashflow'].values
    Q_base = df_schedule['Q_fit_spline'].values

    # Spread-adjusted discount factors
    Z_adj = Z0 * np.exp(-spread * t)

    # Hazard-bumped survival probabilities: Q' = Q * exp(-Δh * t)
    Q_bump = Q_base * np.exp(-hazard_bump * t)

    # Previous survival probabilities under bumped hazard
    Q_prev_bump = np.roll(Q_bump, 1)
    Q_prev_bump[0] = 1.0

    pv_cash = Z_adj * C * Q_bump
    pv_rec  = Z_adj * recovery_rate * par * (Q_prev_bump - Q_bump)

    return (pv_cash + pv_rec).sum()


def duration_hazard(df_schedule, oas_decimal, recovery_rate, par, bump_bp=1.0):
    """
    Hazard-rate duration via parallel hazard bump.
    bump_bp: hazard bump in basis points (per year).
    """
    bump = bump_bp / 1e4  # 1 bp of hazard

    # Base price (no hazard bump)
    P0  = price_with_spread_and_hazard_bump(df_schedule, oas_decimal, recovery_rate, par, hazard_bump=0.0)
    P_up = price_with_spread_and_hazard_bump(df_schedule, oas_decimal, recovery_rate, par, hazard_bump=bump)
    P_dn = price_with_spread_and_hazard_bump(df_schedule, oas_decimal, recovery_rate, par, hazard_bump=-bump)

    D_h = -(P_up - P_dn) / (2 * P0 * bump)
    return D_h, P0

par = 100
def value_on_default(price_per_100, recovery_rate, par=100.0):
    """
    VOD = 1 - Recovery / Price_fraction
    """
    price_fraction = price_per_100 / par  # e.g. 80 / 100 = 0.80
    vod = 1.0 - recovery_rate / price_fraction
    return vod