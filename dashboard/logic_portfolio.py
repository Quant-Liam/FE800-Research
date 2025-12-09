import numpy as np
from core.pricing.portfolio_construction import duration_oas, price_with_spread_and_hazard_bump, duration_hazard, value_on_default
from dashboard.logic_bond import compute_bond_framework
def compute_portfolio_sensitivities():

    par = 100 #Change depending on bond
    Recovery = 0.4 #Recovery Rate
    P0 = 126.82 #The actual market price of the bond
    notional = 1000000  # 1 million USD notional, this can be adjust depending on the portfolio
    _, _, df_schedule_with_rates, _, oasf_result, _ = compute_bond_framework()
        #Build Sensitivity Matrix
    oasf = oasf_result['oas_decimal']

    D_rate_i, P0_rate = duration_oas(
        df_schedule_with_rates,
        oasf,
        Recovery,
        par,
        bump_bp=1.0
    )

    D_hazard_i, P0_hazard = duration_hazard(
        df_schedule_with_rates,
        oasf,
        Recovery,
        par,
        bump_bp=1.0
    )

    VOD_i = value_on_default(P0, Recovery, par=par)

    notional = 1_000_000  # or whatever
    MV_i = P0 / 100.0 * notional #Portfolio Size

    risk_vector_i = np.array([
    MV_i,
    MV_i * D_rate_i,
    MV_i * D_hazard_i,
    MV_i * VOD_i   # optional (only if using default-neutral framework)
    ])

    return D_rate_i, D_hazard_i, VOD_i, MV_i, risk_vector_i


