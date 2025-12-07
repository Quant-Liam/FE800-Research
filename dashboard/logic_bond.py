# dashboard/layout.py
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
from scipy.optimize import curve_fit
from core.pricing.hazard_models import calculate_survival_probabilities
from core.pricing.bond_pricing import fit_exponential_spline, exp_spline, nelson_siegel, generate_bond_schedule, calibrate_oas
from core.pricing.bond_pricing import create_survival_curve, fit_yield_curve, calculate_discount_factors, price_bond_with_default_risk 
from dashboard.components import plot_hazard_curve, plot_cds_curve_placeholder, render_portfolio_placeholder  

def compute_bond_framework():
    ###Bond Framework###
    df = pd.read_csv(r"/Users/liamrodgers/Desktop/Python/FE800/FE800-Research/data/raw/NewUAL/UALPricingnew.csv")
    expm = datetime(2026, 4, 11).date()
    #Going to need a dataloader for the database item, but this works for now

    df = calculate_survival_probabilities(df, expm, recovery_rate=0.4)
    ####Plot the hazard rate curve####### Need to write this in later, I want to get outputs done first

    ####Fitting the spline###
    params, covariance = fit_exponential_spline(df['Time_to_Maturity'], df['s_tau'])

    df['Q_fit_spline'] = exp_spline(df['Time_to_Maturity'], *params)
    df['Q_fit_spline'] = np.where(df['Q_fit_spline'] > 1.0, 1.0, df['Q_fit_spline'])

    ########Fitting the Nelson_Sigle Curve####
    p0 = [0.8, -0.2, -0.1, 0.5]
    params_ns, _ = curve_fit(nelson_siegel, df['Time_to_Maturity'], df['s_tau'], p0=p0)
    df['NS_fit'] = nelson_siegel(df['Time_to_Maturity'], *params_ns)

    ####Changing over to the survival df#####
    df_survival = df.loc[:, ['date', 's_tau', 'Q_fit_spline']]
    df_survival['date'] = pd.to_datetime(df_survival['date'])
    df_survival = df_survival.resample('ME', on='date').mean()
    df_survival.reset_index(inplace=True)

    #####Adding the Schedueling#####

    #Usage### These we want to add as inputs
    coupon_freq = 2 # semi-annual
    coupon_rate = 0.04
    issue_date = datetime(2014, 7, 28)
    first_coupon_date = datetime(2015, 4, 1)
    par = 100
    maturity = datetime(2026, 11, 4)

    df_schedule = generate_bond_schedule(par, coupon_rate, coupon_freq, first_coupon_date, maturity)

    Q_curve = create_survival_curve(df)

    ######Adding the interesting rates######
    tenors = np.array([0.25, 0.5, 1, 2, 3, 5, 7, 10])
    rates = np.array([0.547, 0.543, 0.517, 0.487, 0.465, 0.452, 0.451, 0.450])/100 #2024-05-10
    params_ns_rates = fit_yield_curve(tenors, rates)

    # Apply to schedule
    pricing_date = datetime(2024, 5, 10) #Pricing Observation Date

    df_schedule_with_rates = df_schedule.copy()
    df_schedule_with_rates['date'] = pd.to_datetime(df_schedule_with_rates['date'])

    df_schedule_with_rates['t'] = (
        df_schedule_with_rates['date'] - pricing_date
    ).dt.days / 365.0

    df_schedule_with_rates = df_schedule_with_rates[df_schedule_with_rates['t'] > 0]

    df_schedule_with_rates['Q_fit_spline'] = Q_curve(df_schedule_with_rates['t'])
    df_schedule_with_rates['risk_free_rate'] = nelson_siegel(df_schedule_with_rates['t'], *params_ns_rates)
    df_schedule_with_rates['discount_factor'] = calculate_discount_factors(df_schedule_with_rates['t'], params_ns_rates)

    df_schedule_with_rates['Q_fit_spline'] = df_schedule_with_rates['Q_fit_spline'].replace(0, np.nan).interpolate(method='linear') #only replaces Nan Values not 0s


    Recovery = 0.40
    df_schedule_with_rates, bond_pv = price_bond_with_default_risk(
        df_schedule_with_rates,
        df_schedule_with_rates['discount_factor'].values,
        df_schedule_with_rates['Q_fit_spline'].values,
        Recovery,
        par
    )


    ###Calculating the option adjusted spread###
    P_mkt = 126.83 #2024-05-10
    oas_result = calibrate_oas(df_schedule_with_rates, P_mkt, Recovery, par)


    return df, df_survival, df_schedule_with_rates, bond_pv, oas_result, P_mkt
