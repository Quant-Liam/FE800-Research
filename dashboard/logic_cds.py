import streamlit as st
import pandas as pd
import numpy as np
from core.pricing.cds_pricing import (load_cds_data,Bond_CDS_matching, merge_all_cds_bond_data, 
generate_payment_dates, bootstrap_all_hazards, construct_survival_curve, 
setup_payment_dates_and_segments, calculate_all_basis_across_dates)
from dashboard.components import plot_cds_basis_curves
from datetime import datetime

def compute_cds_basis_df():
    # Loading data
    (df_1y, df_5y, df_10y, df_risk_free, df_term_structure, df_bond_price,
    year_start_date, year_end_date) = load_cds_data(
        r'/Users/liamrodgers/Desktop/Python/FE800/FE800-Research/data/raw/CDSdata/UAL_1Y_CDS_prices.csv',
        r'/Users/liamrodgers/Desktop/Python/FE800/FE800-Research/data/raw/CDSdata/UAL_5Y_CDS_2014FullData.csv',
        r'/Users/liamrodgers/Desktop/Python/FE800/FE800-Research/data/raw/CDSdata/UAL_10Y_CDS_prices.csv',
        r'/Users/liamrodgers/Desktop/Python/FE800/FE800-Research/data/raw/CDSdata/risk_free_curve_2015.csv',
        r'/Users/liamrodgers/Desktop/Python/FE800/FE800-Research/data/raw/CDSdata/CDS_term_structure_UAL.csv',
        r'/Users/liamrodgers/Desktop/Python/FE800/FE800-Research/data/raw/CDSdata/UAL_hazard_rates.csv'
    )

    # Assign date ranges (all tenors use same date range in this case)
    Y1_end_date, Y5_end_date, Y10_end_date = year_end_date, year_end_date, year_end_date
    Y1_start_date, Y5_start_date, Y10_start_date = year_start_date, year_start_date, year_start_date

    ##Price Matching##

    df_1y = Bond_CDS_matching(df_1y, df_bond_price)
    df_5y = Bond_CDS_matching(df_5y, df_bond_price) #Data issue
    df_10y = Bond_CDS_matching(df_10y, df_bond_price)

    #Merging
    df_CDS_Bond = merge_all_cds_bond_data(df_bond_price, df_1y, df_5y, df_10y)

    # Generate payment dates for each tenor
    bond_maturity = datetime(2026, 4, 11)
    payment_dates = generate_payment_dates(Y1_start_date, bond_maturity, 3)
    h_1y, h_5y, h_10y = bootstrap_all_hazards(df_1y, df_5y, df_10y, payment_dates, df_risk_free, Y1_end_date, 0.4)
    y1_bond_hazard = df_bond_price.loc[Y1_end_date, 'hazard']


    # Setup all tenor configurations
    tenor_config = setup_payment_dates_and_segments(
        Y1_start_date, Y1_end_date,
        Y5_start_date, Y5_end_date,
        Y10_start_date, Y10_end_date,
        h_1y, h_5y, h_10y
    )

    payment_dates_1y = tenor_config['payment_dates_1y']
    payment_dates_5y = tenor_config['payment_dates_5y']
    payment_dates_10y = tenor_config['payment_dates_10y']
    segment_starts = tenor_config['segment_starts']
    hazard_rates = tenor_config['hazard_rates']
    all_payment_dates = tenor_config['all_payment_dates']
    survival_probs = construct_survival_curve(all_payment_dates, hazard_rates, segment_starts)

    # Define tenor configurations (payment dates and market column names)
    tenor_configs_dict = {
        '1Y': (payment_dates_1y, 'PX_LAST'),
        '5Y': (payment_dates_5y, 'PX_LAST_CDS_5Y'),
        '10Y': (payment_dates_10y, 'PX_LAST_CDS_10Y')
    }

    # Calculate all bases
    basis_df = calculate_all_basis_across_dates(
        df_CDS_Bond, tenor_configs_dict, segment_starts,
        all_payment_dates, df_risk_free, 0.4
    )

    # Format the DataFrame
    basis_df['Date'] = pd.to_datetime(basis_df['Date'])

    for tenor in ['1Y', '5Y', '10Y']:
        basis_df[f'CDS_Basis_{tenor}'] = round(basis_df[f'CDS_Basis_{tenor}'], 3)

    return basis_df