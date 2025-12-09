import pandas as pd
import numpy as np
from datetime import datetime
from scipy.optimize import curve_fit

def dates(x):
    
    for fmt in ("%m/%d/%Y","%m/%d/%y","%Y-%m-%d"):
        
        try: return datetime.strptime(str(x), fmt).date()
        except: pass
    return pd.NaT


def tnum(x):
    
    s = str(x).replace(",", "").strip()
    
    try: return float(s)
    except: return np.nan

def calculate_survival_probabilities(df, maturity_date, recovery_rate=0.4):
    """
    Calculate survival probabilities from CDS spread data using reduced-form model.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with 'date' and 'last_value' columns
    maturity_date : datetime.date
        Expected maturity date of the bond
    recovery_rate : float, optional
        Assumed recovery rate (default: 0.4)
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with survival probabilities and related metrics
    """
    # Parse dates and numeric values
    df = df.rename(columns = {'Date': 'date', 'Last': 'last_value'})
    df["date"] = df["date"].apply(dates)
    df["last_value"] = df["last_value"].apply(tnum)
    df = df.dropna(subset=["date", "last_value"]).sort_values("date").reset_index(drop=True)
    
    # Calculate spread in basis points and decimal
    df["spread_bps"] = df["last_value"]          
    df["spread_dec"] = df["spread_bps"] / 1e4
    
    # Calculate hazard rate assuming recovery rate
    df["hazard"] = df["spread_dec"] / (1 - recovery_rate)
    
    # Calculate time to maturity
    df["tau_yrs"] = (pd.to_datetime(maturity_date) - pd.to_datetime(df["date"])).dt.days / 365.0
    df = df[df["tau_yrs"] >= 0].copy()
    
    # Calculate survival probability under constant hazard to maturity
    df["s_tau"] = np.exp(-df["hazard"] * df["tau_yrs"])
    
    df.rename(columns ={'tau_yrs': 'Time_to_Maturity'}, inplace=True)
    df = df[['date', 'Time_to_Maturity', 'last_value', 'spread_bps', 'spread_dec', 'hazard', 's_tau']]

    return df