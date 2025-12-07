# %%
#import libraries 

import pandas as pd, numpy as np
from datetime import datetime
import matplotlib.pyplot as plt

#========================================

#loading csv 

file = r"/Users/liamrodgers/Desktop/Python/FE800/FE800-Research/data/raw/NewUAL/UALPricingnew.csv"
df = pd.read_csv(file)



#keeping date to last in spread 

df = df.rename(columns={"Date":"date","Last":"last_value"}).loc[:,["date","last_value"]]

#========================================

#parse dates + numeric 

def dates(x):
    
    for fmt in ("%m/%d/%Y","%m/%d/%y","%Y-%m-%d"):
        
        try: return datetime.strptime(str(x), fmt).date()
        except: pass
    return pd.NaT


def tnum(x):
    
    s = str(x).replace(",", "").strip()
    
    try: return float(s)
    except: return np.nan

##Function
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
    
    return df
    

# Apply the function
expm = datetime(2026, 4, 11).date()
df = calculate_survival_probabilities(df, expm, recovery_rate=0.4)

#save as csv 

#saved = "/Users/anthonygiller/Downloads/UAL_reduced_form_survival.csv"
#df.to_csv("UAL_survival_to_maturity.csv", index=False)

#========================================

#plot survival prob 

plt.figure(figsize=(8,5))

plt.plot(df["date"], df["s_tau"])
plt.title("UAL Survival to Maturity (rate 4% w/ exp 04/11/2026)")
plt.xlabel("quote date") 
plt.ylabel("survival prob")

plt.xticks(rotation=45) 
plt.tight_layout() 
plt.show()


print(df.head(12))
print(df.columns)

# %%
df.rename(columns ={'tau_yrs': 'Time_to_Maturity'}, inplace=True)
df = df[['date', 'Time_to_Maturity', 'last_value', 'spread_bps', 'spread_dec', 'hazard', 's_tau']]

# %%
#Fitting the hazard rate curve with exponential spline

import numpy as np
from scipy.optimize import curve_fit

###Function
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

##Function
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


# Fit the model
params, covariance = fit_exponential_spline(df['Time_to_Maturity'], df['s_tau'])

print("Fitted parameters:", params)

df['Q_fit_spline'] = exp_spline(df['Time_to_Maturity'], *params)
df['Q_fit_spline'] = np.where(df['Q_fit_spline'] > 1.0, 1.0, df['Q_fit_spline'])

# %%
#Fitting Nelson-Siegel model
###Function
def nelson_siegel(t, beta0, beta1, beta2, lamb):
    t = np.maximum(t, 1e-6)
    return beta0 + beta1 * (1 - np.exp(-lamb * t)) / (lamb * t) + beta2 * ((1 - np.exp(-lamb * t)) / (lamb * t) - np.exp(-lamb * t))

p0 = [0.8, -0.2, -0.1, 0.5]
params_ns, _ = curve_fit(nelson_siegel, df['Time_to_Maturity'], df['s_tau'], p0=p0)
df['NS_fit'] = nelson_siegel(df['Time_to_Maturity'], *params_ns)

# %%
from scipy.interpolate import UnivariateSpline
df_sorted = df.sort_values('Time_to_Maturity')
cubic_spline = UnivariateSpline(df_sorted['Time_to_Maturity'], df_sorted['s_tau'], k=3)
df_sorted['cubic_fit'] = cubic_spline(df_sorted['Time_to_Maturity'])

# %%
from scipy.interpolate import interp1d
piecewise_fit = interp1d(df['Time_to_Maturity'], df['s_tau'], kind='linear', fill_value="extrapolate")
df['piecewise_fit'] = piecewise_fit(df['Time_to_Maturity'])

# %%
# Create fitted curve

plt.scatter(df['Time_to_Maturity'], df['s_tau'], label="Observed", alpha=0.7)
plt.plot(df['Time_to_Maturity'], exp_spline(df['Time_to_Maturity'], *params), label="Exponential Spline", color = 'red')
plt.plot(df['Time_to_Maturity'], df['NS_fit'], label="Nelson-Siegel")
#plt.plot(df['Time_to_Maturity'], df['cubic_fit'], label="Cubic Spline", linestyle="--")
plt.plot(df['Time_to_Maturity'], df['piecewise_fit'], label="Piecewise Spline")
plt.plot(df_sorted['Time_to_Maturity'], df_sorted['cubic_fit'], label="Cubic Spline")
plt.title('Modeling Survival Probabilities')
plt.xlabel("Time (years)")
plt.ylabel("Survival Probability")
plt.legend()
plt.show()


# %%
#Exp Spline plot
plt.figure(figsize=(10, 6))
plt.scatter(df['Time_to_Maturity'], df['s_tau'], label="Observed", alpha=0.7)
plt.plot(df['Time_to_Maturity'], exp_spline(df['Time_to_Maturity'], *params), label="Exponential Spline", color = 'red')
plt.xlabel("Time (years)")
plt.ylabel("Survival Probability")
plt.title("Exponential Spline Fit to Survival Probability")
plt.legend()
plt.show()

#Nelson-Siegel plot
plt.figure(figsize=(10, 6))
plt.scatter(df['Time_to_Maturity'], df['s_tau'], label="Observed", alpha=0.7)
plt.plot(df['Time_to_Maturity'], nelson_siegel(df['Time_to_Maturity'], *params_ns), label="Nelson-Siegel", color = 'green')
plt.xlabel("Time (years)")
plt.ylabel("Survival Probability")
plt.title("Nelson-Siegel Fit to Survival Probability")
plt.legend()
plt.show()

#Piecewise plot
plt.figure(figsize=(10, 6))
plt.scatter(df['Time_to_Maturity'], df['s_tau'], label="Observed", alpha=0.7)
plt.plot(df['Time_to_Maturity'], df['piecewise_fit'], label="Piecewise Spline", color = 'purple')
plt.xlabel("Time (years)")
plt.ylabel("Survival Probability")
plt.title("Piecewise Spline Fit to Survival Probability")
plt.legend()
plt.show()

#Cublic Spline plot
plt.figure(figsize=(10, 6))
plt.scatter(df['Time_to_Maturity'], df['s_tau'], label="Observed", alpha=0.7)
plt.plot(df_sorted['Time_to_Maturity'], df_sorted['cubic_fit'], label="Cubic Spline")
plt.xlabel("Time (years)")
plt.ylabel("Survival Probability")
plt.title("Cubic Spline Fit to Survival Probability")
plt.legend()
plt.show()

# %%
df_survival = df.loc[:, ['date', 's_tau', 'Q_fit_spline']]
df_survival['date'] = pd.to_datetime(df_survival['date'])
df_survival = df_survival.resample('ME', on='date').mean()
df_survival.reset_index(inplace=True)

# %%
#Bond information
from datetime import datetime
##Function
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


#Usage
coupon_freq = 2 # semi-annual
coupon_rate = 0.04
issue_date = datetime(2014, 7, 28)
first_coupon_date = datetime(2015, 4, 1)
par = 100
maturity = datetime(2026, 11, 4)

df_schedule = generate_bond_schedule(par, coupon_rate, coupon_freq, first_coupon_date, maturity)

# %%
from scipy.interpolate import interp1d
##Function
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


Q_curve = create_survival_curve(df)

# %%
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


# Pull yields from FRED
tenors = np.array([0.25, 0.5, 1, 2, 3, 5, 7, 10])
#rates = np.array([0.03, 0.12, 0.27, 0.56, 0.87, 1.35, 1.67, 1.90])/100 #2015-04-01
#rates = np.array([0.09, 0.05, 0.17, 0.20, 0.24, 0.37, 0.51, 0.62])/100 #2020-04-29
#rates = np.array([0.223, 0.214, 0.198, 0.187, 0.182, 0.184, 0.193, 0.204])/100 #2019-07-05
rates = np.array([0.547, 0.543, 0.517, 0.487, 0.465, 0.452, 0.451, 0.450])/100 #2024-05-10

print("Tenors (years):", tenors)
print(f'Interest Rates: {rates}')

# Fit yield curve
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

# %%
plt.scatter(tenors, rates*100, label='Observed Yields', linestyle='--', color='blue')
plt.plot(tenors, nelson_siegel(tenors, *params_ns_rates)*100, label='Nelson-Siegel Fit', color='red')
plt.xlabel('Tenor (Years)')
plt.ylabel('Yield (%)')
plt.title('Observed Treasury Yields')
plt.legend()
plt.grid()
plt.show()
print(f"Nelson Siegel Rate Estimations: {nelson_siegel(tenors, *params_ns_rates)*100}")

# %%
df_schedule_with_rates['Q_fit_spline'] = df_schedule_with_rates['Q_fit_spline'].replace(0, np.nan).interpolate(method='linear') #only replaces Nan Values not 0s

# %%
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


# Price the bond
Recovery = 0.40
df_schedule_with_rates, bond_pv = price_bond_with_default_risk(
    df_schedule_with_rates,
    df_schedule_with_rates['discount_factor'].values,
    df_schedule_with_rates['Q_fit_spline'].values,
    Recovery,
    par
)

print(f'Bond PV: {bond_pv:.2f}')

# %%
"""
Next Steps: 
2. Add a dynamic FRED rate grabber via API keys so we can auto adjust dates
3. Get more bond and CDS data to see how we fit the whole term structure
4. Design and application for this?
 - Using streamlit or similar to make a bond pricing app?



"""

# %%
#Option Adjusted Spread Calibration
from scipy.optimize import brentq

def calibrate_oas(df_schedule, market_price, recovery_rate, par, search_range=(-0.20, 0.20)):
    """
    Calibrate option-adjusted spread (OAS) to match market price.
    
    Parameters:
    -----------
    df_schedule : pd.DataFrame
        Bond schedule with columns: t, discount_factor, cashflow, Q_fit_spline, Q_previous
    market_price : float
        Observed market price of the bond
    recovery_rate : float
        Recovery rate on default
    par : float
        Par value of the bond
    search_range : tuple
        (min_spread, max_spread) in decimal form
    
    Returns:
    --------
    dict
        Dictionary with 'oas_decimal' and 'oas_bps' keys
    """
    df_cf = df_schedule.copy()
    
    def price_with_spread(s):
        """
        Price the bond under an additional constant spread s (decimal).
        s > 0 => higher discounting => lower price.
        """
        t = df_cf['t'].values
        Z0 = df_cf['discount_factor'].values
        C = df_cf['cashflow'].values
        Q = df_cf['Q_fit_spline'].values
        Q_prev = df_cf['Q_previous'].values
        
        # Adjusted discount factors: Z_i(s) = Z_i * exp(-s * t_i)
        Z_adj = Z0 * np.exp(-s * t)
        
        pv_cash = Z_adj * C * Q
        pv_rec = Z_adj * recovery_rate * par * (Q_prev - Q)
        
        return (pv_cash + pv_rec).sum()
    
    def objective(s):
        return price_with_spread(s) - market_price
    
    oas_decimal = brentq(objective, search_range[0], search_range[1])
    
    return {
        'oas_decimal': oas_decimal,
        'oas_bps': oas_decimal * 1e4
    }


# Calibrate OAS to market price
#P_mkt = 116.248 #At Observation Date 2015-04-07
#P_mkt = 76.2 
#P_mkt = 103.89
P_mkt = 126.83 #2024-05-10

Recovery = 0.40
par = 100.0

oas_result = calibrate_oas(df_schedule_with_rates, P_mkt, Recovery, par)

print(f"OASF (decimal): {oas_result['oas_decimal']:.6f}")
print(f"OASF: {oas_result['oas_bps']:.2f} bps")


