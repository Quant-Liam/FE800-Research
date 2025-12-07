import streamlit as st
from dashboard.components import plot_cds_basis_curves
# app.py
import streamlit as st
from dashboard.logic_cds import compute_cds_basis_df
from dashboard.components import plot_hazard_curve
from dashboard.logic_bond import compute_bond_framework  # the function above

st.set_page_config(
    page_title="Risk Model Pricing Framework",
    layout="wide",
)

def main():
    st.title("Risk Model Pricing Framework")

    # --- BOND PRICING SECTION ---
    st.header("Bond Pricing Framework")

    # compute everything
    df, df_survival, df_schedule_with_rates, bond_pv, oas_result, P_mkt = compute_bond_framework()

    # optional: store in session_state for later reuse
    st.session_state["df"] = df
    st.session_state["df_survival"] = df_survival
    st.session_state["df_schedule_with_rates"] = df_schedule_with_rates
    st.session_state["bond_pv"] = bond_pv
    st.session_state["oas_result"] = oas_result
    
    st.write('Bond Market Price')
    st.write(P_mkt)
    st.subheader("Model Bond PV")
    st.write(bond_pv)
    st.subheader("Calibrated OAS")
    st.write(oas_result)


    # --- CDS SECTION (if you already have this wired) ---
    basis_df = compute_cds_basis_df()
    fig = plot_cds_basis_curves(basis_df, title="CDS Basis All Tenors", figsize=(12, 6))
    st.pyplot(fig)

    st.divider()

    
if __name__ == "__main__":
    main()
