# app.py
import streamlit as st
import pandas as pd
from dashboard.logic_bond import compute_bond_framework
from dashboard.logic_cds import compute_cds_basis_df
from dashboard.components import plot_hazard_curve, plot_cds_basis_curves
from dashboard.logic_portfolio import compute_portfolio_sensitivities

st.set_page_config(
    page_title="Risk Model Pricing Framework",
    layout="wide",
)

def main():
    # -------- PAGE TITLE --------
    st.title("Risk Model Pricing Framework")

    # -------- TOP BAR: DATABASE ELEMENT + BOND DROPDOWN --------
    top_left, top_right = st.columns([3, 1])

    with top_left:
        # Matches "Database element" label in your design
        st.markdown("#### Database element")

    with top_left:
        # Placeholder dropdown that *shows* UAL
        selected_bond = st.selectbox(
            "Bond Prices",
            options=["UAL"],
            index=0,
        )

    st.divider()

    # -------- COMPUTE DATA ONCE --------
    df, df_survival, df_schedule_with_rates, bond_pv, oas_result, P_mkt = compute_bond_framework()
    basis_df = compute_cds_basis_df()

    #-------- COMPUTE DATA Portfolio --------
    D_rate_i, D_hazard_i, VOD_i, MV_i, risk_vector_i = compute_portfolio_sensitivities()

    # Optional: store for reuse
    st.session_state["df"] = df
    st.session_state["df_survival"] = df_survival
    st.session_state["df_schedule_with_rates"] = df_schedule_with_rates
    st.session_state["bond_pv"] = bond_pv
    st.session_state["oas_result"] = oas_result
    st.session_state["P_mkt"] = P_mkt
    st.session_state["basis_df"] = basis_df

    # -------- SECOND ROW: BOND PRICE (LEFT) / CDS PRICE (RIGHT) --------
    col_bond, col_cds = st.columns(2)

    # ----- LEFT: BOND PRICE SECTION -----
    with col_bond:
        st.markdown("### Bond Price")
        #st.subheader("Hazard Curve with Exponential Spline Fit")

        # Graph area in your sketch -> hazard/survival curve
        fig_hazard = plot_hazard_curve(df)
        st.pyplot(fig_hazard, use_container_width=True)

        st.markdown("#### Bond Metrics")

        # Three lines: Risk Price, Market Price, OASF (OASF ~ your OAS)
        # Using columns makes it cleaner but still matches your three fields
        m1, m2, m3 = st.columns(3)

        with m1:
            st.markdown("**Risk Price (Model PV)**")
            st.write(f"{bond_pv:,.4f}")

        with m2:
            st.markdown("**Market Price**")
            st.write(f"{P_mkt:,.4f}")

        with m3:
            st.markdown("**OASF**")
            oas_results = pd.DataFrame([oas_result])
            st.write(f"{oas_results['oas_bps'].iloc[0]:.4f} bps")

    # ----- RIGHT: CDS PRICE SECTION -----
    with col_cds:
        st.markdown("### CDS Price")

        # Graph in your sketch on the CDS side: spread / basis curves
        fig_cds = plot_cds_basis_curves(
            basis_df,
            title="CDS Basis All Tenors",
            figsize=(10, 5),
        )
        st.pyplot(fig_cds, use_container_width=True)

        # Placeholder for "other info / key" area under the CDS graph in the sketch
        st.markdown("#### CDS Info")
        st.caption("Additional CDS metrics / keys can be added here later.")

    st.divider()

    # -------- THIRD ROW: PORTFOLIO CONSTRUCTION --------
    st.markdown("### Portfolio Construction")

    col_portfolio, col_portfolio_info = st.columns(2)

    # ----- LEFT: PORTFOLIO OUTPUTS -----
    with col_portfolio:
        st.markdown("#### Portfolio Outputs")

        p1, p2 = st.columns(2)

        with p1:
            st.markdown("**Market Value (Notional 1M)**")
            st.write(f"{MV_i:,.4f}")

            st.markdown("**IR Duration**")
            st.write(f"{D_rate_i:,.4f}")

        with p2:
            st.markdown("**Hazard Duration**")
            st.write(f"{D_hazard_i:,.4f}")

            st.markdown("**Value on Default**")
            st.write(f"{VOD_i:,.4f}")

    # ----- RIGHT: PORTFOLIO INFO / PLACEHOLDER -----
    with col_portfolio_info:
        st.markdown("#### Portfolio Info")
        st.caption("Space for Plotting")

if __name__ == "__main__":
    main()
