import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.dates as mdates
def plot_hazard_curve(hazard_df: pd.DataFrame | None):
    """
    Plot survival points and precomputed spline curve on Streamlit.
    
    hazard_df must contain:
        - Time_to_Maturity
        - s_tau
        - spline_val   (your precomputed spline values)
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    if hazard_df is None or hazard_df.empty:
        ax.text(
            0.5, 0.5,
            "Survival curve/spline will appear here.",
            ha="center", va="center", fontsize=10
        )
        ax.set_axis_off()
    else:
        # Observed survival points
        ax.scatter(
            hazard_df["Time_to_Maturity"],
            hazard_df["s_tau"],
            label="Observed",
            alpha=0.7
        )

        # Precomputed spline line
        if "Q_fit_spline" in hazard_df.columns:
            ax.plot(
                hazard_df["Time_to_Maturity"],
                hazard_df["Q_fit_spline"],
                label="Exponential Spline",
                color="red"
            )
        else:
            ax.text(
                0.5, 0.1,
                "WARNING: Q_fit_spline column missing.",
                ha="center",
                fontsize=9,
                color="red",
                transform=ax.transAxes
            )

        # Titles & labels to match original style
        ax.set_xlabel("Time (years)")
        ax.set_ylabel("Survival Probability")
        ax.set_title("Exponential Spline Fit to Survival Probability")
        ax.legend()

        fig.tight_layout()

    st.pyplot(fig)





def plot_cds_curve_placeholder(cds_df: pd.DataFrame | None):
    """
    Placeholder for the CDS tenor spread graph.
    """
    fig, ax = plt.subplots()

    if cds_df is None or cds_df.empty:
        ax.text(
            0.5,
            0.5,
            "CDS tenor spread curve will appear here.",
            ha="center",
            va="center",
            fontsize=10,
        )
        ax.set_axis_off()
    else:
        # Example expected columns: ["tenor", "spread_bps"]
        ax.plot(cds_df["tenor"], cds_df["spread_bps"])
        ax.set_xlabel("Tenor (years)")
        ax.set_ylabel("Spread (bps)")
        ax.set_title("CDS Tenor Spread Curve")

    st.pyplot(fig)


def render_portfolio_placeholder():
    """
    Placeholder for the portfolio construction / outputs section.
    """
    st.markdown(
        """
        This section will summarize **portfolio-level construction and outputs**.

        Examples of what you might add later:
        - List of bonds/CDS currently in the portfolio  
        - Portfolio-level hazard curve or survival curve  
        - Aggregated risk price vs market price  
        - Scenario analysis and stress testing

        For now, this is a placeholder so that wiring it in later is painless.
        """
    )


##CDS Plots##
def plot_cds_basis_curves(basis_df, title='CDS Basis Over Time', figsize=(12, 6)):
    """
    Create a CDS basis curves figure for all tenors and return the Matplotlib figure.

    Args:
        basis_df: DataFrame with 'Date' column and 'CDS_Basis_*Y' columns
        title: plot title
        figsize: figure size (width, height)

    Returns:
        fig: Matplotlib Figure object
    """
    tenors_config = [
        ('CDS_Basis_1Y', 'steelblue', 'CDS Basis 1Y'),
        ('CDS_Basis_5Y', 'orange', 'CDS Basis 5Y'),
        ('CDS_Basis_10Y', 'red', 'CDS Basis 10Y')
    ]
    
    fig, ax = plt.subplots(figsize=figsize)

    for col_name, color, label in tenors_config:
        if col_name in basis_df.columns:
            ax.plot(basis_df['Date'], basis_df[col_name],
                    color=color, linewidth=2, label=label)

    ax.set_title(title, fontsize=16, fontweight='bold', pad=15)
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Basis (bps)', fontsize=12)

    # Format date axis
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    fig.autofmt_xdate()

    ax.grid(True, which='major', linestyle='--', alpha=0.6)
    ax.legend(frameon=False, fontsize=11)
    fig.tight_layout()

    return fig
