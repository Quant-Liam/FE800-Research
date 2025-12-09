import pandas as pd
from pathlib import Path
from datetime import datetime, date
import pandas as pd
import numpy as np

DATA_DIR = "FE800/FE800-Research/data/raw"   # you can rename this to your real folder name

DATA_ROOT = Path(__file__).resolve().parents[2] / "data" / "raw"


PRICING_DATE: date = datetime(2024, 5, 10).date()


def load_ual_bond_price_series() -> pd.DataFrame:
    """
    Load UAL bond price history from:
        data/raw/NewUAL/UALPricingnew.csv

    CSV columns:
        Date, Last

    Returns:
        DataFrame with normalized columns:
            'date' (datetime.date)
            'price' (float)
    """
    csv_path = DATA_ROOT / "NewUAL" / "UALPricingnew.csv"
    df = pd.read_csv(csv_path)

    # Normalize column names
    df = df.rename(columns={"Date": "date", "Last": "price"})

    # Parse dates
    df["date"] = pd.to_datetime(df["date"]).dt.date

    # Clean numeric values (strip spaces/commas)
    df["price"] = (
        df["price"]
        .astype(str)
        .str.replace(",", "")
        .str.strip()
        .astype(float)
    )

    return df


def get_ual_market_price_on_pricing_date(
    pricing_date: date = PRICING_DATE,
) -> dict:
    """
    Get the UAL bond market price as of the given pricing_date.

    Returns:
        {
            "market_price": float,
            "observation_date": date
        }
    """
    df = load_ual_bond_price_series()

    # Try exact match first
    row = df[df["date"] == pricing_date]

    # If no exact match, take the last available date BEFORE pricing_date
    if row.empty:
        df_before = df[df["date"] <= pricing_date].sort_values("date")
        if df_before.empty:
            raise ValueError(
                f"No UAL price on or before {pricing_date} in UALPricingnew.csv"
            )
        row = df_before.tail(1)

    market_price = float(row["price"].iloc[0])
    obs_date = row["date"].iloc[0]

    return {
        "market_price": market_price,
        "observation_date": obs_date,
    }