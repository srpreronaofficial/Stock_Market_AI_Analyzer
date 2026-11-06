"""Phase 1 - Data Collection and Preparation.

Downloads (or reloads) historical OHLCV data for the selected instruments and
performs the cleaning steps required by the assignment:

    1. Load the data using Pandas
    2. Convert the Date column into datetime
    3. Sort all records chronologically
    4. Check for missing values
    5. Check for duplicate records
    6. Handle missing / invalid values
    7. Display basic statistical information
    8. Calculate the percentage daily return for each stock

Dataset source: Yahoo Finance via the ``yfinance`` package.

Market Index:
    * ^NSEI -> NIFTY 50

Tata Group:
    * TCS.NS       -> Tata Consultancy Services Ltd
    * TATASTEEL.NS -> Tata Steel Ltd
    * TATAPOWER.NS -> Tata Power Company Ltd
    * TATACONSUM.NS -> Tata Consumer Products Ltd

Infosys Group:
    * INFY.NS -> Infosys Ltd

Reliance Group:
    * RELIANCE.NS -> Reliance Industries Ltd
    * JIOFIN.NS   -> Jio Financial Services Ltd
    * RIIL.NS     -> Reliance Industrial Infrastructure Ltd

The raw download is cached in ``data/stock_data.csv`` so the project can be
re-run offline once the cache exists. If the ticker list changes, the cache
is automatically refreshed.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------

TICKERS = {
    # Market Index
    "^NSEI": "NIFTY 50",

    # Tata Group
    "TCS.NS": "Tata Consultancy Services",
    "TATASTEEL.NS": "Tata Steel",
    "TATAPOWER.NS": "Tata Power",
    "TATACONSUM.NS": "Tata Consumer Products",

    # Infosys Group
    "INFY.NS": "Infosys",

    # Reliance Group
    "RELIANCE.NS": "Reliance Industries",
    "JIOFIN.NS": "Jio Financial Services",
    "RIIL.NS": "Reliance Industrial Infrastructure",
}


# Historical data start date
START_DATE = "2018-01-01"

# End date = today
# Yahoo Finance will return data up to the latest available trading session.
END_DATE = pd.Timestamp.today().strftime("%Y-%m-%d")


# --------------------------------------------------------------------------
# Project paths
# --------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"

RAW_CSV = DATA_DIR / "stock_data.csv"

PROCESSED_CSV = DATA_DIR / "processed" / "features.csv"


# Columns we keep for modelling
PRICE_COLS = [
    "Open",
    "High",
    "Low",
    "Close",
    "Volume",
]


# --------------------------------------------------------------------------
# 1-3. Download / Load, convert dates, sort
# --------------------------------------------------------------------------

def download_ohlcv(
    tickers: dict[str, str] = TICKERS,
    start: str = START_DATE,
    end: str = END_DATE,
) -> pd.DataFrame:
    """Download OHLCV data from Yahoo Finance."""

    import yfinance as yf

    frames = []

    for symbol in tickers:

        print(f"Downloading {symbol} ...")

        raw = yf.download(
            symbol,
            start=start,
            end=end,
            auto_adjust=True,
            progress=False,
            threads=False,
        )

        if raw is None or raw.empty:
            print(f"  !! no data returned for {symbol}")
            continue

        # yfinance may return MultiIndex columns
        # for a single ticker in recent versions.
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)

        df = raw.reset_index()

        df["Symbol"] = symbol

        df["Name"] = tickers[symbol]

        frames.append(df)

    if not frames:
        raise RuntimeError(
            "No data downloaded - check network connection "
            "and Yahoo Finance ticker symbols."
        )

    out = pd.concat(
        frames,
        ignore_index=True
    )

    # Convert Date to datetime
    out["Date"] = pd.to_datetime(
        out["Date"]
    )

    # Remove timezone if present
    try:
        out["Date"] = out["Date"].dt.tz_localize(None)
    except TypeError:
        pass

    # Sort by company and date
    out = out.sort_values(
        ["Symbol", "Date"]
    ).reset_index(drop=True)

    return out


# --------------------------------------------------------------------------
# Load raw data
# --------------------------------------------------------------------------

def load_raw(force_download: bool = False) -> pd.DataFrame:
    """
    Load the raw dataset.

    The data is downloaded when:

    1. force_download=True
    2. stock_data.csv does not exist
    3. the existing CSV does not contain all required tickers

    This prevents the old 3-stock cache from being used after expanding
    the project to 8 companies + NIFTY 50.
    """

    required_symbols = set(TICKERS.keys())

    needs_download = (
        force_download
        or not RAW_CSV.exists()
    )

    # --------------------------------------------------------------
    # Check whether the existing cached CSV contains all tickers
    # --------------------------------------------------------------

    if not needs_download:

        try:

            cached = pd.read_csv(
                RAW_CSV,
                usecols=["Symbol"]
            )

            cached_symbols = set(
                cached["Symbol"]
                .dropna()
                .unique()
            )

            missing_symbols = (
                required_symbols - cached_symbols
            )

            if missing_symbols:

                print(
                    "Ticker list has changed."
                )

                print(
                    "Missing symbols in cached data:",
                    sorted(missing_symbols)
                )

                print(
                    "Refreshing stock data..."
                )

                needs_download = True

        except Exception as e:

            print(
                "Existing cache could not be validated."
            )

            print(
                f"Reason: {e}"
            )

            print(
                "Refreshing stock data..."
            )

            needs_download = True

    # --------------------------------------------------------------
    # Download new dataset
    # --------------------------------------------------------------

    if needs_download:

        df = download_ohlcv()

        RAW_CSV.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        df.to_csv(
            RAW_CSV,
            index=False
        )

        print(
            f"Cached raw data -> {RAW_CSV}"
        )

    # --------------------------------------------------------------
    # Load existing dataset
    # --------------------------------------------------------------

    else:

        df = pd.read_csv(
            RAW_CSV,
            parse_dates=["Date"]
        )

        print(
            f"Loaded cached data <- {RAW_CSV}"
        )

    return df


# --------------------------------------------------------------------------
# 4-7. Cleaning
# --------------------------------------------------------------------------

def clean(
    df: pd.DataFrame
) -> tuple[pd.DataFrame, dict]:
    """
    Apply the required cleaning steps.

    Returns:
        cleaned dataframe
        cleaning report dictionary
    """

    report: dict = {}

    # Make a copy
    df = df.copy()

    # --------------------------------------------------------------
    # 2. Convert Date to datetime
    # 3. Sort chronologically per symbol
    # --------------------------------------------------------------

    df["Date"] = pd.to_datetime(
        df["Date"]
    )

    df = df.sort_values(
        ["Symbol", "Date"]
    ).reset_index(drop=True)

    # --------------------------------------------------------------
    # 4. Check missing values
    # --------------------------------------------------------------

    missing_before = (
        df.isna()
        .sum()
        .to_dict()
    )

    report["missing_before"] = {
        k: int(v)
        for k, v in missing_before.items()
        if v
    }

    # --------------------------------------------------------------
    # 5. Check duplicate records
    # --------------------------------------------------------------

    dup_mask = df.duplicated(
        subset=[
            "Symbol",
            "Date"
        ],
        keep="first"
    )

    report["duplicate_rows"] = int(
        dup_mask.sum()
    )

    df = df[
        ~dup_mask
    ].reset_index(drop=True)

    # --------------------------------------------------------------
    # 6a. Convert numeric columns
    # Invalid values become NaN
    # --------------------------------------------------------------

    for col in PRICE_COLS:

        if col in df.columns:

            df[col] = pd.to_numeric(
                df[col],
                errors="coerce"
            )

    # --------------------------------------------------------------
    # 6b. Remove obviously invalid prices
    # --------------------------------------------------------------

    price_like = [
        "Open",
        "High",
        "Low",
        "Close",
    ]

    for col in price_like:

        if col in df.columns:

            df.loc[
                df[col] <= 0,
                col
            ] = np.nan

    # --------------------------------------------------------------
    # Invalid / zero trading volume
    # --------------------------------------------------------------

    if "Volume" in df.columns:

        df.loc[
            df["Volume"] <= 0,
            "Volume"
        ] = np.nan

    # --------------------------------------------------------------
    # 6c. Fill missing values separately for each company
    # --------------------------------------------------------------

    for col in price_like:

        if col in df.columns:

            df[col] = (
                df.groupby("Symbol")[col]
                .transform(
                    lambda s: s.ffill().bfill()
                )
            )

    if "Volume" in df.columns:

        df["Volume"] = (
            df.groupby("Symbol")["Volume"]
            .transform(
                lambda s: s.ffill().bfill()
            )
        )

    # --------------------------------------------------------------
    # Final missing values report
    # --------------------------------------------------------------

    report["missing_after"] = {
        k: int(v)
        for k, v in df.isna().sum().items()
        if v
    }

    # --------------------------------------------------------------
    # Dataset information
    # --------------------------------------------------------------

    report["rows"] = int(
        len(df)
    )

    report["symbols"] = (
        df["Symbol"]
        .unique()
        .tolist()
    )

    report["date_min"] = str(
        df["Date"]
        .min()
        .date()
    )

    report["date_max"] = str(
        df["Date"]
        .max()
        .date()
    )

    return df, report


# --------------------------------------------------------------------------
# 7. Basic statistical information
# --------------------------------------------------------------------------

def basic_stats(
    df: pd.DataFrame
) -> pd.DataFrame:
    """
    Generate basic statistical information
    for each stock/index.
    """

    return (
        df.groupby("Symbol")[PRICE_COLS]
        .describe()
        .round(2)
    )


# --------------------------------------------------------------------------
# 8. Daily percentage return
# --------------------------------------------------------------------------

def add_daily_return(
    df: pd.DataFrame
) -> pd.DataFrame:
    """
    Calculate percentage daily return separately
    for each stock/index.
    """

    df = df.copy()

    df["Daily_Return"] = (
        df.groupby("Symbol")["Close"]
        .pct_change(fill_method=None)
        * 100.0
    )

    return df


# --------------------------------------------------------------------------
# Complete Phase-1 pipeline
# --------------------------------------------------------------------------

def prepare() -> pd.DataFrame:
    """
    Full Phase-1 pipeline:

        Load
        ↓
        Clean
        ↓
        Calculate daily return
        ↓
        Return processed dataframe
    """

    raw = load_raw()

    clean_df, report = clean(
        raw
    )

    clean_df = add_daily_return(
        clean_df
    )

    print(
        "\n--- Phase 1 cleaning report ---"
    )

    for k, v in report.items():

        print(
            f"  {k}: {v}"
        )

    return clean_df


# --------------------------------------------------------------------------
# Run directly
# --------------------------------------------------------------------------

if __name__ == "__main__":

    df = prepare()

    print(
        "\n--- Basic Statistics ---"
    )

    print(
        basic_stats(df)
    )

    print(
        "\n--- First 5 Rows ---"
    )

    print(
        df.head()
    )