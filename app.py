"""
Stock Market AI Analyzer
Professional Streamlit Dashboard

Run:
    streamlit run app.py
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import joblib
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots


# ============================================================================
# PROJECT PATH
# ============================================================================

BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR
sys.path.insert(0, str(PROJECT_DIR))


# ============================================================================
# EXISTING PROJECT LOGIC
# ============================================================================

from src.live import (
    DISPLAY,
    get_history,
    latest_features,
    latest_news,
    predict_direction,
    price_snapshot,
    volatility_from_history,
)


# ============================================================================
# PATHS
# ============================================================================

MODELS_DIR = PROJECT_DIR / "models"
BUNDLE_PATH = MODELS_DIR / "bundle.pkl"
REPORTS_DIR = PROJECT_DIR / "reports"
NEWS_DATA_PATH = PROJECT_DIR / "data" / "news_data.csv"

COMPANY_GROUPS = {
    "Market Index": {
        "^NSEI": "NIFTY 50 Index",
    },
    "Tata Group": {
        "TCS.NS": "Tata Consultancy Services",
        "TATASTEEL.NS": "Tata Steel",
        "TATAPOWER.NS": "Tata Power",
        "TATACONSUM.NS": "Tata Consumer Products",
    },
    "Infosys Group": {
        "INFY.NS": "Infosys",
    },
    "Reliance Group": {
        "RELIANCE.NS": "Reliance Industries",
        "JIOFIN.NS": "Jio Financial Services",
        "RIIL.NS": "Reliance Industrial Infrastructure",
    },
}


# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="Stock Market AI Analyzer",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================================
# CSS
# ============================================================================

st.markdown(
    """
    <style>
    /* ================================================================
       GENERAL DASHBOARD
       ================================================================ */

    .stApp {
        background-color: #f5f7fb !important;
    }

    .block-container {
        max-width: 1500px;
        padding: 2rem 2rem 3rem;
    }

    /* ================================================================
       SIDEBAR
       ================================================================ */

    section[data-testid="stSidebar"] {
        background-color: #111827 !important;
        min-width: 360px;
    }

    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3,
    section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"],
    section[data-testid="stSidebar"] [data-testid="stCaptionContainer"] {
        color: #f8fafc !important;
    }

    section[data-testid="stSidebar"] hr {
        border-color: #374151 !important;
    }

    /* White dropdown backgrounds */
    section[data-testid="stSidebar"] [data-baseweb="select"] > div {
        background-color: #ffffff !important;
        border-color: #cbd5e1 !important;
        border-radius: 9px !important;
    }

    /* Selected stock name and selected company group */
    section[data-testid="stSidebar"] [data-baseweb="select"] [class*="singleValue"],
    section[data-testid="stSidebar"] [data-baseweb="select"] [class*="SingleValue"] {
        color: #111827 !important;
        -webkit-text-fill-color: #111827 !important;
        opacity: 1 !important;
    }

    /* Placeholder text */
    section[data-testid="stSidebar"] [data-baseweb="select"] [class*="placeholder"] {
        color: #64748b !important;
        -webkit-text-fill-color: #64748b !important;
        opacity: 1 !important;
    }

    /* Dropdown's input and value container */
    section[data-testid="stSidebar"] [data-baseweb="select"] input {
        color: #111827 !important;
        -webkit-text-fill-color: #111827 !important;
        caret-color: #111827 !important;
        opacity: 1 !important;
    }

    section[data-testid="stSidebar"] [data-baseweb="select"] [class*="valueContainer"] {
        color: #111827 !important;
    }

    /* Dropdown arrow */
    section[data-testid="stSidebar"] [data-baseweb="select"] svg {
        color: #374151 !important;
        fill: #374151 !important;
    }

    /* Dropdown menu and option text */
    div[data-baseweb="popover"],
    div[data-baseweb="popover"] [role="listbox"] {
        background-color: #ffffff !important;
    }

    div[data-baseweb="popover"] [role="option"],
    div[data-baseweb="popover"] [role="option"] * {
        color: #111827 !important;
        -webkit-text-fill-color: #111827 !important;
        opacity: 1 !important;
    }

    div[data-baseweb="popover"] [role="option"]:hover {
        background-color: #eaf2ff !important;
    }

    /* Sidebar refresh button */
    section[data-testid="stSidebar"] .stButton > button {
        background-color: #2563eb !important;
        color: #ffffff !important;
        border: none !important;
        border-radius: 9px !important;
        min-height: 44px;
        font-weight: 700;
    }

    section[data-testid="stSidebar"] .stButton > button:hover {
        background-color: #1d4ed8 !important;
        color: #ffffff !important;
    }

    /* ================================================================
       HEADINGS AND DESCRIPTIONS
       ================================================================ */

    h1 {
        color: #111827 !important;
        font-weight: 850 !important;
        letter-spacing: -1px;
    }

    h2 {
        color: #111827 !important;
        font-weight: 800 !important;
    }

    h3 {
        color: #111827 !important;
        font-weight: 750 !important;
    }

    section.main h3 {
        font-size: 1.65rem !important;
        line-height: 1.28 !important;
        letter-spacing: -0.35px;
    }

    .section-description {
        color: #858b96 !important;
        font-size: 0.9rem;
        line-height: 1.55;
        margin-top: -0.4rem;
        margin-bottom: 1rem;
    }

    /* ================================================================
       STANDARD BORDERED CONTAINERS
       Preserve styling for the rest of the dashboard.
       ================================================================ */

    section.main div[data-testid="stVerticalBlockBorderWrapper"] {
        background: #f5f7fb !important;
        border: 1px solid #cbd0da !important;
        border-radius: 11px !important;
        box-shadow: none !important;
        box-sizing: border-box !important;
    }

    section.main div[data-testid="stVerticalBlockBorderWrapper"] > div {
        background: transparent !important;
        box-sizing: border-box !important;
    }

    div[data-testid="stHorizontalBlock"] > div[data-testid="column"] {
        min-width: 0 !important;
    }

    /* ================================================================
       MARKET OVERVIEW METRICS
       ================================================================ */

    section.main div[data-testid="stMetric"] {
        background: #f8fafc !important;
        border: 1px solid #e1e7f0 !important;
        border-radius: 13px !important;
        padding: 14px 15px !important;
        min-height: 104px !important;
        box-sizing: border-box !important;
        justify-content: center !important;
    }

    section.main div[data-testid="stMetricLabel"] {
        color: #64748b !important;
        font-size: 0.82rem !important;
        font-weight: 650 !important;
    }

    section.main div[data-testid="stMetricValue"] {
        color: #111827 !important;
        font-weight: 800 !important;
        font-size: 1.35rem !important;
        line-height: 1.3 !important;
        overflow-wrap: anywhere !important;
    }

    section.main div[data-testid="stMetricDelta"] {
        font-weight: 650 !important;
        font-size: 0.8rem !important;
    }

    /* ================================================================
       TECHNICAL INDICATOR STATUS MESSAGES
       ================================================================ */

    section.main div[data-testid="stAlert"] {
        border: none !important;
        border-radius: 11px !important;
        padding: 10px 14px !important;
    }

    section.main div[data-testid="stAlert"] p {
        font-size: 0.95rem !important;
        line-height: 1.45 !important;
    }

    /* ================================================================
       PREDICTION
       ================================================================ */

    .prediction-up {
        color: #15803d;
        font-size: 3.8rem;
        font-weight: 900;
        line-height: 1;
        margin-top: 0.4rem;
        margin-bottom: 0.4rem;
    }

    .prediction-down {
        color: #dc2626;
        font-size: 3.8rem;
        font-weight: 900;
        line-height: 1;
        margin-top: 0.4rem;
        margin-bottom: 0.4rem;
    }

    .prediction-label {
        color: #64748b;
        font-size: 0.85rem;
        font-weight: 600;
    }

    /* Chart explanations */
    .chart-description-title {
        color: #111827;
        font-size: 1rem;
        font-weight: 800;
        margin-bottom: 0.35rem;
    }

    .chart-description-text {
        color: #64748b;
        font-size: 0.82rem;
        line-height: 1.55;
    }

    /* News */
    .news-title {
        color: #111827;
        font-size: 0.95rem;
        font-weight: 700;
    }

    /* Tabs */
    button[data-baseweb="tab"] {
        font-weight: 700;
    }

    /* ================================================================
       CUSTOM MARKET OVERVIEW CARDS
       Five evenly sized cards.
       ================================================================ */

    .market-cards {
        display: grid;
        grid-template-columns: repeat(5, minmax(0, 1fr));
        gap: 16px;
        width: 100%;
        margin: 0.25rem 0 1.5rem;
    }

    .market-card {
        background: #f5f7fb !important;
        border: 1px solid #cbd0da !important;
        border-radius: 11px !important;
        padding: 18px 16px !important;
        min-height: 125px;
        min-width: 0;
        box-sizing: border-box;
        display: flex;
        flex-direction: column;
        justify-content: center;
        gap: 10px;
    }

    .market-card-label {
        color: #64748b !important;
        font-size: 0.86rem;
        line-height: 1.4;
        font-weight: 600;
    }

    .market-card-value {
        color: #111827 !important;
        font-size: clamp(1.05rem, 1.45vw, 1.45rem);
        line-height: 1.3;
        font-weight: 800;
        overflow-wrap: anywhere;
    }

    .market-card-delta {
        color: #64748b !important;
        font-size: 0.82rem;
        font-weight: 600;
    }

    /* ================================================================
       TECHNICAL INDICATOR CARDS
       Matches the reference: four outlined cards, inner value panels,
       colored status panels, and descriptions aligned near the bottom.
       ================================================================ */

    .indicator-cards {
        display: grid;
        grid-template-columns: repeat(4, minmax(0, 1fr));
        gap: 20px;
        align-items: stretch;
        width: 100%;
        margin: 0.5rem 0 1.5rem;
    }

    .indicator-card {
        background: #f5f7fb !important;
        border: 1px solid #cbd0da !important;
        border-radius: 11px !important;
        padding: 20px 19px 18px !important;
        min-width: 0;
        min-height: 385px;
        box-sizing: border-box;
        display: flex;
        flex-direction: column;
    }

    .indicator-card-title {
        color: #111827 !important;
        font-size: clamp(1.25rem, 1.8vw, 1.85rem);
        font-weight: 800;
        line-height: 1.25;
        margin: 12px 0 20px;
        min-height: 42px;
    }

    .indicator-value-panel {
        background: #f8fafc !important;
        border: 1px solid #e1e7f0 !important;
        border-radius: 13px !important;
        padding: 22px 16px !important;
        min-height: 135px;
        box-sizing: border-box;
        display: flex;
        flex-direction: column;
        justify-content: center;
        margin-bottom: 20px;
    }

    .indicator-value-label {
        color: #172033 !important;
        font-size: 0.92rem;
        margin-bottom: 7px;
    }

    .indicator-value {
        color: #111827 !important;
        font-size: 1.5rem;
        line-height: 1.25;
        font-weight: 800;
        overflow-wrap: anywhere;
    }

    .indicator-secondary {
        color: #8a909b !important;
        font-size: 0.9rem;
        margin: 3px 0 20px;
    }

    .indicator-status {
        border-radius: 11px;
        padding: 20px 18px;
        min-height: 70px;
        box-sizing: border-box;
        display: flex;
        align-items: center;
        font-size: 1rem;
        line-height: 1.4;
        margin: 0 0 22px;
    }

    .indicator-status.good {
        background: #dff3eb !important;
        color: #00834f !important;
    }

    .indicator-status.warn {
        background: #f7f7e2 !important;
        color: #9a6900 !important;
    }

    .indicator-status.info {
        background: #e7f0ff !important;
        color: #2459a9 !important;
    }

    .indicator-description {
        color: #8a909b !important;
        font-size: 0.9rem;
        line-height: 1.55;
        margin-top: auto;
    }

    /* ================================================================
       RESPONSIVE LAYOUT
       ================================================================ */

    @media (max-width: 1100px) {
        .market-cards {
            grid-template-columns: repeat(3, minmax(0, 1fr));
        }

        .indicator-cards {
            grid-template-columns: repeat(2, minmax(0, 1fr));
        }
    }

    @media (max-width: 650px) {
        .block-container {
            padding: 1.25rem 1rem 2rem;
        }

        .market-cards,
        .indicator-cards {
            grid-template-columns: 1fr;
        }

        .indicator-card {
            min-height: 0;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)
# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def safe_number(value):
    """Return a finite number, or None when a value is missing or invalid."""
    try:
        number = float(value)
        return number if math.isfinite(number) else None
    except (TypeError, ValueError):
        return None


def display_number(value, decimals=2, prefix="", suffix=""):
    """Format a number safely for display."""
    number = safe_number(value)
    if number is None:
        return "N/A"
    return f"{prefix}{number:,.{decimals}f}{suffix}"


# ============================================================================
# NEWS SOURCE DATA
# ============================================================================

@st.cache_data(show_spinner=False)
def load_news_sources() -> pd.DataFrame:
    if not NEWS_DATA_PATH.exists():
        return pd.DataFrame()

    try:
        df = pd.read_csv(NEWS_DATA_PATH)

        required = {"Symbol", "Date", "Headline", "URL"}
        if not required.issubset(df.columns):
            return pd.DataFrame()

        df["Date"] = pd.to_datetime(
            df["Date"], errors="coerce"
        ).dt.date

        return df

    except Exception:
        return pd.DataFrame()


# ============================================================================
# GET NEWS SOURCE
# ============================================================================

def get_news_source(symbol: str, row: pd.Series):
    url = row.get("URL", "")
    domain = row.get("Domain", "")

    if pd.notna(url) and str(url).strip():
        return (
            str(url).strip(),
            str(domain).strip() if pd.notna(domain) else "",
        )

    sources = load_news_sources()
    if sources.empty:
        return "", ""

    date_value = pd.to_datetime(row.get("Date"), errors="coerce")
    if pd.isna(date_value):
        return "", ""

    date_value = date_value.date()
    headline = str(row.get("Headline", "")).strip()

    match = sources[
        (sources["Symbol"].astype(str) == str(symbol))
        & (sources["Date"] == date_value)
        & (sources["Headline"].astype(str).str.strip() == headline)
    ]

    if match.empty:
        return "", ""

    source_row = match.iloc[0]

    return (
        str(source_row["URL"]).strip(),
        str(source_row.get("Domain", "")).strip(),
    )


# ============================================================================
# LOAD MODEL
# ============================================================================

@st.cache_resource(show_spinner="Loading trained model...")
def load_bundle():
    if not BUNDLE_PATH.exists():
        return {}

    try:
        return joblib.load(BUNDLE_PATH)
    except Exception:
        return {}


# ============================================================================
# LOAD HISTORY
# ============================================================================

@st.cache_data(ttl=60 * 60, show_spinner=False)
def history(symbol: str, refresh: bool = False):
    return get_history(symbol, force_refresh=refresh)


# ============================================================================
# MAIN
# ============================================================================

def main():

    # ========================================================================
    # LOAD MODEL
    # ========================================================================

    bundle = load_bundle()

    if not bundle:
        st.error(
            "No trained model found. Please ensure models/bundle.pkl "
            "exists inside the Stock_Market_AI_Analyzer project."
        )
        st.stop()

    model = bundle["model"]
    features = bundle["features"]
    thresholds = bundle["thresholds"]
    model_name = bundle.get("model_name", "Model")
    with_news = bundle.get("with_news", False)
    stats = bundle.get("backtest_stats", {})

    # ========================================================================
    # SIDEBAR
    # ========================================================================

    with st.sidebar:
        st.markdown("# 📈 Stock AI")
        st.caption("Market Intelligence Dashboard")
        st.markdown("### SELECT INSTRUMENT")

        group_col, company_col = st.columns([1.05, 1.45], gap="small")

        with group_col:
            group = st.selectbox(
                "Company group",
                options=list(COMPANY_GROUPS.keys()),
                label_visibility="collapsed",
                key="company_group",
            )

        with company_col:
            group_companies = COMPANY_GROUPS[group]
            symbol = st.selectbox(
                "Company",
                options=list(group_companies.keys()),
                format_func=lambda s: group_companies[s],
                label_visibility="collapsed",
                key="company_symbol",
            )

        refresh = st.button(
            "↻ Refresh Market Data",
            width="stretch",
        )

        st.divider()
        st.markdown("### MODEL INFORMATION")
        st.caption("Model")
        st.write(f"**{model_name}**")
        st.caption("News features")
        st.write(f"**{'Enabled' if with_news else 'Disabled'}**")

        if stats:
            st.caption("Held-out accuracy")
            accuracy_stat = safe_number(
                stats.get("Directional Accuracy %")
            )
            st.write(
                f"**{accuracy_stat:.1f}%**"
                if accuracy_stat is not None
                else "**N/A**"
            )

        st.divider()
        st.caption("Educational / research use only.")
        st.caption(
            "Predictions are experimental and should not be used as "
            "financial advice or as a basis for real-money trading."
        )

    # ========================================================================
    # GET MARKET DATA
    # ========================================================================

    try:
        hist = history(symbol, refresh=refresh)
    except Exception as exc:
        st.error(f"Unable to load market history: {exc}")
        st.stop()

    if hist is None or len(hist) < 2:
        st.error(
            "Not enough market history. Refresh the data or select another instrument."
        )
        st.stop()

    # ========================================================================
    # FEATURES
    # ========================================================================

    feats, last_row = latest_features(
        hist,
        features=features,
        symbol=symbol,
    )

    if last_row.empty:
        st.error("History is too short to calculate the required indicators.")
        st.stop()

    missing = [f for f in features if f not in last_row.columns]
    if missing:
        st.error("Missing model features: " + ", ".join(missing))
        st.stop()

    # ========================================================================
    # CALCULATIONS
    # ========================================================================

    snap = price_snapshot(hist)
    pred = predict_direction(model, features, last_row)
    vol = volatility_from_history(hist, thresholds)
    headlines, senti = latest_news(symbol)
    r = last_row.iloc[0]

    close_value = safe_number(snap.get("close"))
    change_value = safe_number(snap.get("change"))
    change_pct_value = safe_number(snap.get("change_pct"))
    volume_value = safe_number(snap.get("volume"))
    rsi_value = safe_number(r.get("RSI"))
    volatility_value = safe_number(vol.get("value"))

    # ========================================================================
    # PAGE HEADER
    # ========================================================================

    st.title("Stock Market AI Analyzer")

    st.markdown(
        '<p class="section-description">'
        'AI-powered market intelligence, technical analysis '
        'and financial news sentiment'
        '</p>',
        unsafe_allow_html=True,
    )

    # ========================================================================
    # SELECTED INSTRUMENT AND CURRENT PRICE
    # ========================================================================

    selected_col, price_col = st.columns([3, 1], gap="medium")

    with selected_col:
        with st.container(border=True):
            st.caption("SELECTED INSTRUMENT")
            st.subheader(f"{DISPLAY.get(symbol, symbol)} ({symbol})")
            st.caption(f"Market data as of {snap.get('date', 'N/A')}")

    with price_col:
        with st.container(border=True):
            st.caption("CURRENT PRICE")

            if close_value is None:
                st.subheader("N/A")
            elif symbol == "^NSEI":
                st.subheader(f"{close_value:,.2f}")
            else:
                st.subheader(f"₹{close_value:,.2f}")

            if change_value is not None and change_pct_value is not None:
                st.caption(
                    f"{change_value:+,.2f} ({change_pct_value:+.2f}%)"
                )
            else:
                st.caption("Daily change: N/A")

    # ========================================================================
    # MARKET OVERVIEW
    # ========================================================================

    st.header("📊 Market Overview")

    st.markdown(
        '<p class="section-description">'
        'Current market position and key technical indicators.'
        '</p>',
        unsafe_allow_html=True,
    )

    price_text = (
        f"{close_value:,.2f}"
        if close_value is not None and symbol == "^NSEI"
        else (f"₹{close_value:,.2f}" if close_value is not None else "N/A")
    )
    daily_change_text = f"{change_value:+,.2f}" if change_value is not None else "N/A"
    daily_change_delta = f"{change_pct_value:+.2f}%" if change_pct_value is not None else ""
    volume_text = f"{volume_value:,.0f}" if volume_value is not None and volume_value > 0 else "N/A"
    rsi_text = f"{rsi_value:.1f}" if rsi_value is not None else "N/A"
    volatility_text = str(vol.get("level", "N/A")) if volatility_value is not None else "N/A"
    volatility_delta = f"{volatility_value:.1f}%" if volatility_value is not None else ""

    market_cards = [
        ("💰 Latest Price", price_text, ""),
        ("📈 Daily Change", daily_change_text, daily_change_delta),
        ("📊 Volume", volume_text, ""),
        ("🎯 RSI (14)", rsi_text, ""),
        ("⚡ Volatility", volatility_text, volatility_delta),
    ]
    market_html = '<div class="market-cards">'
    for label, value, delta in market_cards:
        delta_html = f'<div class="market-card-delta">{delta}</div>' if delta else ""
        market_html += (
            '<div class="market-card">'
            f'<div class="market-card-label">{label}</div>'
            f'<div class="market-card-value">{value}</div>'
            f'{delta_html}'
            '</div>'
        )
    market_html += '</div>'
    st.markdown(market_html, unsafe_allow_html=True)

    # ========================================================================
    # ANALYSIS
    # ========================================================================

    st.header("📊 Analysis")

    st.markdown(
        '<p class="section-description">'
        'Machine-learning forecast and financial-news sentiment.'
        '</p>',
        unsafe_allow_html=True,
    )

    prediction_col, sentiment_col = st.columns([3, 2], gap="medium")

    # ========================================================================
    # PREDICTION
    # ========================================================================

    with prediction_col:
        with st.container(border=True):
            st.subheader("🤖 Next-Day ML Prediction")

            p1, p2, p3 = st.columns([1.5, 1, 1.4])

            with p1:
                if pred["direction"] == "UP":
                    st.markdown(
                        '<div class="prediction-up">↑ UP</div>',
                        unsafe_allow_html=True,
                    )
                    st.markdown(
                        '<div class="prediction-label">'
                        'AI predicts an upward movement</div>',
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        '<div class="prediction-down">↓ DOWN</div>',
                        unsafe_allow_html=True,
                    )
                    st.markdown(
                        '<div class="prediction-label">'
                        'AI predicts a downward movement</div>',
                        unsafe_allow_html=True,
                    )

            with p2:
                st.metric(
                    "Confidence",
                    display_number(pred.get("confidence"), 1, suffix="%"),
                )

            with p3:
                probability_up = safe_number(pred.get("probability_up"))
                st.metric(
                    "Probability UP",
                    display_number(
                        probability_up * 100
                        if probability_up is not None
                        else None,
                        1,
                        suffix="%",
                    ),
                )

            if probability_up is not None:
                st.progress(
                    min(1.0, max(0.0, probability_up)),
                    text=f"Probability of UP = {100 * probability_up:.1f}%",
                )

            st.caption(f"Last model input date: {snap.get('date', 'N/A')}")

    # ========================================================================
    # NEWS SENTIMENT
    # ========================================================================

    with sentiment_col:
        with st.container(border=True):
            st.subheader("📰 News Sentiment")

            sentiment_label = senti.get("label", "Neutral")
            emoji = {
                "Positive": "🟢",
                "Negative": "🔴",
                "Neutral": "⚪",
            }.get(sentiment_label, "⚪")

            s1, s2 = st.columns(2)

            with s1:
                st.metric("Sentiment", f"{emoji} {sentiment_label}")

            with s2:
                st.metric(
                    "Compound Score",
                    display_number(senti.get("compound"), 3, prefix="+"),
                )

            st.caption(f"Based on {senti.get('headlines', 0)} recent headlines")

            if senti.get("last_date"):
                st.caption(f"Latest: {senti['last_date']}")

    # ========================================================================
    # TECHNICAL ANALYSIS
    # ========================================================================

    st.header("📈 Technical Analysis")

    st.markdown(
        '<p class="section-description">'
        'Price trend, moving averages, trading volume and RSI.'
        '</p>',
        unsafe_allow_html=True,
    )

    # ========================================================================
    # ANALYSIS WINDOW
    # ========================================================================

    span = max(len(hist), 2)

    if span > 60:
        lo, hi = 60, span
    else:
        lo, hi = max(1, span // 2), span

    if hi <= lo:
        lo, hi = 1, max(2, span)

    step = 30 if hi - lo > 60 else 1
    default = max(lo, min(365, hi))

    window = st.slider(
        "Analysis window (trading days)",
        min_value=lo,
        max_value=hi,
        value=default,
        step=step,
    )

    plot = feats.tail(window)

    # ========================================================================
    # THREE-PANEL CHART
    # ========================================================================

    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        row_heights=[0.60, 0.20, 0.20],
        vertical_spacing=0.04,
        subplot_titles=[
            "Price & Moving Averages",
            "Trading Volume",
            "Relative Strength Index (RSI)",
        ],
    )

    fig.add_trace(
        go.Scatter(
            x=plot["Date"],
            y=plot["Close"],
            name="Close Price",
            line=dict(color="#2563eb", width=2.5),
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=plot["Date"],
            y=plot["MA_5"],
            name="MA 5",
            line=dict(color="#f59e0b", width=1.5),
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=plot["Date"],
            y=plot["MA_20"],
            name="MA 20",
            line=dict(color="#16a34a", width=1.5),
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Bar(
            x=plot["Date"],
            y=plot["Volume"],
            name="Trading Volume",
            marker_color="#93c5fd",
        ),
        row=2,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=plot["Date"],
            y=plot["RSI"],
            name="RSI",
            line=dict(color="#7c3aed", width=1.7),
        ),
        row=3,
        col=1,
    )

    fig.add_hline(
        y=70,
        line_dash="dot",
        line_color="#dc2626",
        row=3,
        col=1,
    )

    fig.add_hline(
        y=30,
        line_dash="dot",
        line_color="#16a34a",
        row=3,
        col=1,
    )

    fig.update_layout(
        height=740,
        paper_bgcolor="white",
        plot_bgcolor="white",
        hovermode="x unified",
        margin=dict(t=65, b=25, l=50, r=25),
        legend=dict(orientation="h", y=1.03),
    )

    fig.update_xaxes(showgrid=True, gridcolor="#edf1f5")
    fig.update_yaxes(showgrid=True, gridcolor="#edf1f5")
    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="Volume", row=2, col=1)
    fig.update_yaxes(title_text="RSI", range=[0, 100], row=3, col=1)

    st.plotly_chart(fig, width="stretch")

    # ========================================================================
    # GRAPH EXPLANATIONS
    # ========================================================================

    st.subheader("📚 How to Read the Charts")
    st.caption(
        "These three charts explain the major technical signals "
        "used by the Stock Market AI Analyzer."
    )

    g1, g2, g3 = st.columns(3, gap="medium")

    with g1:
        with st.container(border=True):
            st.markdown("### 📈 Price & Moving Averages")
            st.markdown(
                """
                **What it shows**

                The blue line represents the actual closing price of the selected stock.

                **MA-5** is the average closing price over the last 5 trading days.

                **MA-20** is the average closing price over the last 20 trading days.

                **How to understand it**

                - Price above MA-20 → stronger upward trend
                - Price below MA-20 → weaker downward trend
                - MA-5 above MA-20 → positive short-term momentum
                - MA-5 below MA-20 → weaker short-term momentum
                """
            )

    with g2:
        with st.container(border=True):
            st.markdown("### 📊 Trading Volume")
            st.markdown(
                """
                **What it shows**

                The bars represent the number of shares traded during each trading session.

                **How to understand it**

                - High volume + rising price → stronger buying interest
                - High volume + falling price → stronger selling interest
                - Low volume → relatively weaker participation

                Volume helps confirm whether a price movement has strong market activity behind it.
                """
            )

    with g3:
        with st.container(border=True):
            st.markdown("### 📉 RSI — Momentum")
            st.markdown(
                """
                **What it shows**

                RSI (Relative Strength Index) measures price momentum on a scale from 0 to 100.

                **Important levels**

                - RSI above 70 → potentially overbought
                - RSI below 30 → potentially oversold
                - RSI around 50 → relatively balanced momentum

                RSI helps identify the strength of recent price movement but does not guarantee a future price move.
                """
            )

    # ========================================================================
    # TECHNICAL INDICATOR CARDS
    # ========================================================================

    st.subheader("🔎 Technical Indicators")
    st.caption(
        "Key technical signals calculated from the selected stock's "
        "historical market data."
    )

    rsi_current = safe_number(r.get("RSI"))
    macd_current = safe_number(r.get("MACD"))
    ma5_current = safe_number(r.get("MA_5"))
    ma20_current = safe_number(r.get("MA_20"))
    close_vs_ma20 = safe_number(r.get("Close_vs_MA20"))

    # Status text/colors mirror the reference card design.
    if rsi_current is None:
        rsi_status, rsi_class = "RSI unavailable", "info"
    elif rsi_current > 70:
        rsi_status, rsi_class = "Potentially overbought", "warn"
    elif rsi_current < 30:
        rsi_status, rsi_class = "Potentially oversold", "info"
    else:
        rsi_status, rsi_class = "Neutral momentum", "good"

    if macd_current is None:
        macd_status, macd_class = "MACD unavailable", "info"
    elif macd_current > 0:
        macd_status, macd_class = "Positive momentum", "good"
    else:
        macd_status, macd_class = "Negative momentum", "warn"

    if ma5_current is None or ma20_current is None:
        ma_status, ma_class = "Moving average unavailable", "info"
    elif ma5_current > ma20_current:
        ma_status, ma_class = "Short-term trend stronger", "good"
    else:
        ma_status, ma_class = "Short-term trend weaker", "warn"

    if close_vs_ma20 is None:
        price_status, price_class = "Price position unavailable", "info"
    elif close_vs_ma20 > 0:
        price_status, price_class = "Price above MA-20", "good"
    else:
        price_status, price_class = "Price below MA-20", "warn"

    price_position_text = display_number(
        close_vs_ma20 * 100 if close_vs_ma20 is not None else None,
        2,
        suffix="%",
        prefix="+" if close_vs_ma20 is not None and close_vs_ma20 >= 0 else "",
    )

    indicator_data = [
        {
            "title": "🎯 RSI",
            "label": "Current RSI",
            "value": display_number(rsi_current, 1),
            "secondary": "",
            "status": rsi_status,
            "status_class": rsi_class,
            "description": "Measures price momentum from 0–100.",
        },
        {
            "title": "⚡ MACD",
            "label": "Current MACD",
            "value": display_number(macd_current, 2),
            "secondary": "",
            "status": macd_status,
            "status_class": macd_class,
            "description": "Helps identify trend and momentum changes.",
        },
        {
            "title": "📐 Moving Average",
            "label": "MA-5",
            "value": display_number(ma5_current, 2),
            "secondary": f"MA-20: {display_number(ma20_current, 2)}",
            "status": ma_status,
            "status_class": ma_class,
            "description": "Compares the short-term and 20-day moving averages.",
        },
        {
            "title": "📍 Price Position",
            "label": "vs MA-20",
            "value": price_position_text,
            "secondary": "",
            "status": price_status,
            "status_class": price_class,
            "description": "Current price compared with the 20-day average.",
        },
    ]

    indicator_html = '<div class="indicator-cards">'
    for card in indicator_data:
        secondary_html = (
            f'<div class="indicator-secondary">{card["secondary"]}</div>'
            if card["secondary"] else ""
        )
        indicator_html += (
            '<div class="indicator-card">'
            f'<div class="indicator-card-title">{card["title"]}</div>'
            '<div class="indicator-value-panel">'
            f'<div class="indicator-value-label">{card["label"]}</div>'
            f'<div class="indicator-value">{card["value"]}</div>'
            '</div>'
            f'{secondary_html}'
            f'<div class="indicator-status {card["status_class"]}">{card["status"]}</div>'
            f'<div class="indicator-description">{card["description"]}</div>'
            '</div>'
        )
    indicator_html += '</div>'
    st.markdown(indicator_html, unsafe_allow_html=True)

    # ========================================================================
    # TABS
    # ========================================================================

    st.divider()

    tab_news, tab_model, tab_about = st.tabs(
        ["📰 Financial News", "📊 Model Performance", "ℹ️ Methodology"]
    )

    # ========================================================================
    # NEWS TAB
    # ========================================================================

    with tab_news:
        st.subheader(f"Financial News — {DISPLAY.get(symbol, symbol)}")
        st.caption("Click a headline to open the original source.")

        if headlines.empty:
            st.info("No headlines available for this symbol.")
        else:
            for _, row in headlines.iterrows():
                label = row.get("label", "Neutral")
                emoji = {
                    "Positive": "🟢",
                    "Negative": "🔴",
                    "Neutral": "⚪",
                }.get(label, "⚪")

                url, domain = get_news_source(symbol, row)

                date_value = pd.to_datetime(row.get("Date"), errors="coerce")
                date_text = "" if pd.isna(date_value) else str(date_value.date())
                headline = str(row.get("Headline", "")).strip()
                score = display_number(row.get("compound"), 3, prefix="+")

                with st.container(border=True):
                    st.caption(f"{emoji} {date_text}")

                    if url:
                        st.markdown(f"**[{headline}]({url})**")
                    else:
                        st.markdown(f"**{headline}**")

                    n1, n2 = st.columns([4, 1])

                    with n1:
                        st.caption(
                            f"Source: {domain}"
                            if domain
                            else "Original source unavailable"
                        )

                    with n2:
                        st.caption(f"Sentiment: {score}")

    # ========================================================================
    # MODEL PERFORMANCE TAB
    # ========================================================================

    with tab_model:
        left, right = st.columns([3, 2], gap="large")
        comp = bundle.get("comparison")

        with left:
            st.subheader("Model Comparison")

            if isinstance(comp, pd.DataFrame) and not comp.empty:
                show = comp.drop(
                    columns=[
                        c for c in ["BestParams", "Experiment"]
                        if c in comp.columns
                    ]
                )
                st.dataframe(show, width="stretch", hide_index=True)
            else:
                st.info("Model comparison is unavailable.")

        with right:
            st.subheader("Backtest Results")

            if stats:
                for label, key in [
                    ("Total Predictions", "Total Predictions"),
                    ("Correct Predictions", "Correct Predictions"),
                    ("Incorrect Predictions", "Incorrect Predictions"),
                    ("UP Predictions", "UP Predictions"),
                    ("DOWN Predictions", "DOWN Predictions"),
                ]:
                    st.write(f"**{label}:** {stats.get(key, 'N/A')}")

                accuracy = safe_number(stats.get("Directional Accuracy %"))
                st.metric(
                    "Directional Accuracy",
                    display_number(accuracy, 1, suffix="%"),
                )
            else:
                st.info("Backtest results unavailable.")

            cm_images = sorted(REPORTS_DIR.glob("confusion_matrix_*.png"))
            if cm_images:
                st.image(
                    cm_images[-1],
                    caption="Confusion Matrix",
                    width="stretch",
                )

        st.divider()
        st.caption(
            "Volatility rule: 10-day rolling annualised volatility of returns."
        )
        st.caption(
            f"LOW ≤ {100 * thresholds['low_max']:.1f}% | "
            f"MEDIUM ≤ {100 * thresholds['medium_max']:.1f}% | "
            "HIGH above that"
        )

    # ========================================================================
    # METHODOLOGY TAB
    # ========================================================================

    with tab_about:
        st.subheader("ℹ️ About This Project")

        st.markdown(
            """
            ### What This Application Does

            **1. Market Data**

            The application retrieves historical OHLCV data for the selected NSE stock or index.

            **2. Technical Analysis**

            The application calculates important technical indicators including:

            - Moving Average (MA-5)
            - Moving Average (MA-20)
            - RSI
            - MACD
            - Volatility
            - Volume-related features

            **3. Machine Learning Prediction**

            The trained machine-learning model predicts whether the next trading day's closing price is likely to move **UP or DOWN**.

            **4. Financial News Sentiment**

            Recent financial headlines are analyzed and classified as:

            - 🟢 Positive
            - 🔴 Negative
            - ⚪ Neutral

            **5. Integrated Dashboard**

            The application combines market data, technical indicators, machine-learning prediction and financial news sentiment in one dashboard.

            ### Important Note

            The prediction is an experimental machine-learning result. It does not guarantee future stock performance.
            """
        )

        st.warning(
            "Educational / research use only. This application is not financial "
            "advice and should not be used for real-money trading."
        )

    # ========================================================================
    # FOOTER
    # ========================================================================

    st.divider()
    st.caption(
        "Stock Market AI Analyzer • Machine Learning • "
        "Technical Analysis • Financial News Sentiment"
    )


# ============================================================================
# RUN APPLICATION
# ============================================================================

if __name__ == "__main__":
    main()
