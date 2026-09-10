from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from cryptopulse.analytics import (
    add_market_features,
    aggregate_category_drivers,
    aggregate_sentiment,
    build_sentiment_market_history,
    load_market_data,
    load_sentiment_data,
)
from cryptopulse.config import Settings
from cryptopulse.db.database import open_database
from cryptopulse.db.schema import create_schema

ASSETS = ("BTC", "ETH", "SOL", "BNB")
ASSET_COLORS = {
    "BTC": "#c87913",
    "ETH": "#2563a6",
    "SOL": "#6b5aa6",
    "BNB": "#8a6a00",
}
CATEGORY_LABELS = {
    "etf_flows": "ETF flows",
    "institutional_adoption": "Institutional adoption",
    "regulation": "Regulation",
    "technology": "Technology",
    "security_hacks": "Security and hacks",
    "market_movement": "Market movement",
    "macro": "Macro",
    "exchange_activity": "Exchange activity",
    "other": "Other",
}

st.set_page_config(page_title="CryptoPulse AI", layout="wide")
st.markdown(
    """
    <style>
    .block-container {max-width: 1440px; padding-top: 1.7rem;}
    h1, h2, h3 {letter-spacing: 0;}
    h1 {font-size: 2rem; margin-bottom: 0.2rem;}
    h2 {font-size: 1.25rem;}
    div[data-testid="stMetric"] {
        border: 1px solid #dce4df;
        border-radius: 6px;
        padding: 0.8rem 0.9rem;
        background: #ffffff;
    }
    div[data-testid="stMetricLabel"] {color: #53615b;}
    div[data-testid="stMetricValue"] {font-size: 1.35rem;}
    .status-line {color: #53615b; font-size: 0.88rem; margin-bottom: 1rem;}
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(ttl=300, show_spinner=False)
def load_dashboard_data(database_path: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    with open_database(Path(database_path)) as connection:
        create_schema(connection)
        sentiment = load_sentiment_data(connection, provider="openai")
        market = load_market_data(connection)
    return sentiment, add_market_features(market)


def format_price(value: float) -> str:
    if pd.isna(value):
        return "No data"
    if value >= 1_000:
        return f"${value:,.0f}"
    return f"${value:,.2f}"


def date_bounds(
    sentiment: pd.DataFrame,
    market: pd.DataFrame,
) -> tuple[object, object]:
    dates = []
    if not sentiment.empty:
        dates.extend(sentiment["published_at"].dt.date.tolist())
    if not market.empty:
        dates.extend(market["candle_timestamp"].dt.date.tolist())
    today = pd.Timestamp.now(tz="UTC").date()
    return (min(dates), max(dates)) if dates else (today, today)


def filter_by_date(
    frame: pd.DataFrame,
    timestamp_column: str,
    start_date: object,
    end_date: object,
) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    dates = frame[timestamp_column].dt.date
    return frame.loc[dates.between(start_date, end_date)].copy()


def latest_market_snapshot(market: pd.DataFrame) -> pd.DataFrame:
    if market.empty:
        return market.copy()
    return (
        market.sort_values(["asset", "candle_timestamp"])
        .groupby("asset", observed=True)
        .tail(1)
        .set_index("asset")
    )


def render_overview(
    selected_assets: list[str],
    market: pd.DataFrame,
    sentiment_history: pd.DataFrame,
) -> None:
    latest_market = latest_market_snapshot(market)
    latest_sentiment = (
        sentiment_history.sort_values(["asset", "period"])
        .groupby("asset", observed=True)
        .tail(1)
        .set_index("asset")
        if not sentiment_history.empty
        else pd.DataFrame()
    )

    columns = st.columns(max(1, len(selected_assets)))
    for column, asset in zip(columns, selected_assets, strict=True):
        market_row = latest_market.loc[asset] if asset in latest_market.index else None
        sentiment_row = (
            latest_sentiment.loc[asset]
            if asset in latest_sentiment.index
            else None
        )
        price = format_price(market_row["close"]) if market_row is not None else "No data"
        delta = None
        if market_row is not None and pd.notna(market_row["price_change_24h_pct"]):
            delta = f"{market_row['price_change_24h_pct']:+.2f}% / 24h"
        column.metric(f"{asset} price", price, delta=delta)
        if sentiment_row is None:
            column.caption("Sentiment: awaiting classified stories")
        else:
            score = sentiment_row["sentiment_score_0_100"]
            stories = int(sentiment_row["story_count"])
            column.caption(f"Sentiment {score:.1f}/100 from {stories} stories")

    st.subheader("Latest asset snapshot")
    rows = []
    for asset in selected_assets:
        market_row = latest_market.loc[asset] if asset in latest_market.index else None
        sentiment_row = (
            latest_sentiment.loc[asset]
            if asset in latest_sentiment.index
            else None
        )
        rows.append(
            {
                "Asset": asset,
                "Price": market_row["close"] if market_row is not None else None,
                "24h return (%)": (
                    market_row["price_change_24h_pct"]
                    if market_row is not None
                    else None
                ),
                "24h volatility (%)": (
                    market_row["rolling_volatility_24h_pct"]
                    if market_row is not None
                    else None
                ),
                "Sentiment (0-100)": (
                    sentiment_row["sentiment_score_0_100"]
                    if sentiment_row is not None
                    else None
                ),
                "Stories": (
                    int(sentiment_row["story_count"])
                    if sentiment_row is not None
                    else 0
                ),
            }
        )
    st.dataframe(
        pd.DataFrame(rows),
        hide_index=True,
        width="stretch",
        column_config={
            "Price": st.column_config.NumberColumn(format="$%.2f"),
            "24h return (%)": st.column_config.NumberColumn(format="%.2f"),
            "24h volatility (%)": st.column_config.NumberColumn(format="%.2f"),
            "Sentiment (0-100)": st.column_config.NumberColumn(format="%.1f"),
        },
    )


def render_sentiment_history(sentiment_history: pd.DataFrame) -> None:
    if sentiment_history.empty:
        st.info("No OpenAI sentiment results match the current filters.")
        return
    figure = px.line(
        sentiment_history,
        x="period",
        y="sentiment_score_0_100",
        color="asset",
        markers=True,
        color_discrete_map=ASSET_COLORS,
        labels={
            "period": "Date",
            "sentiment_score_0_100": "Sentiment score",
            "asset": "Asset",
        },
    )
    figure.add_hline(y=50, line_dash="dot", line_color="#8b9691")
    figure.update_yaxes(range=[0, 100])
    figure.update_layout(template="simple_white", hovermode="x unified", height=470)
    st.plotly_chart(figure, width="stretch")


def render_category_drivers(drivers: pd.DataFrame) -> None:
    if drivers.empty:
        st.info("Category drivers will appear after relevant stories are classified.")
        return
    frame = drivers.copy()
    frame["News category"] = frame["category"].map(CATEGORY_LABELS)
    figure = px.bar(
        frame,
        x="News category",
        y="weighted_sentiment",
        color="asset",
        barmode="group",
        text="story_count",
        color_discrete_map=ASSET_COLORS,
        labels={
            "weighted_sentiment": "Weighted sentiment (-1 to +1)",
            "asset": "Asset",
            "story_count": "Stories",
        },
    )
    figure.add_hline(y=0, line_color="#8b9691")
    figure.update_yaxes(range=[-1, 1])
    figure.update_layout(template="simple_white", height=470)
    st.plotly_chart(figure, width="stretch")


def render_stories(sentiment: pd.DataFrame) -> None:
    if sentiment.empty:
        st.info("No classified stories match the current filters.")
        return
    stories = sentiment.sort_values("published_at", ascending=False).copy()
    stories["published_at"] = stories["published_at"].dt.strftime(
        "%Y-%m-%d %H:%M UTC"
    )
    stories["category"] = stories["category"].map(CATEGORY_LABELS)
    stories = stories.rename(
        columns={
            "published_at": "Published",
            "asset": "Asset",
            "title": "Headline",
            "sentiment": "Sentiment",
            "sentiment_score": "Score",
            "confidence": "Confidence",
            "category": "Category",
            "url": "Source",
        }
    )
    st.dataframe(
        stories[
            [
                "Published",
                "Asset",
                "Headline",
                "Sentiment",
                "Score",
                "Confidence",
                "Category",
                "Source",
            ]
        ],
        hide_index=True,
        width="stretch",
        column_config={
            "Score": st.column_config.NumberColumn(format="%.2f"),
            "Confidence": st.column_config.ProgressColumn(min_value=0, max_value=1),
            "Source": st.column_config.LinkColumn(display_text="Open"),
        },
    )


def render_market_analysis(
    selected_assets: list[str],
    market: pd.DataFrame,
    combined_history: pd.DataFrame,
) -> None:
    if market.empty:
        st.info("No market observations match the current filters.")
        return
    chart_asset = st.selectbox("Asset", selected_assets, key="market_chart_asset")
    asset_market = market[market["asset"] == chart_asset]
    asset_combined = combined_history[combined_history["asset"] == chart_asset]

    figure = make_subplots(specs=[[{"secondary_y": True}]])
    figure.add_trace(
        go.Scatter(
            x=asset_market["candle_timestamp"],
            y=asset_market["close"],
            name=f"{chart_asset} price",
            line={"color": ASSET_COLORS[chart_asset], "width": 2},
        ),
        secondary_y=False,
    )
    if not asset_combined.empty:
        figure.add_trace(
            go.Scatter(
                x=asset_combined["period"],
                y=asset_combined["sentiment_score_0_100"],
                name="Daily sentiment",
                mode="lines+markers",
                line={"color": "#16794f", "dash": "dot", "width": 2},
            ),
            secondary_y=True,
        )
        figure.update_yaxes(
            title_text="Sentiment (0-100)", range=[0, 100], secondary_y=True
        )
    figure.update_yaxes(title_text="Price (USDT)", secondary_y=False)
    figure.update_layout(template="simple_white", hovermode="x unified", height=470)
    st.plotly_chart(figure, width="stretch")

    volatility = px.line(
        asset_market,
        x="candle_timestamp",
        y="rolling_volatility_24h_pct",
        labels={
            "candle_timestamp": "Time",
            "rolling_volatility_24h_pct": "Rolling 24h volatility (%)",
        },
    )
    volatility.update_traces(line_color="#b4493d")
    volatility.update_layout(template="simple_white", height=330, showlegend=False)
    st.plotly_chart(volatility, width="stretch")


def render_evaluation() -> None:
    metrics_path = Path("evaluation/model_metrics.csv")
    if not metrics_path.exists():
        st.info("Evaluation results are pending the completed human-labelled dataset.")
        return
    metrics = pd.read_csv(metrics_path)
    st.dataframe(metrics, hide_index=True, width="stretch")


settings = Settings()
sentiment_data, market_data = load_dashboard_data(str(settings.database_path))
minimum_date, maximum_date = date_bounds(sentiment_data, market_data)

st.title("CryptoPulse AI")
latest_timestamp = (
    market_data["candle_timestamp"].max().strftime("%Y-%m-%d %H:%M UTC")
    if not market_data.empty
    else "No market data"
)
st.markdown(
    f'<div class="status-line">Market and news intelligence | Data through {latest_timestamp}</div>',
    unsafe_allow_html=True,
)

st.sidebar.header("Filters")
selected_assets = st.sidebar.multiselect(
    "Assets",
    ASSETS,
    default=list(ASSETS),
)
if not selected_assets:
    st.sidebar.warning("Select at least one asset.")
    st.stop()

selected_dates = st.sidebar.date_input(
    "Date range",
    value=(minimum_date, maximum_date),
    min_value=minimum_date,
    max_value=maximum_date,
)
if isinstance(selected_dates, tuple) and len(selected_dates) == 2:
    start_date, end_date = selected_dates
else:
    start_date = end_date = (
        selected_dates[0] if isinstance(selected_dates, tuple) else selected_dates
    )

sentiment_options = ["bullish", "neutral", "bearish"]
selected_sentiments = st.sidebar.multiselect(
    "Sentiment",
    sentiment_options,
    default=sentiment_options,
)
available_categories = sorted(sentiment_data["category"].dropna().unique())
selected_categories = st.sidebar.multiselect(
    "News categories",
    available_categories,
    default=available_categories,
    format_func=lambda category: CATEGORY_LABELS.get(category, category),
)
if st.sidebar.button("Refresh data", width="stretch"):
    st.cache_data.clear()
    st.rerun()
st.sidebar.caption("Dashboard sentiment uses OpenAI classifications only.")

filtered_market = filter_by_date(
    market_data[market_data["asset"].isin(selected_assets)],
    "candle_timestamp",
    start_date,
    end_date,
)
filtered_sentiment = filter_by_date(
    sentiment_data[
        sentiment_data["asset"].isin(selected_assets)
        & sentiment_data["sentiment"].isin(selected_sentiments)
        & sentiment_data["category"].isin(selected_categories)
    ],
    "published_at",
    start_date,
    end_date,
)
sentiment_history = aggregate_sentiment(filtered_sentiment)
category_drivers = aggregate_category_drivers(filtered_sentiment)
combined_history = build_sentiment_market_history(
    sentiment_history,
    filtered_market,
)

overview_tab, history_tab, drivers_tab, stories_tab, market_tab, evaluation_tab = st.tabs(
    [
        "Overview",
        "Sentiment history",
        "News drivers",
        "Recent stories",
        "Market analysis",
        "Model evaluation",
    ]
)
with overview_tab:
    render_overview(selected_assets, filtered_market, sentiment_history)
with history_tab:
    render_sentiment_history(sentiment_history)
with drivers_tab:
    render_category_drivers(category_drivers)
with stories_tab:
    render_stories(filtered_sentiment)
with market_tab:
    render_market_analysis(selected_assets, filtered_market, combined_history)
with evaluation_tab:
    render_evaluation()

st.caption(
    "CryptoPulse is an educational analytics project. Sentiment and market "
    "co-movement do not establish causation and are not financial advice."
)
