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
    build_sentiment_price_correlation,
    load_market_data,
    load_sentiment_data,
)
from cryptopulse.config import Settings
from cryptopulse.dashboard import resolve_dashboard_database
from cryptopulse.db.database import open_database, open_readonly_database
from cryptopulse.db.schema import create_schema

ASSETS = ("BTC", "ETH", "SOL", "BNB")
DASHBOARD_CACHE_VERSION = 2
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
def load_dashboard_data(
    database_path: str,
    read_only: bool,
    cache_version: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    del cache_version  # Included in Streamlit's cache key for schema changes.
    database_context = open_readonly_database if read_only else open_database
    with database_context(Path(database_path)) as connection:
        if not read_only:
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
    if "summary" not in stories.columns:
        stories["summary"] = None
    stories["published_at"] = stories["published_at"].dt.strftime(
        "%Y-%m-%d %H:%M UTC"
    )
    stories["category"] = stories["category"].map(CATEGORY_LABELS)
    stories = stories.rename(
        columns={
            "published_at": "Published",
            "asset": "Asset",
            "title": "Headline",
            "summary": "Summary",
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
                "Summary",
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


def render_sentiment_price_correlation(
    selected_assets: list[str],
    sentiment: pd.DataFrame,
    market: pd.DataFrame,
    start_date: object,
    end_date: object,
) -> None:
    st.caption(
        "Sentiment and price are aligned to fixed six-hour UTC periods. The "
        "seven-day rolling correlation requires at least 10 valid observations."
    )
    if sentiment.empty:
        st.info("No classified stories match the current filters.")
        return
    if market.empty:
        st.info("No market observations are available for correlation analysis.")
        return

    timing_labels = {
        "same_6h": "Same 6-hour period",
        "next_6h": "Following 6 hours",
        "next_24h": "Following 24 hours",
    }
    timing = st.selectbox(
        "Price comparison",
        list(timing_labels),
        index=1,
        format_func=timing_labels.get,
        key="correlation_timing",
    )
    correlation = build_sentiment_price_correlation(
        sentiment,
        market,
        timing=timing,
    )
    visible = filter_by_date(
        correlation,
        "period_end",
        start_date,
        end_date,
    )
    if visible.empty:
        st.info("No completed six-hour periods match the current filters.")
        return

    st.subheader("Seven-day rolling correlation")
    correlation_history = visible.dropna(subset=["rolling_correlation"])
    if correlation_history.empty:
        st.info(
            "At least 10 aligned observations are needed before a rolling "
            "correlation can be shown."
        )
    else:
        history_figure = px.line(
            correlation_history,
            x="period_end",
            y="rolling_correlation",
            color="asset",
            color_discrete_map=ASSET_COLORS,
            labels={
                "period_end": "Period ending",
                "rolling_correlation": "Rolling Spearman correlation",
                "asset": "Asset",
            },
        )
        history_figure.add_hline(y=0, line_color="#8b9691")
        history_figure.update_yaxes(range=[-1, 1])
        history_figure.update_layout(
            template="simple_white", hovermode="x unified", height=420
        )
        st.plotly_chart(history_figure, width="stretch")

    available_assets = [
        asset for asset in selected_assets if asset in set(visible["asset"])
    ]
    if not available_assets:
        st.info("No selected asset has completed correlation periods.")
        return
    detail_asset = st.selectbox(
        "Coin details",
        available_assets,
        key="correlation_asset",
    )
    asset_history = visible[visible["asset"].eq(detail_asset)].sort_values(
        "period_end"
    )
    latest_pair_rows = asset_history.dropna(
        subset=["sentiment_change", "price_change_pct"]
    )

    metric_columns = st.columns(3)
    latest_period = asset_history.iloc[-1]
    if pd.isna(latest_period["rolling_correlation"]):
        metric_columns[0].metric("Current 7-day correlation", "Insufficient data")
    else:
        metric_columns[0].metric(
            "Current 7-day correlation",
            f"{latest_period['rolling_correlation']:+.2f}",
        )
    metric_columns[1].metric(
        "Aligned observations",
        f"{int(latest_period['rolling_observations'])} / 28",
    )
    if latest_pair_rows.empty:
        metric_columns[2].metric("Latest price change", "No data")
    else:
        latest_pair = latest_pair_rows.iloc[-1]
        metric_columns[2].metric(
            "Latest price change",
            f"{latest_pair['price_change_pct']:+.2f}%",
            delta=f"{latest_pair['sentiment_change']:+.1f} sentiment points",
        )

    paired = asset_history.dropna(
        subset=["sentiment_change", "price_change_pct"]
    )
    if paired.empty:
        st.info("No aligned sentiment and price changes are available for this coin.")
        return

    comparison_figure = make_subplots(specs=[[{"secondary_y": True}]])
    comparison_figure.add_trace(
        go.Bar(
            x=paired["period_end"],
            y=paired["sentiment_change"],
            name="Sentiment change",
            marker_color="#16794f",
            opacity=0.72,
        ),
        secondary_y=False,
    )
    comparison_figure.add_trace(
        go.Scatter(
            x=paired["period_end"],
            y=paired["price_change_pct"],
            name="Price change",
            mode="lines+markers",
            line={"color": ASSET_COLORS[detail_asset], "width": 2},
        ),
        secondary_y=True,
    )
    comparison_figure.add_hline(y=0, line_color="#8b9691")
    comparison_figure.update_yaxes(
        title_text="Sentiment change (points)", secondary_y=False
    )
    comparison_figure.update_yaxes(
        title_text="Price change (%)", secondary_y=True
    )
    comparison_figure.update_layout(
        template="simple_white", hovermode="x unified", height=440
    )
    st.plotly_chart(comparison_figure, width="stretch")

    st.subheader("Period record")
    records = asset_history.rename(
        columns={
            "period_end": "Period ending",
            "story_count": "Stories",
            "sentiment_score_0_100": "Sentiment score",
            "sentiment_change": "Sentiment change",
            "price_change_pct": "Price change (%)",
            "rolling_observations": "Rolling observations",
            "rolling_correlation": "7-day correlation",
        }
    )
    records["Period ending"] = records["Period ending"].dt.strftime(
        "%Y-%m-%d %H:%M UTC"
    )
    st.dataframe(
        records[
            [
                "Period ending",
                "Stories",
                "Sentiment score",
                "Sentiment change",
                "Price change (%)",
                "Rolling observations",
                "7-day correlation",
            ]
        ],
        hide_index=True,
        width="stretch",
        column_config={
            "Sentiment score": st.column_config.NumberColumn(format="%.1f"),
            "Sentiment change": st.column_config.NumberColumn(format="%+.1f"),
            "Price change (%)": st.column_config.NumberColumn(format="%+.2f"),
            "7-day correlation": st.column_config.NumberColumn(format="%+.2f"),
        },
    )


def render_evaluation() -> None:
    metrics_path = Path("evaluation/model_metrics.csv")
    if not metrics_path.exists():
        st.info("Evaluation results have not been generated yet.")
        return
    metrics = pd.read_csv(metrics_path)
    st.caption(
        "Relevance is measured against independent human labels. Sentiment results "
        "measure agreement with AI-assisted labels and are not human-grounded accuracy. "
        "This fixed Stage 3 benchmark used headline-only inputs."
    )
    display_metrics = metrics.copy()
    percentage_columns = (
        "accuracy",
        "macro_precision",
        "macro_recall",
        "macro_f1",
        "weighted_f1",
    )
    for column in percentage_columns:
        display_metrics[column] = display_metrics[column].map(
            lambda value: f"{float(value):.1%}"
        )
    st.dataframe(display_metrics, hide_index=True, width="stretch")

    errors_path = Path("evaluation/model_errors.csv")
    if not errors_path.exists():
        return
    errors = pd.read_csv(errors_path).fillna("")
    st.subheader("Model disagreements")
    selected_task = st.selectbox(
        "Evaluation task",
        errors["task"].drop_duplicates().tolist(),
        key="evaluation_task",
    )
    task_errors = errors[errors["task"] == selected_task]
    selected_model = st.selectbox(
        "Model",
        task_errors["model"].drop_duplicates().tolist(),
        key="evaluation_model",
    )
    visible_columns = [
        "title",
        "asset",
        "reference",
        "prediction",
    ]
    st.dataframe(
        task_errors[task_errors["model"] == selected_model][visible_columns],
        hide_index=True,
        width="stretch",
    )


settings = Settings()
dashboard_database = resolve_dashboard_database(settings.database_path)
sentiment_data, market_data = load_dashboard_data(
    str(dashboard_database.path),
    dashboard_database.read_only,
    DASHBOARD_CACHE_VERSION,
)
minimum_date, maximum_date = date_bounds(sentiment_data, market_data)

st.title("CryptoPulse AI")
latest_timestamp = (
    market_data["candle_timestamp"].max().strftime("%Y-%m-%d %H:%M UTC")
    if not market_data.empty
    else "No market data"
)
data_mode = "Read-only demo snapshot" if dashboard_database.read_only else "Local data"
st.markdown(
    '<div class="status-line">Market and news intelligence | '
    f'{data_mode} | Data through {latest_timestamp}</div>',
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
st.sidebar.caption(
    f"Data mode: {data_mode}. Sentiment is model-generated from headlines and "
    "available RSS summaries."
)

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

(
    overview_tab,
    history_tab,
    drivers_tab,
    stories_tab,
    market_tab,
    correlation_tab,
    evaluation_tab,
) = st.tabs(
    [
        "Overview",
        "Sentiment history",
        "News drivers",
        "Recent stories",
        "Market analysis",
        "Sentiment-price correlation",
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
with correlation_tab:
    correlation_sentiment = sentiment_data[
        sentiment_data["asset"].isin(selected_assets)
        & sentiment_data["sentiment"].isin(selected_sentiments)
        & sentiment_data["category"].isin(selected_categories)
    ]
    render_sentiment_price_correlation(
        selected_assets,
        correlation_sentiment,
        market_data[market_data["asset"].isin(selected_assets)],
        start_date,
        end_date,
    )
with evaluation_tab:
    render_evaluation()

st.caption(
    "CryptoPulse is an educational analytics project. Sentiment and market "
    "co-movement do not establish causation and are not financial advice."
)
