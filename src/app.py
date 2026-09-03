"""Interactive Streamlit dashboard for the stock analytics project."""

from __future__ import annotations

from datetime import date

import pandas as pd
import streamlit as st

from src.charts import (
    correlation_chart,
    drawdown_chart,
    fundamental_chart,
    growth_chart,
    risk_return_chart,
)
from src.data import (
    clean_tickers,
    download_prices,
    fetch_earnings_dates,
    fetch_fundamentals,
    fetch_recent_news,
)
from src.metrics import (
    build_metrics_table,
    calculate_correlation,
    calculate_drawdown,
    event_return_table,
    normalize_growth,
)


# Configure the browser tab, icon and wide dashboard layout.
st.set_page_config(
    page_title="Medwin Stock Analytics",
    page_icon="📈",
    layout="wide",
)

# Add small visual refinements without changing Streamlit's accessible structure.
st.markdown(
    """
    <style>
    .block-container {padding-top: 2rem; padding-bottom: 3rem;}
    [data-testid="stMetric"] {background: #f4f8fb; border: 1px solid #d9e1e8; padding: 1rem; border-radius: 0.65rem;}
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(ttl=3600, show_spinner=False)
def cached_prices(symbols: tuple[str, ...], start: str, end: str) -> pd.DataFrame:
    """Cache prices for one hour so ordinary dashboard changes stay responsive."""
    # Delegate the actual download and cleaning work to the reusable data module.
    return download_prices(symbols, start, end)


@st.cache_data(ttl=3600, show_spinner=False)
def cached_fundamentals(symbols: tuple[str, ...]) -> pd.DataFrame:
    """Cache fundamental requests because they are slower than chart updates."""
    # Delegate the provider request to the reusable data module.
    return fetch_fundamentals(symbols)


@st.cache_data(ttl=1800, show_spinner=False)
def cached_news(symbol: str) -> pd.DataFrame:
    """Cache recent headlines for thirty minutes."""
    # Request a compact table of the ten most recent available headlines.
    return fetch_recent_news(symbol, limit=10)


# Display the project title at the top of the dashboard.
st.title("Stock Performance and Fundamental Analysis")

# Explain the dashboard in one sentence for a first-time user.
st.caption(
    "Compare historical performance, risk, company fundamentals, earnings events and recent headlines. "
    "Educational analysis only - not investment advice."
)

# Create a sidebar for all user-editable analysis choices.
with st.sidebar:
    # Label the controls as one coherent configuration section.
    st.header("Analysis settings")

    # Let the user enter comma-separated company symbols.
    ticker_text = st.text_input(
        "Stock symbols",
        value="AAPL, MSFT, NVDA, AMZN, JPM",
        help="Enter up to eight Yahoo Finance ticker symbols separated by commas.",
    )

    # Let the user choose a market benchmark.
    benchmark = st.text_input(
        "Benchmark",
        value="SPY",
        help="SPY is commonly used as a tradable S&P 500 benchmark.",
    )

    # Let the user choose the beginning of the historical period.
    start_date = st.date_input("Start date", value=date(2021, 1, 1))

    # Use yesterday as the inclusive user-facing end date.
    end_date = st.date_input("End date", value=date.today())

    # Let the user adjust the simplifying annual risk-free assumption.
    risk_free_percent = st.slider(
        "Risk-free rate assumption",
        min_value=0.0,
        max_value=10.0,
        value=4.0,
        step=0.25,
        format="%.2f%%",
    )

    # Provide a button that makes the data refresh action obvious.
    run_analysis = st.button("Run analysis", type="primary", width="stretch")

# Split and standardize the comma-separated stock symbols.
stocks = clean_tickers(ticker_text.split(","))[:8]

# Standardize the benchmark as a one-item ticker list.
benchmark_symbols = clean_tickers([benchmark])

# Use SPY if the user deletes the benchmark input.
benchmark_symbol = benchmark_symbols[0] if benchmark_symbols else "SPY"

# Combine stocks and benchmark without duplicating a symbol.
all_symbols = clean_tickers(stocks + [benchmark_symbol])

# Stop before downloading if the selected dates are reversed.
if start_date >= end_date:
    # Display a specific correction for the invalid date selection.
    st.error("The start date must be earlier than the end date.")

    # Stop execution after showing the correction.
    st.stop()

# Stop before downloading if no individual company was entered.
if not stocks:
    # Display a readable correction for the missing ticker list.
    st.error("Enter at least one stock symbol.")

    # Stop execution after showing the correction.
    st.stop()

# Explain that the button also provides a deliberate refresh point.
if run_analysis:
    # Clear cached values after the user deliberately requests a fresh analysis.
    st.cache_data.clear()

# Show a progress message while live market data is downloaded.
with st.spinner("Downloading and analyzing public market data..."):
    # Protect the user interface from invalid symbols or temporary provider failures.
    try:
        # Download the adjusted prices using hashable values for Streamlit caching.
        prices = cached_prices(
            tuple(all_symbols),
            start_date.isoformat(),
            end_date.isoformat(),
        )
    except Exception as error:
        # Display the readable provider or validation error.
        st.error(str(error))

        # Stop because every later calculation depends on the price table.
        st.stop()

# Keep only the symbols that produced usable price history.
available_symbols = list(prices.columns)

# Warn the user when a requested symbol was omitted by the data provider.
missing_symbols = [symbol for symbol in all_symbols if symbol not in available_symbols]

# Display the omitted symbols when necessary.
if missing_symbols:
    # Explain which inputs did not produce a usable series.
    st.warning("No usable history was returned for: " + ", ".join(missing_symbols))

# Convert the percentage control into a decimal calculation input.
risk_free_rate = risk_free_percent / 100

# Calculate the complete performance-and-risk table.
metrics = build_metrics_table(prices, risk_free_rate)

# Calculate normalized investment growth for comparison.
growth = normalize_growth(prices)

# Calculate the daily-return correlation matrix.
correlation = calculate_correlation(prices)

# Create a row of summary cards for the benchmark and top observations.
summary_columns = st.columns(4)

# Identify the symbol with the highest annualized historical return.
highest_return_symbol = metrics["Annualized Return"].idxmax()

# Identify the symbol with the lowest annualized historical volatility.
lowest_volatility_symbol = metrics["Annualized Volatility"].idxmin()

# Display the number of symbols included in the cleaned analysis.
summary_columns[0].metric("Symbols analyzed", len(available_symbols))

# Display the highest annualized historical return.
summary_columns[1].metric(
    "Highest annualized return",
    f"{metrics.loc[highest_return_symbol, 'Annualized Return']:.1%}",
    highest_return_symbol,
    delta_color="off",
)

# Display the lowest annualized historical volatility.
summary_columns[2].metric(
    "Lowest volatility",
    f"{metrics.loc[lowest_volatility_symbol, 'Annualized Volatility']:.1%}",
    lowest_volatility_symbol,
    delta_color="off",
)

# Display the available historical date range.
summary_columns[3].metric(
    "History",
    f"{prices.index.min():%b %Y}",
    f"through {prices.index.max():%b %Y}",
    delta_color="off",
)

# Separate the dashboard into focused analysis tabs.
overview_tab, risk_tab, fundamentals_tab, events_tab, methodology_tab = st.tabs(
    ["Overview", "Risk", "Fundamentals", "Events & News", "Methodology"]
)

# Build the main performance-comparison tab.
with overview_tab:
    # Introduce the normalized investment comparison.
    st.subheader("Performance comparison")

    # Render the interactive growth chart at the available page width.
    st.plotly_chart(growth_chart(growth), width="stretch")

    # Create a readable copy of the numeric metric table.
    metrics_display = metrics.copy()

    # Apply percentage formatting to return, volatility and drawdown measures.
    st.dataframe(
        metrics_display.style.format(
            {
                "Total Return": "{:.2%}",
                "Annualized Return": "{:.2%}",
                "Annualized Volatility": "{:.2%}",
                "Maximum Drawdown": "{:.2%}",
                "Sharpe Ratio": "{:.2f}",
            }
        ),
        width="stretch",
    )

    # Convert the numeric metrics into downloadable CSV bytes.
    metrics_csv = metrics.to_csv().encode("utf-8")

    # Provide a download button for the clean metric table.
    st.download_button(
        "Download performance metrics",
        data=metrics_csv,
        file_name="stock_summary_metrics.csv",
        mime="text/csv",
    )

# Build the historical-risk tab.
with risk_tab:
    # Display the interactive return-and-volatility comparison.
    st.plotly_chart(risk_return_chart(metrics), width="stretch")

    # Display the interactive diversification heatmap.
    st.plotly_chart(correlation_chart(correlation), width="stretch")

    # Let the user choose one available symbol for detailed drawdown analysis.
    focus_symbol = st.selectbox("Drawdown focus", options=available_symbols)

    # Calculate the selected symbol's complete drawdown series.
    focus_drawdown = calculate_drawdown(prices[focus_symbol])

    # Display the interactive drawdown chart.
    st.plotly_chart(drawdown_chart(focus_drawdown, focus_symbol), width="stretch")

    # Explain the worst drawdown in a plain-English sentence.
    st.info(
        f"The worst {focus_symbol} drawdown in the selected period was "
        f"{focus_drawdown.min():.2%}. A drawdown is a decline from a previous high."
    )

# Build the company-fundamentals tab.
with fundamentals_tab:
    # Explain the limitations of third-party fundamental fields.
    st.caption("Fundamental figures can be delayed or unavailable. Verify important values against company filings.")

    # Show a progress indicator while the slower company requests run.
    with st.spinner("Retrieving company fundamentals..."):
        # Retrieve fundamentals only for individual stocks, not the benchmark ETF.
        fundamentals = cached_fundamentals(tuple([symbol for symbol in stocks if symbol in available_symbols]))

    # Display a message when no company fundamentals were returned.
    if fundamentals.empty or fundamentals.dropna(how="all").empty:
        # Explain the provider limitation without failing the entire dashboard.
        st.warning("No fundamental data was returned for the selected companies.")
    else:
        # Create a display copy so the underlying numeric values remain downloadable.
        fundamentals_display = fundamentals.copy()

        # Render the numeric table with finance-appropriate formatting.
        st.dataframe(
            fundamentals_display.style.format(
                {
                    "Market Cap": "${:,.0f}",
                    "P/E Ratio": "{:.2f}",
                    "Revenue Growth": "{:.2%}",
                    "Earnings Growth": "{:.2%}",
                    "Profit Margin": "{:.2%}",
                    "Debt to Equity": "{:.2f}",
                },
                na_rep="N/A",
            ),
            width="stretch",
        )

        # Build the optional growth-versus-valuation chart.
        fundamentals_figure = fundamental_chart(fundamentals)

        # Display the chart only when both required fields are available.
        if fundamentals_figure is not None:
            # Render the interactive company comparison.
            st.plotly_chart(fundamentals_figure, width="stretch")

        # Provide the underlying numeric fundamentals as a downloadable CSV.
        st.download_button(
            "Download fundamentals",
            data=fundamentals.to_csv().encode("utf-8"),
            file_name="stock_fundamentals.csv",
            mime="text/csv",
        )

# Build the earnings-and-news research tab.
with events_tab:
    # Let the user choose a company rather than the benchmark ETF.
    event_options = [symbol for symbol in stocks if symbol in available_symbols]

    # Stop this optional tab gracefully if only the benchmark produced valid data.
    if not event_options:
        # Tell the user why company-specific event analysis cannot run.
        st.warning("No selected company produced usable price history for event analysis.")

        # Stop this app run after preserving the analysis already displayed in earlier tabs.
        st.stop()

    # Select the first valid company by default.
    event_symbol = st.selectbox("Company to research", options=event_options)

    # Explain what the event calculation can and cannot establish.
    st.caption("Returns near an event describe timing; they do not prove that the event caused the price movement.")

    # Protect the optional earnings section from temporary provider failures.
    try:
        # Retrieve recent reported earnings dates.
        earnings_dates = fetch_earnings_dates(event_symbol, limit=8)

        # Calculate event-day and five-trading-day returns.
        earnings_study = event_return_table(prices[event_symbol], earnings_dates)

        # Display the event table when at least one event can be measured.
        if not earnings_study.empty:
            # Format return values as percentages in the dashboard table.
            st.dataframe(
                earnings_study.style.format(
                    {"Event-Day Return": "{:.2%}", "5-Day Return": "{:.2%}"}
                ),
                width="stretch",
            )
        else:
            # Explain why a table could not be produced.
            st.info("No recent earnings dates had enough surrounding price history.")
    except Exception as error:
        # Keep the dashboard running when optional earnings data is unavailable.
        st.warning(f"Earnings-event data is currently unavailable: {error}")

    # Add a divider between event analysis and the headline feed.
    st.divider()

    # Introduce the financial-news research section.
    st.subheader("Recent headlines")

    # Explain that headlines are prompts rather than complete evidence.
    st.caption("Use headlines to identify topics for further research, then verify them with reliable reporting and filings.")

    # Protect the optional news feed from temporary provider failures.
    try:
        # Retrieve a cached table of recent headlines.
        recent_news = cached_news(event_symbol)

        # Display the headline table when records are available.
        if not recent_news.empty:
            # Render the clean headline table without its numeric index.
            st.dataframe(recent_news, width="stretch", hide_index=True)
        else:
            # Explain that the provider returned no current headlines.
            st.info("No recent headlines were returned for this company.")
    except Exception as error:
        # Keep the rest of the analysis usable when the headline request fails.
        st.warning(f"Recent headlines are currently unavailable: {error}")

# Build the methodology and limitations tab.
with methodology_tab:
    # Explain the analytical approach in a concise ordered sequence.
    st.subheader("Method")

    # Display the reproducible workflow as a numbered list.
    st.markdown(
        """
        1. Download adjusted daily closing prices.
        2. Clean incomplete observations and align symbols by date.
        3. Calculate historical returns, annualized volatility, maximum drawdown and a simplified Sharpe ratio.
        4. Compare normalized growth and daily-return correlations.
        5. Retrieve selected company fundamentals.
        6. Measure price changes near reported earnings dates.
        7. Display recent headlines as research prompts, not causal evidence.
        """
    )

    # Introduce the assumptions and analytical boundaries.
    st.subheader("Important limitations")

    # Display the limitations that prevent overclaiming.
    st.markdown(
        """
        - Historical performance does not predict future performance.
        - The selected companies form a small, non-random sample.
        - The risk-free rate is a simplifying user-selected assumption.
        - Taxes, transaction costs, inflation and portfolio rebalancing are not modeled.
        - Correlations and volatility can change during unusual market periods.
        - Third-party fundamental and news fields may be delayed, revised or unavailable.
        - Price movement around an event does not prove causation.
        """
    )
