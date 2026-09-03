"""Financial calculations used by both the notebook and dashboard."""

from __future__ import annotations

import numpy as np
import pandas as pd


TRADING_DAYS = 252


def calculate_drawdown(price_series: pd.Series) -> pd.Series:
    """Calculate the percentage decline from each previous running high."""
    # Remove missing prices so peaks are calculated only from real observations.
    clean_prices = price_series.dropna()

    # Track the highest price reached up to every date.
    running_peak = clean_prices.cummax()

    # Express each price as a percentage above or below its previous peak.
    drawdown = clean_prices / running_peak - 1

    # Return the complete drawdown history.
    return drawdown


def calculate_metrics(
    price_series: pd.Series,
    risk_free_rate: float = 0.04,
) -> dict[str, float]:
    """Calculate return, volatility, drawdown and Sharpe measures for one symbol."""
    # Remove missing prices before calculating returns.
    clean_prices = price_series.dropna()

    # Require at least two prices to calculate a change.
    if len(clean_prices) < 2:
        # Stop with a clear error instead of returning misleading statistics.
        raise ValueError("At least two price observations are required.")

    # Convert consecutive prices into daily percentage returns.
    daily_returns = clean_prices.pct_change(fill_method=None).dropna()

    # Calculate the total return from the first price to the final price.
    total_return = clean_prices.iloc[-1] / clean_prices.iloc[0] - 1

    # Convert the total return into a compound annual growth estimate.
    annualized_return = (1 + total_return) ** (TRADING_DAYS / len(daily_returns)) - 1

    # Scale daily standard deviation to an annual measure.
    annualized_volatility = daily_returns.std() * np.sqrt(TRADING_DAYS)

    # Find the most negative point in the drawdown history.
    maximum_drawdown = calculate_drawdown(clean_prices).min()

    # Calculate a simplified Sharpe ratio when volatility is nonzero.
    sharpe_ratio = (
        (annualized_return - risk_free_rate) / annualized_volatility
        if annualized_volatility > 0
        else np.nan
    )

    # Return the five calculated measures with descriptive names.
    return {
        "Total Return": float(total_return),
        "Annualized Return": float(annualized_return),
        "Annualized Volatility": float(annualized_volatility),
        "Maximum Drawdown": float(maximum_drawdown),
        "Sharpe Ratio": float(sharpe_ratio),
    }


def build_metrics_table(
    prices: pd.DataFrame,
    risk_free_rate: float = 0.04,
) -> pd.DataFrame:
    """Apply the same metric calculations to every price column."""
    # Calculate one metric dictionary for each ticker symbol.
    records = {
        symbol: calculate_metrics(prices[symbol], risk_free_rate)
        for symbol in prices.columns
    }

    # Convert the nested dictionaries into a symbol-by-metric table.
    metrics = pd.DataFrame(records).T

    # Sort the result from highest to lowest annualized historical return.
    return metrics.sort_values("Annualized Return", ascending=False)


def normalize_growth(prices: pd.DataFrame, starting_value: float = 100.0) -> pd.DataFrame:
    """Rebase each price series to the same hypothetical starting investment."""
    # Find the first real price in each column, even when symbols began trading on different dates.
    first_valid_prices = prices.apply(lambda column: column.dropna().iloc[0])

    # Divide every price by its own first available value.
    normalized = prices.divide(first_valid_prices)

    # Scale the result to the selected starting investment.
    return normalized.multiply(starting_value)


def calculate_correlation(prices: pd.DataFrame) -> pd.DataFrame:
    """Calculate correlation from daily percentage returns."""
    # Convert prices into consecutive daily percentage returns.
    daily_returns = prices.pct_change(fill_method=None)

    # Calculate and return the pairwise return-correlation matrix.
    return daily_returns.corr()


def event_return_table(
    price_series: pd.Series,
    event_dates: pd.DatetimeIndex,
    forward_days: int = 5,
) -> pd.DataFrame:
    """Measure returns from the prior close through each event window."""
    # Remove missing prices before matching event dates.
    clean_prices = price_series.dropna()

    # Create an empty list for one calculated record per event.
    records: list[dict[str, object]] = []

    # Examine every supplied event date.
    for raw_date in event_dates:
        # Convert the event into a timezone-free normalized timestamp.
        event_date = pd.Timestamp(raw_date).tz_localize(None).normalize()

        # Find trading dates on or after the event date.
        following_dates = clean_prices.index[clean_prices.index >= event_date]

        # Skip events that occur after the available price history.
        if len(following_dates) == 0:
            # Continue directly to the next event.
            continue

        # Match the event with the first available trading day.
        trading_date = following_dates[0]

        # Find the integer location of the matching trading date.
        position = clean_prices.index.get_loc(trading_date)

        # Require one prior close and the requested number of later trading days.
        if position >= 1 and position + forward_days < len(clean_prices):
            # Calculate the event-day return from the preceding close.
            event_day_return = clean_prices.iloc[position] / clean_prices.iloc[position - 1] - 1

            # Calculate the complete forward-window return from the preceding close.
            forward_return = clean_prices.iloc[position + forward_days] / clean_prices.iloc[position - 1] - 1

            # Add the event result to the output list.
            records.append(
                {
                    "Event Date": event_date.date(),
                    "Matched Trading Date": trading_date.date(),
                    "Event-Day Return": event_day_return,
                    f"{forward_days}-Day Return": forward_return,
                }
            )

    # Convert all calculated event records into a table.
    return pd.DataFrame(records)
