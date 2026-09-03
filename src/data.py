"""Functions for retrieving and cleaning public market data."""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd
import yfinance as yf


FUNDAMENTAL_FIELDS = {
    "marketCap": "Market Cap",
    "trailingPE": "Trailing P/E",
    "revenueGrowth": "Revenue Growth",
    "earningsGrowth": "Earnings Growth",
    "profitMargins": "Profit Margin",
    "debtToEquity": "Debt to Equity",
}


def _statement_row(statement: pd.DataFrame, possible_names: list[str]) -> pd.Series:
    """Return the first matching financial-statement row or an empty series."""
    # Check the preferred row labels in order because providers sometimes rename fields.
    for name in possible_names:
        # Return the row as soon as an exact label is available.
        if name in statement.index:
            # Convert its values to numbers and replace unexpected text with missing values.
            return pd.to_numeric(statement.loc[name], errors="coerce").dropna()

    # Return a typed empty series when none of the possible labels exists.
    return pd.Series(dtype="float64")


def _statement_fundamentals(stock: yf.Ticker) -> dict[str, float]:
    """Derive selected fundamentals from price and financial-statement data."""
    # Start with missing values so any unavailable calculation remains explicit.
    result = {display_name: np.nan for display_name in FUNDAMENTAL_FIELDS.values()}

    # Retrieve the faster price summary used for market capitalization and last price.
    try:
        # Convert the provider's lazy mapping into a normal dictionary.
        fast_information = dict(stock.fast_info)
    except Exception:
        # Continue with an empty dictionary if the fast endpoint is unavailable.
        fast_information = {}

    # Store market capitalization when the fast endpoint supplies it.
    result["Market Cap"] = fast_information.get("marketCap", np.nan)

    # Protect the remaining fallback from missing financial-statement endpoints.
    try:
        # Retrieve annual income-statement data for growth and profitability measures.
        annual_income = stock.income_stmt

        # Retrieve quarterly income data for trailing twelve-month earnings per share.
        quarterly_income = stock.quarterly_income_stmt

        # Retrieve the annual balance sheet for leverage measures.
        balance_sheet = stock.balance_sheet

        # Find total revenue using the common provider row labels.
        annual_revenue = _statement_row(annual_income, ["Total Revenue", "Operating Revenue"])

        # Find net income using the common provider row labels.
        annual_net_income = _statement_row(
            annual_income,
            ["Net Income", "Net Income Common Stockholders"],
        )

        # Calculate annual revenue growth when two complete fiscal years are available.
        if len(annual_revenue) >= 2 and annual_revenue.iloc[1] != 0:
            # Compare the most recent fiscal year with the previous fiscal year.
            result["Revenue Growth"] = annual_revenue.iloc[0] / annual_revenue.iloc[1] - 1

        # Calculate annual earnings growth when two complete fiscal years are available.
        if len(annual_net_income) >= 2 and annual_net_income.iloc[1] != 0:
            # Use the absolute previous value so a prior loss does not reverse the growth sign.
            result["Earnings Growth"] = (
                annual_net_income.iloc[0] - annual_net_income.iloc[1]
            ) / abs(annual_net_income.iloc[1])

        # Calculate the latest full-year net profit margin.
        if len(annual_revenue) >= 1 and len(annual_net_income) >= 1 and annual_revenue.iloc[0] != 0:
            # Divide net income by revenue for the same latest annual reporting column.
            result["Profit Margin"] = annual_net_income.iloc[0] / annual_revenue.iloc[0]

        # Retrieve diluted earnings per share for the latest reported quarters.
        quarterly_eps = _statement_row(quarterly_income, ["Diluted EPS", "Basic EPS"])

        # Read the latest market price from the fast summary.
        latest_price = fast_information.get("lastPrice", np.nan)

        # Calculate trailing P/E when four quarters and a positive EPS total are available.
        if len(quarterly_eps) >= 4 and pd.notna(latest_price):
            # Add the latest four reported quarters to form trailing-twelve-month EPS.
            trailing_eps = quarterly_eps.iloc[:4].sum()

            # Avoid a misleading P/E value when trailing earnings are zero or negative.
            if trailing_eps > 0:
                # Divide the latest price by trailing-twelve-month diluted EPS.
                result["Trailing P/E"] = latest_price / trailing_eps

        # Find the latest reported total debt.
        total_debt = _statement_row(balance_sheet, ["Total Debt"])

        # Find the latest reported shareholder equity.
        stockholder_equity = _statement_row(
            balance_sheet,
            ["Stockholders Equity", "Common Stock Equity"],
        )

        # Calculate debt to equity when both balance-sheet values are available.
        if len(total_debt) >= 1 and len(stockholder_equity) >= 1 and stockholder_equity.iloc[0] != 0:
            # Express total debt as a multiple of shareholder equity.
            result["Debt to Equity"] = total_debt.iloc[0] / stockholder_equity.iloc[0]
    except Exception:
        # Keep any values already found when one statement endpoint fails.
        pass

    # Return the complete fallback dictionary.
    return result


def clean_tickers(tickers: Iterable[str]) -> list[str]:
    """Return unique, uppercase ticker symbols while preserving their order."""
    # Create an empty list that will keep the cleaned ticker symbols.
    cleaned: list[str] = []

    # Examine each user-provided ticker symbol.
    for ticker in tickers:
        # Remove spaces and convert the symbol to uppercase.
        symbol = str(ticker).strip().upper()

        # Keep a non-empty symbol only once.
        if symbol and symbol not in cleaned:
            # Add the valid symbol to the output list.
            cleaned.append(symbol)

    # Return the cleaned symbols in their original order.
    return cleaned


def download_prices(
    tickers: Iterable[str],
    start_date: str,
    end_date: str | None = None,
) -> pd.DataFrame:
    """Download adjusted closing prices and return a clean date-by-symbol table."""
    # Standardize the ticker list before sending it to the data provider.
    symbols = clean_tickers(tickers)

    # Stop early if the user did not supply any valid ticker symbols.
    if not symbols:
        # Raise a readable error that can be displayed in the dashboard.
        raise ValueError("Enter at least one valid ticker symbol.")

    # Download adjusted historical market data for every symbol in one request.
    raw_data = yf.download(
        symbols,
        start=start_date,
        end=end_date,
        auto_adjust=True,
        progress=False,
        group_by="column",
    )

    # Stop if the provider returned no rows.
    if raw_data.empty:
        # Raise a readable message instead of allowing later calculations to fail.
        raise ValueError("No market data was returned. Check the symbols and dates.")

    # Handle the multi-level columns returned when several symbols are requested.
    if isinstance(raw_data.columns, pd.MultiIndex):
        # Select the adjusted closing-price group.
        prices = raw_data["Close"].copy()
    else:
        # Select the single closing-price column returned for one symbol.
        prices = raw_data[["Close"]].copy()

        # Rename the generic Close column to the requested symbol.
        prices.columns = [symbols[0]]

    # Put the columns in the same order as the user's cleaned ticker list.
    prices = prices.reindex(columns=symbols)

    # Remove dates on which every symbol is missing.
    prices = prices.dropna(how="all")

    # Carry the previous valid price forward across occasional isolated gaps.
    prices = prices.ffill()

    # Remove symbols that contain no usable prices across the entire period.
    prices = prices.dropna(axis=1, how="all")

    # Stop if cleaning removed every requested symbol.
    if prices.empty or prices.shape[1] == 0:
        # Explain the most likely cause to the dashboard user.
        raise ValueError("None of the requested symbols had usable price history.")

    # Return the cleaned table to the analysis layer.
    return prices


def fetch_fundamentals(tickers: Iterable[str]) -> pd.DataFrame:
    """Retrieve selected company fundamentals without failing on missing fields."""
    # Create an empty list for one company record per ticker.
    rows: list[dict[str, object]] = []

    # Request fundamentals for every cleaned ticker symbol.
    for symbol in clean_tickers(tickers):
        # Begin the record with the ticker symbol.
        row: dict[str, object] = {"Symbol": symbol}

        # Create one reusable company object for all provider requests.
        stock = yf.Ticker(symbol)

        # Protect the dashboard from temporary provider failures.
        try:
            # Retrieve the latest company information dictionary.
            company_info = stock.info

            # Copy each selected provider field into a readable column.
            for source_name, display_name in FUNDAMENTAL_FIELDS.items():
                # Use get so unavailable fields become missing values.
                value = company_info.get(source_name, np.nan)

                # Convert provider nulls into numeric missing values for consistent display.
                row[display_name] = np.nan if value is None else value
        except Exception:
            # Fill every requested field with a numeric missing value after a failure.
            for display_name in FUNDAMENTAL_FIELDS.values():
                # Store NaN so pandas handles the field consistently.
                row[display_name] = np.nan

        # Derive missing fields from public statements when the summary endpoint is blocked.
        fallback_values = _statement_fundamentals(stock)

        # Fill only fields that the primary company-summary request did not provide.
        for display_name, fallback_value in fallback_values.items():
            # Preserve a valid summary value and use the fallback only for missing data.
            if pd.isna(row.get(display_name)) and pd.notna(fallback_value):
                # Store the calculated or alternative provider value.
                row[display_name] = fallback_value

        # Add the completed company record to the result list.
        rows.append(row)

    # Return an empty but correctly labeled table when no symbols were supplied.
    if not rows:
        # Build the empty table with the same columns as an ordinary result.
        return pd.DataFrame(columns=["Symbol", *FUNDAMENTAL_FIELDS.values()]).set_index("Symbol")

    # Convert the company records into a table indexed by stock symbol.
    fundamentals = pd.DataFrame(rows).set_index("Symbol")

    # Convert all displayed fields to numeric values and preserve unavailable values as NaN.
    fundamentals = fundamentals.apply(pd.to_numeric, errors="coerce")

    # Identify debt-to-equity values retrieved from Yahoo's percentage-like summary scale.
    summary_scale = fundamentals["Debt to Equity"].abs() > 10

    # Convert only those percentage-like summary values into ordinary ratios.
    fundamentals.loc[summary_scale, "Debt to Equity"] = (
        fundamentals.loc[summary_scale, "Debt to Equity"] / 100
    )

    # Return the numeric table so the display layer can choose its own formatting.
    return fundamentals


def fetch_recent_news(ticker: str, limit: int = 10) -> pd.DataFrame:
    """Return a compact table of recent headlines for one ticker."""
    # Standardize the focus symbol.
    symbol = clean_tickers([ticker])[0]

    # Retrieve recent news records from the provider.
    raw_news = yf.Ticker(symbol).news

    # Create an empty list for clean headline records.
    rows: list[dict[str, object]] = []

    # Process only the requested number of news items.
    for item in raw_news[:limit]:
        # Support provider versions that nest article fields under content.
        content = item.get("content", item)

        # Retrieve the headline title.
        title = content.get("title")

        # Retrieve the publisher information.
        provider = content.get("provider", {})

        # Extract a readable publisher from a dictionary or plain string.
        publisher = provider.get("displayName") if isinstance(provider, dict) else provider

        # Retrieve either form of publication timestamp used by the provider.
        published = content.get("pubDate") or content.get("providerPublishTime")

        # Add only records containing a real headline.
        if title:
            # Store the clean headline fields.
            rows.append(
                {
                    "Published": published,
                    "Publisher": publisher,
                    "Headline": title,
                }
            )

    # Convert the clean records into a display table.
    return pd.DataFrame(rows)


def fetch_earnings_dates(ticker: str, limit: int = 8) -> pd.DatetimeIndex:
    """Return recent earnings dates as timezone-free timestamps."""
    # Standardize the focus ticker symbol.
    symbol = clean_tickers([ticker])[0]

    # Request recent reported earnings dates.
    earnings = yf.Ticker(symbol).get_earnings_dates(limit=limit)

    # Return an empty date index when the provider supplies no events.
    if earnings is None or earnings.empty:
        # Create the empty result with the correct index type.
        return pd.DatetimeIndex([])

    # Convert each event timestamp into a timezone-free normalized date.
    dates = [pd.Timestamp(value).tz_localize(None).normalize() for value in earnings.index[:limit]]

    # Return the clean dates in a pandas date index.
    return pd.DatetimeIndex(dates)
