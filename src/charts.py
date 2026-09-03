"""Interactive Plotly charts used by the Streamlit dashboard."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


NAVY = "#17365D"
TEAL = "#168A8A"
RED = "#B04A5A"


def growth_chart(growth: pd.DataFrame) -> go.Figure:
    """Create an interactive line chart of normalized investment growth."""
    # Move the date index into a normal column for Plotly Express.
    long_data = growth.rename_axis("Date").reset_index()

    # Convert the wide symbol columns into a tidy symbol-and-value layout.
    long_data = long_data.melt(
        id_vars="Date",
        var_name="Symbol",
        value_name="Investment Value",
    )

    # Draw one interactive line for every symbol.
    figure = px.line(
        long_data,
        x="Date",
        y="Investment Value",
        color="Symbol",
        title="Growth of a Hypothetical $100 Investment",
    )

    # Add a reference line showing the original investment value.
    figure.add_hline(y=100, line_dash="dash", line_color="gray")

    # Improve spacing and hover behavior.
    figure.update_layout(hovermode="x unified", legend_title_text="Symbol")

    # Return the completed interactive figure.
    return figure


def risk_return_chart(metrics: pd.DataFrame) -> go.Figure:
    """Create an interactive scatter plot of annual return and volatility."""
    # Move the ticker index into a normal Symbol column.
    chart_data = metrics.rename_axis("Symbol").reset_index()

    # Plot volatility horizontally and return vertically.
    figure = px.scatter(
        chart_data,
        x="Annualized Volatility",
        y="Annualized Return",
        text="Symbol",
        hover_name="Symbol",
        title="Historical Risk Versus Return",
        color_discrete_sequence=[TEAL],
    )

    # Increase point size and position ticker labels above each point.
    figure.update_traces(marker={"size": 14}, textposition="top center")

    # Format both numeric axes as percentages.
    figure.update_xaxes(tickformat=".0%")

    # Format the return axis as a percentage too.
    figure.update_yaxes(tickformat=".0%")

    # Return the completed interactive figure.
    return figure


def correlation_chart(correlation: pd.DataFrame) -> go.Figure:
    """Create an annotated heatmap of daily-return correlations."""
    # Draw the full correlation matrix with a fixed scale from -1 to +1.
    figure = px.imshow(
        correlation,
        text_auto=".2f",
        color_continuous_scale="RdBu_r",
        zmin=-1,
        zmax=1,
        title="Correlation of Daily Returns",
        aspect="auto",
    )

    # Label the scale so a non-technical reader knows what the colors represent.
    figure.update_coloraxes(colorbar_title="Correlation")

    # Return the completed heatmap.
    return figure


def drawdown_chart(drawdown: pd.Series, symbol: str) -> go.Figure:
    """Create a filled chart showing declines from previous highs."""
    # Create an empty Plotly figure for the custom area chart.
    figure = go.Figure()

    # Add the complete drawdown history as a filled line.
    figure.add_trace(
        go.Scatter(
            x=drawdown.index,
            y=drawdown.values,
            mode="lines",
            name=symbol,
            line={"color": RED, "width": 2},
            fill="tozeroy",
            fillcolor="rgba(176, 74, 90, 0.20)",
        )
    )

    # Add a descriptive title and percentage formatting.
    figure.update_layout(title=f"{symbol} Drawdown From Previous High", hovermode="x unified")

    # Format the drawdown axis as percentages.
    figure.update_yaxes(title="Drawdown", tickformat=".0%")

    # Label the date axis.
    figure.update_xaxes(title="Date")

    # Return the completed drawdown figure.
    return figure


def fundamental_chart(fundamentals: pd.DataFrame) -> go.Figure | None:
    """Compare recent revenue growth with trailing price-to-earnings ratios."""
    # Keep only rows containing both values needed by the chart.
    chart_data = fundamentals[["Revenue Growth", "Trailing P/E"]].dropna()

    # Return no chart when the data provider supplied no complete rows.
    if chart_data.empty:
        # Allow the dashboard to display a helpful warning instead.
        return None

    # Move the ticker index into a normal Symbol column.
    chart_data = chart_data.rename_axis("Symbol").reset_index()

    # Plot valuation horizontally and recent growth vertically.
    figure = px.scatter(
        chart_data,
        x="Trailing P/E",
        y="Revenue Growth",
        text="Symbol",
        hover_name="Symbol",
        title="Recent Revenue Growth Versus Trailing P/E",
        color_discrete_sequence=[NAVY],
    )

    # Increase marker size and place labels above the points.
    figure.update_traces(marker={"size": 14}, textposition="top center")

    # Format recent revenue growth as a percentage.
    figure.update_yaxes(tickformat=".0%")

    # Return the completed comparison chart.
    return figure

