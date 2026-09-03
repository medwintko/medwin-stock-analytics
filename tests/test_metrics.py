"""Automated checks for the project's financial calculations."""

import unittest

import numpy as np
import pandas as pd

from src.data import clean_tickers
from src.metrics import (
    build_metrics_table,
    calculate_correlation,
    calculate_drawdown,
    calculate_metrics,
    normalize_growth,
)


class TestTickerCleaning(unittest.TestCase):
    """Check that user-entered ticker symbols are standardized safely."""

    def test_clean_tickers_removes_spaces_duplicates_and_blanks(self):
        # Pass mixed capitalization, duplicates and an empty entry.
        result = clean_tickers([" aapl ", "MSFT", "aapl", ""])

        # Confirm that the function preserves order and removes duplicates.
        self.assertEqual(result, ["AAPL", "MSFT"])


class TestFinancialMetrics(unittest.TestCase):
    """Check the numerical behavior of core reusable calculations."""

    def setUp(self):
        # Create five predictable daily dates for the test data.
        dates = pd.date_range("2026-01-01", periods=5, freq="D")

        # Create one rising series and one falling series.
        self.prices = pd.DataFrame(
            {
                "UP": [100.0, 105.0, 110.0, 115.0, 120.0],
                "DOWN": [100.0, 95.0, 90.0, 85.0, 80.0],
            },
            index=dates,
        )

    def test_normalized_growth_starts_at_selected_value(self):
        # Normalize both series to a hypothetical $100 investment.
        result = normalize_growth(self.prices, starting_value=100.0)

        # Confirm that every series starts at exactly $100.
        self.assertTrue((result.iloc[0] == 100.0).all())

    def test_drawdown_is_zero_at_new_highs(self):
        # Calculate drawdown for the continuously rising series.
        result = calculate_drawdown(self.prices["UP"])

        # Confirm that a series setting new highs never falls below its peak.
        self.assertTrue((result == 0.0).all())

    def test_drawdown_captures_twenty_percent_decline(self):
        # Calculate drawdown for the series declining from 100 to 80.
        result = calculate_drawdown(self.prices["DOWN"])

        # Confirm that the final and worst drawdown equals negative twenty percent.
        self.assertAlmostEqual(result.min(), -0.20)

    def test_total_return_matches_price_change(self):
        # Calculate the complete metric dictionary for the rising series.
        result = calculate_metrics(self.prices["UP"], risk_free_rate=0.0)

        # Confirm that moving from 100 to 120 produces a twenty percent total return.
        self.assertAlmostEqual(result["Total Return"], 0.20)

    def test_metrics_table_contains_every_symbol(self):
        # Calculate the comparison table for both sample symbols.
        result = build_metrics_table(self.prices, risk_free_rate=0.0)

        # Confirm that the table contains both original symbol labels.
        self.assertEqual(set(result.index), {"UP", "DOWN"})

    def test_correlation_matrix_is_symmetric(self):
        # Calculate return correlations for the sample prices.
        result = calculate_correlation(self.prices)

        # Confirm that the matrix equals its own transpose.
        self.assertTrue(np.allclose(result.values, result.values.T, equal_nan=True))


# Allow the test file to run directly as a normal Python program.
if __name__ == "__main__":
    # Run every test and print a readable result summary.
    unittest.main()

