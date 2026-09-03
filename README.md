# Medwin Stock Analytics

An interactive Python project for exploring how selected stocks performed relative to a market benchmark. It combines price history, risk measures, company fundamentals, earnings-event analysis, and recent financial headlines in one Streamlit dashboard.

This is an educational data-analysis project, not investment advice.

## What the project demonstrates

- Python programming and organized, reusable code
- Data cleaning and analysis with pandas and NumPy
- Financial measures including total return, annualized return, volatility, maximum drawdown, and a simplified Sharpe ratio
- Interactive data visualization with Plotly
- Live public market-data retrieval with yfinance
- A user-facing dashboard built with Streamlit
- Automated tests for the core calculations
- Clear documentation of assumptions and limitations

The code is deliberately heavily commented so another student or reviewer can follow the reasoning.

## Dashboard features

1. Enter up to eight stock symbols and choose a benchmark.
2. Change the historical date range and risk-free-rate assumption.
3. Compare the growth of a hypothetical $100 investment.
4. Review return, volatility, drawdown, Sharpe ratio, and correlation.
5. Explore selected company fundamentals.
6. Measure returns around reported earnings dates.
7. View recent financial-news headlines as research prompts.
8. Download the calculated metrics and fundamentals as CSV files.

## Project structure

```text
medwin-stock-analytics/
├── app.py                    # Interactive Streamlit dashboard
├── src/
│   ├── data.py               # Market-data retrieval and cleaning
│   ├── metrics.py            # Financial calculations
│   └── charts.py             # Interactive Plotly charts
├── tests/
│   └── test_metrics.py       # Automated calculation checks
├── notebooks/
│   └── stock_analysis.ipynb  # Original exploratory analysis
├── requirements.txt          # Python packages required
└── README.md                 # Project guide
```

## Run it on your computer

You need Python 3.10 or newer. Open a terminal in this project folder and run:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

On Windows, activate the environment with:

```powershell
.venv\Scripts\activate
```

Streamlit will display a local web address. Open that address in a browser to use the dashboard.

## Run the tests

From the project folder, run:

```bash
python -m unittest discover -s tests -v
```

## Methodology

- Prices are adjusted daily closing prices retrieved from Yahoo Finance through yfinance.
- Total return compares the first and last available prices.
- Annualized return compounds the observed return using 252 trading days per year.
- Annualized volatility scales the standard deviation of daily returns by the square root of 252.
- Maximum drawdown measures the largest decline from a previous historical high.
- The simplified Sharpe ratio subtracts the selected annual risk-free rate from annualized return, then divides by annualized volatility.
- Correlation is calculated from daily percentage returns.
- An earnings event is matched to the first trading date on or after its reported date.

## Limitations

- Historical performance does not predict future performance.
- The default companies are a small, non-random sample.
- Taxes, transaction costs, inflation, dividends not reflected in adjusted data, and portfolio rebalancing are not separately modeled.
- Fundamental and headline fields may be delayed, revised, incomplete, or unavailable.
- Correlation, volatility, and relationships between companies can change over time.
- A price change near an earnings date does not prove that earnings caused the movement.

## Data source

The app uses the open-source `yfinance` package to retrieve publicly available Yahoo Finance data. No API key is required, but availability can change and important figures should be verified against company filings or another authoritative source.
