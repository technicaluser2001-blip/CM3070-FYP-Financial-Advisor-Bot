# Beginner Friendly Financial Advisor Bot

CM3070 Final Project — Lim QingXian Javier (230656507)

A machine learning system that predicts whether a stock is likely to rise or
fall over the next 20 trading days, and presents the result to beginner
investors as a clear signal with a plain English explanation.

---

## What is in this repository

| File | Description |
|---|---|
| `dashboard.py` | The Streamlit web application |
| `model.pkl` | Trained XGBoost classifier (produced by the training notebook) |
| `scaler.pkl` | Fitted StandardScaler used to normalise features before prediction |
| `Financial_Bot_training_model.ipynb` | Jupyter notebook covering data acquisition, feature engineering, model comparison, backtesting, and model export |
| `requirements.txt` | Python dependencies |

---

## Setup

The application needs two things installed: the Python dependencies, and the
Ollama application for generating explanations.

### 1. Python dependencies

Python 3.9 or later is required.

```bash
pip install -r requirements.txt
```

### 2. Ollama (required for the explanation feature)

The application generates its plain English explanations using a language
model running locally, rather than a hosted API. This avoids per-request
cost and means the application does not depend on an external service being
reachable.

1. Download and install Ollama from https://ollama.com/download
2. Pull the model used by the application (roughly a 2 GB download):

```bash
ollama pull llama3.2
```

3. Confirm it is working:

```bash
ollama run llama3.2
```

**If you skip this step**, the application still runs and every other feature
works normally. The explanation box will display "Explanation unavailable
right now" instead of a generated explanation.

---

## Running the application

From the folder containing `dashboard.py`:

```bash
streamlit run dashboard.py
```

The application opens automatically in your default browser at
`http://localhost:8501`. If it does not, that address is also printed in the
terminal.

If the `streamlit` command is not recognised, run it through Python instead:

```bash
python -m streamlit run dashboard.py
```

**Note:** `model.pkl` and `scaler.pkl` must be in the same folder as
`dashboard.py`. If they are missing, the application displays a message
explaining that the training notebook needs to be run first.

---

## Using the application

Select a stock from the sidebar dropdown, which contains a small curated set
of well-known companies, or choose "Custom ticker..." to enter any other
symbol. The chart period selector controls the Price Chart tab only.

The interface has three tabs:

- **Overview** — the current price, the predicted signal (Bullish or
  Bearish), the model's confidence, and a plain English explanation of what
  the signal means
- **Price Chart** — closing price with MA10 and MA50 overlays, a Bollinger
  Band envelope, and a daily returns bar chart. These are the same
  indicators the model uses as input features
- **Backtest** — how the signal would have performed against simply buying
  and holding the stock, over a test period the model was never trained on

The first load for a given stock takes a few seconds while data is
downloaded. Subsequent views are cached and much faster.

---

## Retraining the model

`model.pkl` and `scaler.pkl` are included, so the application runs without
retraining. To reproduce them, open `Financial_Bot_training_model.ipynb` and run all cells.
The notebook downloads historical data, engineers the six technical indicator
features, compares Logistic Regression, Random Forest, and XGBoost, backtests
the winning model, and exports both files.

Note that the notebook downloads live data from Yahoo Finance at runtime.
Because Yahoo revises historical prices over time, re-running the notebook
may produce results that differ slightly from those reported in the project
report. This is discussed in Section 6.4 of the report.

---

## Known limitations

- The model's predictive performance is weak. Across 493 S&P 500 stocks it
  achieved a mean test ROC-AUC of 0.504, only marginally better than chance.
  The application is a working demonstration of the pipeline, not a reliable
  forecasting tool, and the Backtest tab is included so that this is visible
  to the user rather than hidden.
- An internet connection is required, since price data is downloaded at
  runtime.
- The 1D and 5D chart periods use intraday data, which Yahoo Finance only
  provides for recent dates.
