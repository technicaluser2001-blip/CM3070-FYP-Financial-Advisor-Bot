#CM3070 Final Project Financial Advisor Bot
#Run with streamlit run dashboard.py

import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta
import ta
import ollama
import backtrader as bt
import joblib
import os
from sklearn.preprocessing import StandardScaler

# Page config
st.set_page_config(
    page_title="Financial Advisor Bot",
    page_icon="📈",
    layout="wide"
)

# Constants
FEATURES = ["MA10_50_ratio", "Daily_Return", "Return_5d", "Volatility", "BB_pct", "RSI"]
HORIZON  = 20   # predict 20 trading days ahead

# Curated starter selection of well-known stocks
CURATED_TICKERS = {
    "NVDA": "NVIDIA",
    "AAPL": "Apple",
    "MSFT": "Microsoft",
    "JNJ": "Johnson & Johnson",
    "XOM": "ExxonMobil",
    "KO": "Coca-Cola",
}

# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------
def add_features(df):
    df = df.copy()
    close = df["Close"].squeeze()

    df["MA10"] = close.rolling(10).mean()
    df["MA50"] = close.rolling(50).mean()
    df["MA10_50_ratio"] = df["MA10"] / df["MA50"]
    df["Daily_Return"] = close.pct_change()
    df["Return_5d"] = close.pct_change(5)
    df["Volatility"] = df["Daily_Return"].rolling(10).std()

    bb = ta.volatility.BollingerBands(close)
    df["BB_pct"] = (close - bb.bollinger_lband())/(bb.bollinger_hband() - bb.bollinger_lband())
    df["RSI"] = ta.momentum.rsi(close, window=14)

    return df

# Data loaders
@st.cache_data(show_spinner=False)
def load_training_data(ticker):
    df = yf.download(ticker, start="2020-01-01", end="2025-01-01", progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df

@st.cache_data(ttl=60, show_spinner=False)
def load_live_data(ticker, period="3mo"):
    period_days = {"1mo": 30, "3mo": 90, "6mo": 180, "1y": 365, "2y": 730}
    days  = period_days.get(period, 90)
    start = (datetime.now() - timedelta(days=days + 100)).strftime("%Y-%m-%d")  # ← key line
    df = yf.download(ticker, start=start, auto_adjust=True, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df

# Chart display. YTD and ALL have calendar meaning rather
# than a fixed day count, so they're computed explicitly below.
CHART_PERIODS = {
    "1D":  {"kind": "intraday", "interval": "5m",  "days": 1},
    "5D":  {"kind": "intraday", "interval": "15m", "days": 5},
    "1M":  {"kind": "daily", "days": 30},
    "6M":  {"kind": "daily", "days": 182},
    "YTD": {"kind": "ytd"},
    "1Y":  {"kind": "daily", "days": 365},
    "5Y":  {"kind": "daily", "days": 365 * 5},
    "ALL": {"kind": "all"},
}

@st.cache_data(ttl=60, show_spinner=False)
def load_chart_data(ticker, period="1Y"):
    # Fetch data for the Price Chart tab only, separate from
    # load_live_data() above which feeds the model's prediction.
    config = CHART_PERIODS.get(period, CHART_PERIODS["1Y"])
    kind = config["kind"]

    if kind == "intraday":
        df = yf.download(ticker, period=f"{config['days']}d", interval=config["interval"],
                          auto_adjust=True, progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        return df, config["interval"], None

    today = datetime.now()
    if kind == "ytd":
        display_start = datetime(today.year, 1, 1)
    elif kind == "all":
        display_start = None
    else:
        display_start = today - timedelta(days=config["days"])

    if display_start is None:
        df = yf.download(ticker, period="max", interval="1d", auto_adjust=True, progress=False)
    else:
        fetch_start = (display_start - timedelta(days=100)).strftime("%Y-%m-%d")
        df = yf.download(ticker, start=fetch_start, interval="1d", auto_adjust=True, progress=False)

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    return df, "1d", display_start

# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
MODEL_PATH = "model.pkl"
SCALER_PATH = "scaler.pkl"

@st.cache_resource(show_spinner=False)
def load_trained_model():
    """Load the model and scaler saved by the notebook's final retrain
    step (Step 7 or Step 14). Both files must sit in the same folder this
    app is run from. Returns (None, None) if they are missing, so the app
    can show a clear message instead of crashing.
    """
    if not (os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH)):
        return None, None
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    return model, scaler

# ---------------------------------------------------------------------------
# Explanation generator (Ollama)
# ---------------------------------------------------------------------------
@st.cache_data(ttl=3600, show_spinner=False)
def explain_signal(ticker, signal, confidence):
    # Ask a local LLM (via Ollama) to turn the raw signal into a short, plain-English note for a beginner investor.
    clean_signal = "Bullish" if "BULLISH" in signal else "Bearish"
    prompt = (
        f"A stock prediction tool is showing a beginner investor a {clean_signal} "
        f"signal for {ticker}, with {confidence:.1%} confidence, over the next 20 "
        f"trading days. Write a 2-3 sentence explanation, speaking directly to "
        f"this user in plain, natural English.\n\n"
        f"Rules:\n"
        f"- The first sentence should be: A {clean_signal} signal for {ticker} at {confidence:.1%} means" 
        f" the prediction [continue from here] \n"
        f"- Be precise and honest about how strong the confidence actually is, "
        f"using this scale: 50-55% should be described as weak or uncertain, not as meaningful confidence. "
        f"55-65% is a mild lean. 65-80% is moderate confidence. Above 80% is "
        f"strong confidence. Never call 50-55% 'relatively high' or 'fairly sure'.\n"
        f"- Do not give financial advice like telling them to buy or sell.\n"
        f"- Do not add any preamble, sign-off, or meta-commentary - only the "
        f"explanation itself."
    )
    try:
        response = ollama.chat(
            model="llama3.2",
            messages=[{"role": "user", "content": prompt}],
        )
        return response["message"]["content"]
    except Exception:
        return "Explanation unavailable right now - please try again shortly."



# ---------------------------------------------------------------------------
# Backtesting (Backtrader)
# ---------------------------------------------------------------------------
class SignalData(bt.feeds.PandasData):
    """Extends Backtrader's standard OHLCV feed with one extra column: the
    model's daily signal (1 = Bullish, 0 = Bearish)."""
    lines = ("signal",)
    params = (("signal", -1),)


class SignalStrategy(bt.Strategy):
    """Simplest possible long/flat rule: fully invested when the model says
    Bullish, fully in cash when it says Bearish."""
    def next(self):
        signal = self.data.signal[0]
        if signal == 1 and not self.position:
            self.order_target_percent(target=1.0)
        elif signal == 0 and self.position:
            self.order_target_percent(target=0.0)


class BuyAndHoldStrategy(bt.Strategy):
    """Benchmark: buy once at the start of the test period, hold throughout."""
    def next(self):
        if not self.position:
            self.order_target_percent(target=1.0)


class ValueTracker(bt.Analyzer):
    """Records the portfolio's value every day, so the equity curve can be
    plotted afterwards."""
    def start(self):
        self.values = []
    def next(self):
        self.values.append(self.strategy.broker.getvalue())


def run_single_backtest(strategy_cls, feed_df, starting_cash=10000):
    cerebro = bt.Cerebro()
    cerebro.broker.setcash(starting_cash)
    cerebro.broker.setcommission(commission=0.001)  # 0.1% per trade
    cerebro.adddata(SignalData(dataname=feed_df))
    cerebro.addstrategy(strategy_cls)
    cerebro.addanalyzer(bt.analyzers.Returns, _name="returns")
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name="drawdown")
    cerebro.addanalyzer(ValueTracker, _name="tracker")

    result = cerebro.run()
    strat = result[0]
    return {
        "total_return_pct": strat.analyzers.returns.get_analysis().get("rtot", 0) * 100,
        "max_drawdown_pct": strat.analyzers.drawdown.get_analysis().max.drawdown,
        "value_series": strat.analyzers.tracker.values,
    }


@st.cache_data(show_spinner=False)
def backtest_ticker(ticker, _reference_model):
    df = load_training_data(ticker)
    if df.empty:
        return None

    df = add_features(df)
    df["Target"] = (df["Close"].squeeze().shift(-HORIZON) > df["Close"].squeeze()).astype(int)
    df.dropna(inplace=True)

    split = int(len(df) * 0.8)
    train, test = df.iloc[:split].copy(), df.iloc[split:].copy()

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(train[FEATURES])
    X_test_s = scaler.transform(test[FEATURES])

    # Fresh, untrained instance of the same class + hyperparameters as the loaded model
    model = type(_reference_model)(**_reference_model.get_params())
    model.fit(X_train_s, train["Target"])

    test["Signal"] = model.predict(X_test_s)
    feed_df = test[["Open", "High", "Low", "Close", "Volume", "Signal"]].rename(columns={"Signal": "signal"})

    strategy_result = run_single_backtest(SignalStrategy, feed_df)
    buyhold_result = run_single_backtest(BuyAndHoldStrategy, feed_df)

    dates = feed_df.index[-len(strategy_result["value_series"]):]

    return {
        "dates": dates,
        "strategy_values": strategy_result["value_series"],
        "buyhold_values": buyhold_result["value_series"],
        "strategy_return": strategy_result["total_return_pct"],
        "buyhold_return": buyhold_result["total_return_pct"],
        "strategy_dd": strategy_result["max_drawdown_pct"],
        "buyhold_dd": buyhold_result["max_drawdown_pct"],
        "test_start": test.index[0].date(),
        "test_end": test.index[-1].date(),
    }


# ---------------------------------------------------------------------------
# Small visual helpers
# ---------------------------------------------------------------------------
def render_signal_badge(signal):
    is_bullish = "BULLISH" in signal
    color = "#22c55e" if is_bullish else "#ef4444"
    bg    = "rgba(34,197,94,0.15)" if is_bullish else "rgba(239,68,68,0.15)"
    label = "▲ BULLISH" if is_bullish else "▼ BEARISH"
    st.markdown(
        f'<div style="display:inline-block; padding:0.55rem 1.3rem; '
        f'border-radius:999px; background:{bg}; border:1.5px solid {color}; '
        f'color:{color}; font-weight:700; font-size:1.4rem; letter-spacing:0.02em;">'
        f'{label}</div>',
        unsafe_allow_html=True,
    )


# Light custom styling
st.markdown(
    """
    <style>
    div[data-testid="stMetricValue"] { font-size: 1.7rem; }
    div[data-testid="stMetricLabel"] { font-size: 0.95rem; opacity: 0.8; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
st.title("📈 Financial Advisor Bot")
st.caption("CM3070 Final Project Financial Advisor Bot")

# Sidebar
with st.sidebar:
    st.header("⚙️ Settings")

    ticker_choice = st.selectbox(
        "Stock ticker",
        options=list(CURATED_TICKERS.keys()) + ["Custom ticker..."],
        format_func=lambda t: f"{t} — {CURATED_TICKERS[t]}" if t in CURATED_TICKERS else t,
    )
    if ticker_choice == "Custom ticker...":
        ticker = st.text_input("Enter any ticker symbol", value="").strip().upper()
    else:
        ticker = ticker_choice

    period = st.selectbox(
        "Chart period",
        ["1D", "5D", "1M", "6M", "YTD", "1Y", "5Y", "ALL"],
        index=5,  # defaults to "1Y"
    )
    st.divider()
    st.caption("Uses the pre-trained model from model_training.ipynb.")
    st.caption("Live signal uses the most recent 3 months.")

if not ticker:
    st.warning("Enter a ticker symbol in the sidebar to get started.")
    st.stop()

# Load the trained model (once - cached across all tickers)
model, scaler = load_trained_model()

if model is None:
    st.error(
        f"Could not find `{MODEL_PATH}` and `{SCALER_PATH}` in this folder. "
        f"Run the training notebook (model_training.ipynb) first, then copy "
        f"both files into the same folder as this app."
    )
    st.stop()

# Live prediction - always uses a fixed daily-bar fetch, independent of
# whatever period the user picks for the Price Chart tab.
live = load_live_data(ticker, "3mo")

if live.empty:
    st.error(f"Could not load data for **{ticker}**. Check the ticker symbol and try again.")
    st.stop()

live = add_features(live)
live.dropna(inplace=True)

row        = live[FEATURES].iloc[[-1]]
pred       = model.predict(scaler.transform(row))[0]
confidence = model.predict_proba(scaler.transform(row))[0].max()
price      = float(live["Close"].iloc[-1])
prev_price = float(live["Close"].iloc[-2])
day_change = price - prev_price
day_pct    = day_change / prev_price * 100
date       = live.index[-1].date()
signal     = "BULLISH 📈" if pred == 1 else "BEARISH 📉"

company_name = CURATED_TICKERS.get(ticker, ticker)
st.subheader(f"Live Signal — {ticker} ({company_name})" if ticker in CURATED_TICKERS else f"Live Signal — {ticker}")

tab_overview, tab_chart, tab_backtest = st.tabs(
    ["🏠 Overview", "📈 Price Chart", "💰 Backtest"]
)

# OVERVIEW TAB
with tab_overview:
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Current Price", f"${price:.2f}", f"{day_change:+.2f}  ({day_pct:+.2f}%)")
    with c2:
        st.caption("Signal")
        render_signal_badge(signal)
    with c3:
        st.metric("Confidence", f"{confidence:.1%}")
        st.progress(float(confidence))

    st.caption(f"Signal as of **{date}**. Predicts whether {ticker} will be higher in ~{HORIZON} trading days.")

    with st.spinner("Generating explanation..."):
        explanation = explain_signal(ticker, signal, confidence)
    st.info(f"**What this means:** {explanation}")

# CHART TAB
with tab_chart:
    chart_df, chart_interval, display_start = load_chart_data(ticker, period)

    if chart_df.empty:
        st.error(f"Could not load chart data for **{ticker}** at this period.")
    else:
        is_daily = (chart_interval == "1d")

        if is_daily:
            st.subheader("Price Chart with Moving Averages")
            chart_df = chart_df.copy()
            chart_df["MA10"] = chart_df["Close"].rolling(10).mean()
            chart_df["MA50"] = chart_df["Close"].rolling(50).mean()

            bb = ta.volatility.BollingerBands(chart_df["Close"].squeeze())
            chart_df["BB_upper"] = bb.bollinger_hband()
            chart_df["BB_lower"] = bb.bollinger_lband()
            chart_df["Daily_Return"] = chart_df["Close"].pct_change() * 100

            if display_start is not None:
                cutoff = pd.Timestamp(display_start)
                idx_tz = chart_df.index.tz
                if idx_tz is not None:
                    cutoff = cutoff.tz_localize(idx_tz)
                chart_df = chart_df[chart_df.index >= cutoff]
        else:
            st.subheader(f"Price Chart ({chart_interval} bars)")
            st.caption(
                "Short periods use intraday bars for a readable chart. "
                "Moving averages, Bollinger Bands, and daily returns aren't "
                "shown here since they're daily-based indicators - see 1M "
                "and longer for those."
            )

        close = chart_df["Close"].squeeze()
        is_up = float(close.iloc[-1]) >= float(close.iloc[0])
        line_clr = "#22c55e" if is_up else "#ef4444"
        fill_clr = "rgba(34,197,94,0.15)" if is_up else "rgba(239,68,68,0.15)"
        min_price = float(close.min()) * 0.998

        fig_chart = go.Figure()

        if is_daily:
            # Upper band first, then lower band filled back up to it,
            # draws a shaded envelope between the two bands, 
            # added before Close/MA so it sits behind them.
            fig_chart.add_trace(go.Scatter(
                x=chart_df.index, y=chart_df["BB_upper"],
                name="Bollinger Band", legendgroup="bb",
                line=dict(color="rgba(139,92,246,0.5)", width=1, dash="dot"),
                hovertemplate="BB Upper: $%{y:.2f}<extra></extra>",
            ))
            fig_chart.add_trace(go.Scatter(
                x=chart_df.index, y=chart_df["BB_lower"],
                name="Bollinger Band", legendgroup="bb", showlegend=False,
                fill="tonexty", fillcolor="rgba(139,92,246,0.08)",
                line=dict(color="rgba(139,92,246,0.5)", width=1, dash="dot"),
                hovertemplate="BB Lower: $%{y:.2f}<extra></extra>",
            ))

        fig_chart.add_trace(go.Scatter(
            x=chart_df.index, y=[min_price] * len(chart_df),
            line=dict(color="rgba(0,0,0,0)", width=0),
            showlegend=False, hoverinfo="skip",
        ))

        fig_chart.add_trace(go.Scatter(
            x=chart_df.index, y=close,
            name="Close", fill="tonexty", fillcolor=fill_clr,
            line=dict(color=line_clr, width=1.5),
            hovertemplate="$%{y:.2f}<extra></extra>",
        ))

        if is_daily:
            fig_chart.add_trace(go.Scatter(
                x=chart_df.index, y=chart_df["MA10"],
                name="MA10", line=dict(color="#22c55e", width=1.5),
                hovertemplate="MA10: $%{y:.2f}<extra></extra>",
            ))

            fig_chart.add_trace(go.Scatter(
                x=chart_df.index, y=chart_df["MA50"],
                name="MA50", line=dict(color="#f59e0b", width=1.5),
                hovertemplate="MA50: $%{y:.2f}<extra></extra>",
            ))

        fig_chart.update_layout(
            template="plotly_dark",
            plot_bgcolor="#0d1117", paper_bgcolor="#0d1117",
            xaxis=dict(showgrid=False, zeroline=False,
                       showspikes=True, spikecolor="#555", spikethickness=1),
            yaxis=dict(showgrid=True, gridcolor="#1e2530",
                       zeroline=False, tickprefix="$", side="right"),
            hovermode="x unified",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
            margin=dict(l=0, r=0, t=40, b=0),
            height=460,
        )

        st.plotly_chart(fig_chart, use_container_width=True)

        if is_daily:
            st.subheader("Daily Returns")
            returns = chart_df["Daily_Return"]
            bar_colors = ["#22c55e" if r >= 0 else "#ef4444" for r in returns]

            fig_returns = go.Figure()
            fig_returns.add_trace(go.Bar(
                x=chart_df.index, y=returns,
                marker=dict(color=bar_colors),
                hovertemplate="%{y:+.2f}%<extra></extra>",
            ))
            fig_returns.update_layout(
                template="plotly_dark",
                plot_bgcolor="#0d1117", paper_bgcolor="#0d1117",
                xaxis=dict(showgrid=False, zeroline=False),
                yaxis=dict(showgrid=True, gridcolor="#1e2530",
                           zeroline=True, zerolinecolor="#555",
                           ticksuffix="%", side="right"),
                margin=dict(l=0, r=0, t=10, b=0),
                height=200,
                showlegend=False,
            )
            st.plotly_chart(fig_returns, use_container_width=True)

# BACKTEST TAB
with tab_backtest:
    st.subheader(f"Would the signal have beaten buy-and-hold on {ticker}?")

    with st.spinner("Running backtest..."):
        bt_result = backtest_ticker(ticker, model)

    if bt_result is None:
        st.error("Could not run the backtest for this ticker.")
    else:
        b1, b2 = st.columns(2)
        with b1:
            st.metric(
                "Signal Strategy return",
                f"{bt_result['strategy_return']:.2f}%",
                help="Long when Bullish, cash when Bearish, over the test period"
            )
            st.metric("Signal Strategy max drawdown", f"{bt_result['strategy_dd']:.2f}%")
        with b2:
            st.metric(
                "Buy & Hold return",
                f"{bt_result['buyhold_return']:.2f}%",
                help="Simple benchmark: buy at the start, hold throughout"
            )
            st.metric("Buy & Hold max drawdown", f"{bt_result['buyhold_dd']:.2f}%")

        st.caption(f"Test period: {bt_result['test_start']} to {bt_result['test_end']}")

        fig_bt = go.Figure()
        fig_bt.add_trace(go.Scatter(
            x=bt_result["dates"], y=bt_result["strategy_values"],
            name="Signal Strategy", line=dict(color="#1f5f5b", width=2),
        ))
        fig_bt.add_trace(go.Scatter(
            x=bt_result["dates"], y=bt_result["buyhold_values"],
            name="Buy & Hold", line=dict(color="#b5482e", width=2),
        ))
        fig_bt.update_layout(
            template="plotly_dark",
            plot_bgcolor="#0d1117", paper_bgcolor="#0d1117",
            yaxis=dict(title="Portfolio Value ($)", gridcolor="#1e2530"),
            xaxis=dict(showgrid=False),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
            margin=dict(l=0, r=0, t=40, b=0),
            height=380,
        )
        st.plotly_chart(fig_bt, use_container_width=True)

        beats = bt_result["strategy_return"] > bt_result["buyhold_return"]
        if beats:
            st.success(f"The Signal Strategy outperformed Buy & Hold on {ticker} over this test period.")
        else:
            st.warning(
                f"The Signal Strategy underperformed Buy & Hold on {ticker} over this test period, "
                f"though it still produced a lower maximum drawdown."
            )
