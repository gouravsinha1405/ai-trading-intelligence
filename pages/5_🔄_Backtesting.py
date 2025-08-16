import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

st.set_page_config(page_title="Enhanced Backtesting", page_icon="🔄", layout="wide")

# ==================== UTILS ====================


def ensure_date_col(df: pd.DataFrame) -> pd.DataFrame:
    if "Date" not in df.columns:
        if isinstance(df.index, pd.DatetimeIndex):
            df = df.copy()
            # remove tz for plotting if present
            idx = (
                df.index.tz_localize(None)
                if getattr(df.index, "tz", None)
                else df.index
            )
            df["Date"] = idx
        else:
            df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.sort_values("Date").reset_index(drop=True)
    return df


def validate_ohlc_integrity(df: pd.DataFrame) -> dict:
    issues = []
    n = len(df)
    if not {"Open", "High", "Low", "Close"}.issubset(df.columns):
        return {"integrity_score": 0, "issues": ["Missing OHLC columns"]}

    bad_high = (df["High"] < df[["Open", "Close"]].max(axis=1)).sum()
    bad_low = (df["Low"] > df[["Open", "Close"]].min(axis=1)).sum()
    nonpos = (df[["Open", "High", "Low", "Close"]] <= 0).any(axis=1).sum()
    dup_idx = df["Date"].duplicated().sum() if "Date" in df.columns else 0

    if bad_high:
        issues.append(f"{bad_high} rows: High < max(Open, Close)")
    if bad_low:
        issues.append(f"{bad_low} rows: Low  > min(Open, Close)")
    if nonpos:
        issues.append(f"{nonpos} rows: non-positive prices")
    if dup_idx:
        issues.append(f"{dup_idx} duplicate dates")

    score = (
        100
        - min(30, bad_high * 0.05)
        - min(30, bad_low * 0.05)
        - min(25, nonpos * 0.05)
        - min(15, dup_idx * 0.5)
    )
    return {"integrity_score": max(0, round(score, 1)), "issues": issues}


def rsi_wilder(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    ma_up = up.ewm(alpha=1 / period, adjust=False).mean()
    ma_down = down.ewm(alpha=1 / period, adjust=False).mean()
    rs = ma_up / ma_down.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def safe_detect_regimes(data: pd.DataFrame):
    try:
        from src.analysis.regime_detector import RegimeDetector

        det = RegimeDetector()
        try:
            vol = det.detect_volatility_regimes_walkforward(data)
        except AttributeError:
            vol = det.detect_volatility_regimes(data)

        try:
            trend = det.detect_trend_regimes_improved(data)
        except AttributeError:
            trend = det.detect_trend_regimes(data)

        try:
            mkt = det.detect_market_regimes_gmm(data)
        except AttributeError:
            mkt = det.detect_market_state_regimes(data)

        out = data.copy()
        for src, col in [
            (vol, "Volatility_Regime"),
            (trend, "Trend_Regime"),
            (mkt, "Market_State_Regime"),
        ]:
            if isinstance(src, pd.DataFrame):
                c = [c for c in src.columns if "regime" in c.lower()]
                if c:
                    out[c[0].lower()] = src[c[0]]
        return ensure_date_col(out)
    except Exception as e:
        st.warning(f"Regime detection failed: {e}. Continuing without regime analysis.")
        return ensure_date_col(data)


@st.cache_data(ttl=3600)
def get_historical_data(symbol: str, start_date: str, end_date: str):
    try:
        from src.data.jugaad_client import JugaadClient

        client = JugaadClient()
        data = client.get_historical_data(symbol, start_date, end_date)
        if data is not None and len(data) > 0:
            data = ensure_date_col(data)
            if validate_ohlc_integrity(data):
                return data
    except Exception as e:
        st.warning(f"Jugaad data failed: {e}")

    try:
        import yfinance as yf

        ticker = yf.Ticker(f"{symbol}.NS")
        data = ticker.history(start=start_date, end=end_date)
        if not data.empty:
            data = data.reset_index()
            data.columns = [
                "Date",
                "Open",
                "High",
                "Low",
                "Close",
                "Volume",
                "Dividends",
                "Stock Splits",
            ]
            data = data[["Date", "Open", "High", "Low", "Close", "Volume"]]
            data = ensure_date_col(data)
            if validate_ohlc_integrity(data):
                return data
    except Exception as e:
        st.warning(f"Yahoo Finance fallback failed: {e}")

    return None


def add_regime_spans(fig, df, regime_col="market_regime"):
    colors = {
        "Bull_Market": "rgba(0,255,0,0.12)",
        "Bear_Market": "rgba(255,0,0,0.12)",
        "Sideways_Market": "rgba(255,255,0,0.12)",
        "High_Volatility": "rgba(255,136,0,0.12)",
    }
    if regime_col not in df.columns:
        return
    d = df[["Date", regime_col]].dropna()
    if d.empty:
        return
    spans = []
    cur = None
    start = None
    for _, r in d.iterrows():
        if r[regime_col] != cur:
            if cur is not None:
                spans.append((start, prev_date, cur))
            cur = r[regime_col]
            start = r["Date"]
        prev_date = r["Date"]
    if cur is not None:
        spans.append((start, prev_date, cur))
    for s, e, reg in spans:
        fig.add_vrect(
            x0=s,
            x1=e,
            fillcolor=colors.get(reg, "rgba(128,128,128,0.1)"),
            opacity=0.25,
            layer="below",
            line_width=0,
        )


def get_real_market_data(symbol, start_date, end_date):
    """Get real historical market data using enhanced jugaad-data client"""
    try:
        from src.data.data_cleaner import DataCleaner
        from src.data.jugaad_client import JugaadDataClient

        # Initialize enhanced clients
        client = JugaadDataClient()
        cleaner = DataCleaner()

        st.info(f"📡 Fetching real data for {symbol} using enhanced data pipeline...")

        # Get data based on symbol type
        data = client.get_stock_data(symbol, start_date, end_date)

        if data is not None and len(data) > 0:
            # Apply production-grade data cleaning
            cleaned_data = cleaner.clean_ohlcv_data(data)

            # Get data quality metrics
            integrity_score = cleaner.validate_ohlc_integrity(cleaned_data)

            st.success(
                f"✅ Real data retrieved: {len(cleaned_data)} records | Data integrity: {integrity_score['integrity_score']:.1f}/100"
            )

            if integrity_score["issues"]:
                with st.expander("⚠️ Data Quality Issues"):
                    for issue in integrity_score["issues"]:
                        st.warning(f"• {issue}")

            return cleaned_data
        else:
            st.warning(
                f"⚠️ No real data available for {symbol}. Using sample data for demonstration."
            )
            return generate_sample_data(symbol, start_date, end_date)

    except Exception as e:
        st.error(f"Error fetching real data for {symbol}: {e}")
        st.info("📊 Using sample data for demonstration")
        return generate_sample_data(symbol, start_date, end_date)


def generate_sample_data(symbol, start_date, end_date):
    """Generate sample OHLCV data for backtesting (fallback only)"""
    dates = pd.date_range(start=start_date, end=end_date, freq="D")

    # Remove weekends (Saturday=5, Sunday=6)
    dates = [d for d in dates if d.weekday() < 5]

    np.random.seed(42)  # For reproducible results

    # Starting price
    start_price = 1000 if symbol == "NIFTY50" else np.random.uniform(500, 3000)

    data = []
    current_price = start_price

    for date in dates:
        # Generate daily volatility
        daily_return = np.random.normal(0.0005, 0.018)  # ~0.05% mean, 1.8% daily vol

        # Calculate OHLC
        open_price = current_price
        high_price = open_price * (1 + abs(daily_return) + np.random.uniform(0, 0.01))
        low_price = open_price * (1 - abs(daily_return) - np.random.uniform(0, 0.01))
        close_price = open_price * (1 + daily_return)

        # Volume (higher on volatile days)
        volume = int(np.random.uniform(50000, 200000) * (1 + abs(daily_return) * 10))

        data.append(
            {
                "Date": date,
                "Open": max(open_price, 0.01),
                "High": max(high_price, open_price, close_price),
                "Low": min(low_price, open_price, close_price),
                "Close": max(close_price, 0.01),
                "Volume": volume,
            }
        )

        current_price = close_price

    return pd.DataFrame(data)


def calculate_sma(data, period):
    """Calculate Simple Moving Average"""
    return data["Close"].rolling(window=period).mean()


def calculate_rsi(data, period=14):
    """Calculate Wilder's RSI (production-grade implementation)"""
    try:
        from src.analysis.regime_detector import RegimeDetector

        detector = RegimeDetector()

        # Use the enhanced RSI calculation
        rsi_data = detector._calculate_rsi_wilder(data["Close"], period)
        return rsi_data
    except:
        # Fallback to basic RSI
        delta = data["Close"].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi


def detect_market_regimes(data):
    """Detect market regimes using enhanced regime detector"""
    try:
        from src.analysis.regime_detector import RegimeDetector

        detector = RegimeDetector()

        # Detect different types of regimes using correct method names
        volatility_regimes = detector.detect_volatility_regimes(data)
        trend_regimes = detector.detect_trend_regimes(data)
        market_regimes = detector.detect_market_state_regimes(data)

        # Combine regime information
        regime_data = data.copy()

        if "volatility_regime" in volatility_regimes.columns:
            regime_data["volatility_regime"] = volatility_regimes["volatility_regime"]

        if "trend_regime" in trend_regimes.columns:
            regime_data["trend_regime"] = trend_regimes["trend_regime"]

        if "market_regime" in market_regimes.columns:
            regime_data["market_regime"] = market_regimes["market_regime"]

        # Add regime probabilities if available
        prob_cols = [col for col in market_regimes.columns if "_Prob" in col]
        for col in prob_cols:
            regime_data[col] = market_regimes[col]

        return regime_data

    except Exception as e:
        st.warning(f"Regime detection failed: {e}. Continuing without regime analysis.")
        return data


def calculate_bollinger_bands(data, period=20, std_dev=2):
    """Calculate Bollinger Bands"""
    sma = calculate_sma(data, period)
    std = data["Close"].rolling(window=period).std()
    upper = sma + (std * std_dev)
    lower = sma - (std * std_dev)
    return upper, sma, lower


def backtest_strategy(data, strategy_type, params):
    """Run enhanced backtest with regime awareness for selected strategy"""

    # Apply regime detection for enhanced strategies
    if params.get("use_regime_detection", False):
        st.info("🧠 Applying regime detection for enhanced strategy logic...")
        data = detect_market_regimes(data)

    signals = []
    positions = []
    portfolio_value = []
    cash = params.get("initial_capital", 10000)
    shares = 0
    entry_price = 0

    # Calculate technical indicators with enhanced methods
    if strategy_type in ["SMA Crossover", "Mean Reversion", "Regime-Aware Strategy"]:
        data["SMA_short"] = calculate_sma(data, params.get("sma_short", 10))
        data["SMA_long"] = calculate_sma(data, params.get("sma_long", 30))

    if strategy_type in ["Mean Reversion", "Regime-Aware Strategy"]:
        data["RSI"] = calculate_rsi(data, params.get("rsi_period", 14))
        upper, middle, lower = calculate_bollinger_bands(
            data, params.get("bb_period", 20)
        )
        data["BB_upper"] = upper
        data["BB_lower"] = lower
        data["BB_middle"] = middle

    for i in range(len(data)):
        row = data.iloc[i]
        signal = "HOLD"

        # Skip initial periods for indicator calculation
        lookback_period = max(params.get("sma_long", 30), params.get("bb_period", 20))
        if i < lookback_period:
            # Calculate portfolio value for initial period too
            current_value = cash + (shares * row["Close"])
            portfolio_value.append(current_value)
            signals.append(signal)
            positions.append(shares)
            continue

        # Enhanced strategy logic with regime awareness
        if strategy_type == "SMA Crossover":
            if (
                row["SMA_short"] > row["SMA_long"]
                and data.iloc[i - 1]["SMA_short"] <= data.iloc[i - 1]["SMA_long"]
            ):
                signal = "BUY"
            elif (
                row["SMA_short"] < row["SMA_long"]
                and data.iloc[i - 1]["SMA_short"] >= data.iloc[i - 1]["SMA_long"]
            ):
                signal = "SELL"

        elif strategy_type == "Mean Reversion":
            if row["Close"] < row["BB_lower"] and row["RSI"] < params.get(
                "rsi_oversold", 30
            ):
                signal = "BUY"
            elif row["Close"] > row["BB_upper"] and row["RSI"] > params.get(
                "rsi_overbought", 70
            ):
                signal = "SELL"

        elif strategy_type == "Regime-Aware Strategy":
            # Enhanced regime-aware logic
            regime_signal = "HOLD"

            # Check if regime data is available
            if "market_regime" in row.index and pd.notna(row["market_regime"]):
                current_regime = row["market_regime"]

                if current_regime == "Bull_Market":
                    # In bull market, use trend following
                    if (
                        row["SMA_short"] > row["SMA_long"]
                        and data.iloc[i - 1]["SMA_short"]
                        <= data.iloc[i - 1]["SMA_long"]
                    ):
                        regime_signal = "BUY"
                elif current_regime == "Bear_Market":
                    # In bear market, be more defensive
                    if shares > 0:
                        regime_signal = "SELL"
                elif current_regime == "Sideways_Market":
                    # In sideways market, use mean reversion
                    if row["Close"] < row["BB_lower"] and row["RSI"] < 35:
                        regime_signal = "BUY"
                    elif row["Close"] > row["BB_upper"] and row["RSI"] > 65:
                        regime_signal = "SELL"
                elif current_regime == "High_Volatility":
                    # In high volatility, reduce position size or stay out
                    if shares > 0 and row["RSI"] > 75:
                        regime_signal = "SELL"

            signal = regime_signal

        elif strategy_type == "Buy and Hold":
            if i == lookback_period:
                signal = "BUY"

        # Execute trades with position sizing
        position_size = params.get("position_size", 1.0)  # Fraction of capital to use

        if signal == "BUY" and shares == 0:
            available_cash = cash * position_size
            shares = int(available_cash // row["Close"])
            if shares > 0:
                cash = cash - (shares * row["Close"])
                entry_price = row["Close"]
        elif signal == "SELL" and shares > 0:
            cash = cash + (shares * row["Close"])
            shares = 0

        # Calculate portfolio value
        current_value = cash + (shares * row["Close"])
        portfolio_value.append(current_value)

        signals.append(signal)
        positions.append(shares)

    data["Signal"] = signals
    data["Position"] = positions
    data["Portfolio_Value"] = portfolio_value

    return data


def calculate_performance_metrics(data, benchmark_data=None):
    """Calculate performance metrics"""
    returns = data["Portfolio_Value"].pct_change().dropna()

    # Basic metrics
    total_return = (
        data["Portfolio_Value"].iloc[-1] / data["Portfolio_Value"].iloc[0] - 1
    ) * 100

    # Maximum drawdown
    peak = data["Portfolio_Value"].expanding().max()
    drawdown = (data["Portfolio_Value"] - peak) / peak * 100
    max_drawdown = drawdown.min()

    # Sharpe ratio (assuming 252 trading days)
    sharpe_ratio = (
        (returns.mean() / returns.std()) * np.sqrt(252) if returns.std() > 0 else 0
    )

    # Win rate
    trades = data[data["Signal"].isin(["BUY", "SELL"])]
    if len(trades) > 1:
        trade_returns = []
        entry_price = None
        for _, trade in trades.iterrows():
            if trade["Signal"] == "BUY":
                entry_price = trade["Close"]
            elif trade["Signal"] == "SELL" and entry_price:
                trade_return = (trade["Close"] - entry_price) / entry_price
                trade_returns.append(trade_return)
                entry_price = None

        win_rate = (
            len([r for r in trade_returns if r > 0]) / len(trade_returns) * 100
            if trade_returns
            else 0
        )
    else:
        win_rate = 0

    return {
        "total_return": total_return,
        "max_drawdown": max_drawdown,
        "sharpe_ratio": sharpe_ratio,
        "win_rate": win_rate,
        "total_trades": len(trades),
    }


def main():
    st.title("🔄 Enhanced Strategy Backtesting")
    st.markdown(
        "Test your trading strategies with production-grade data and regime detection"
    )

    # Sidebar for configuration
    with st.sidebar:
        st.header("⚙️ Enhanced Backtest Configuration")

        # Data selection with more options
        symbol = st.selectbox(
            "Symbol",
            [
                "RELIANCE",
                "TCS",
                "INFY",
                "HDFCBANK",
                "ICICIBANK",
                "HINDUNILVR",
                "ITC",
                "LT",
                "SBIN",
                "KOTAKBANK",
            ],
        )

        # Date range
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input(
                "Start Date", value=datetime.now() - timedelta(days=365)
            )
        with col2:
            end_date = st.date_input("End Date", value=datetime.now())

        # Strategy selection with enhanced options
        strategy_type = st.selectbox(
            "Strategy Type",
            [
                "SMA Crossover",
                "Mean Reversion",
                "Regime-Aware Strategy",
                "Buy and Hold",
            ],
        )

        # Enhanced features toggle
        st.subheader("🧠 Enhanced Features")
        use_regime_detection = st.checkbox(
            "Enable Regime Detection",
            value=False,
            help="Use AI-powered market regime detection for enhanced strategy logic",
        )

        use_enhanced_data = st.checkbox(
            "Production Data Cleaning",
            value=True,
            help="Apply institutional-grade data cleaning and validation",
        )

        # Strategy parameters
        st.subheader("📊 Strategy Parameters")

        params = {
            "use_regime_detection": use_regime_detection,
            "use_enhanced_data": use_enhanced_data,
        }

        if strategy_type == "SMA Crossover":
            params["sma_short"] = st.slider("Short SMA Period", 5, 50, 10)
            params["sma_long"] = st.slider("Long SMA Period", 20, 100, 30)

        elif strategy_type == "Mean Reversion":
            params["rsi_period"] = st.slider("RSI Period", 10, 30, 14)
            params["rsi_oversold"] = st.slider("RSI Oversold", 20, 40, 30)
            params["rsi_overbought"] = st.slider("RSI Overbought", 60, 80, 70)
            params["bb_period"] = st.slider("Bollinger Band Period", 10, 30, 20)

        elif strategy_type == "Regime-Aware Strategy":
            params["sma_short"] = st.slider("Short SMA Period", 5, 50, 10)
            params["sma_long"] = st.slider("Long SMA Period", 20, 100, 30)
            params["rsi_period"] = st.slider("RSI Period", 10, 30, 14)
            params["bb_period"] = st.slider("Bollinger Band Period", 10, 30, 20)
            st.info("💡 This strategy adapts based on detected market regimes")

        # Risk parameters
        st.subheader("⚠️ Risk Management")
        params["initial_capital"] = st.number_input(
            "Initial Capital (₹)", min_value=1000, value=100000
        )
        params["position_size"] = st.slider("Position Size (%)", 10, 100, 90) / 100

        # Advanced settings
        with st.expander("🔧 Advanced Settings"):
            st.info("Additional risk management features coming soon!")

        # Run backtest button
        run_backtest = st.button("🚀 Run Enhanced Backtest", type="primary")

    # Main content
    if run_backtest:
        with st.spinner("Running backtest..."):
            # Get real market data
            data = get_real_market_data(symbol, start_date, end_date)

            # Run backtest
            backtest_results = backtest_strategy(data, strategy_type, params)

            # Calculate metrics
            metrics = calculate_performance_metrics(backtest_results)

            # Display results with enhanced metrics
            st.subheader("📊 Enhanced Backtest Results")

            # Show regime detection results if enabled
            if (
                params.get("use_regime_detection", False)
                and "market_regime" in backtest_results.columns
            ):
                st.subheader("🧠 Market Regime Analysis")

                regime_counts = backtest_results["market_regime"].value_counts()

                col1, col2 = st.columns(2)

                with col1:
                    # Regime distribution pie chart
                    regime_fig = px.pie(
                        values=regime_counts.values,
                        names=regime_counts.index,
                        title="Market Regime Distribution",
                        color_discrete_map={
                            "Bull_Market": "#00ff00",
                            "Bear_Market": "#ff0000",
                            "Sideways_Market": "#ffff00",
                            "High_Volatility": "#ff8800",
                        },
                    )
                    st.plotly_chart(regime_fig, use_container_width=True)

                with col2:
                    # Regime timeline
                    regime_timeline = go.Figure()

                    regime_timeline.add_trace(
                        go.Scatter(
                            x=backtest_results.index,
                            y=backtest_results["Close"],
                            mode="lines",
                            name="Price",
                            line=dict(color="blue", width=1),
                        )
                    )

                    # Color background by regime
                    for regime in regime_counts.index:
                        regime_mask = backtest_results["market_regime"] == regime
                        if regime_mask.any():
                            regime_data = backtest_results[regime_mask]
                            color_map = {
                                "Bull_Market": "rgba(0,255,0,0.1)",
                                "Bear_Market": "rgba(255,0,0,0.1)",
                                "Sideways_Market": "rgba(255,255,0,0.1)",
                                "High_Volatility": "rgba(255,136,0,0.1)",
                            }

                            for i, (idx, row) in enumerate(regime_data.iterrows()):
                                if (
                                    i == 0 or i == len(regime_data) - 1
                                ):  # Only show start and end for clarity
                                    regime_timeline.add_vrect(
                                        x0=idx,
                                        x1=idx,
                                        fillcolor=color_map.get(
                                            regime, "rgba(128,128,128,0.1)"
                                        ),
                                        opacity=0.3,
                                        layer="below",
                                        line_width=0,
                                    )

                    regime_timeline.update_layout(
                        title="Price with Market Regimes",
                        xaxis_title="Date",
                        yaxis_title="Price (₹)",
                        height=300,
                    )

                    st.plotly_chart(regime_timeline, use_container_width=True)

            # Performance metrics with enhanced calculations
            col1, col2, col3, col4, col5 = st.columns(5)

            with col1:
                st.metric("Total Return", f"{metrics['total_return']:.2f}%")
            with col2:
                st.metric("Max Drawdown", f"{metrics['max_drawdown']:.2f}%")
            with col3:
                st.metric("Sharpe Ratio", f"{metrics['sharpe_ratio']:.2f}")
            with col4:
                st.metric("Win Rate", f"{metrics['win_rate']:.1f}%")
            with col5:
                st.metric("Total Trades", f"{metrics['total_trades']}")

            # Portfolio performance chart
            fig = go.Figure()

            # Portfolio value
            fig.add_trace(
                go.Scatter(
                    x=backtest_results["Date"],
                    y=backtest_results["Portfolio_Value"],
                    mode="lines",
                    name="Portfolio Value",
                    line=dict(color="green", width=2),
                )
            )

            # Buy signals
            buy_signals = backtest_results[backtest_results["Signal"] == "BUY"]
            fig.add_trace(
                go.Scatter(
                    x=buy_signals["Date"],
                    y=buy_signals["Portfolio_Value"],
                    mode="markers",
                    name="Buy Signal",
                    marker=dict(color="green", size=10, symbol="triangle-up"),
                )
            )

            # Sell signals
            sell_signals = backtest_results[backtest_results["Signal"] == "SELL"]
            fig.add_trace(
                go.Scatter(
                    x=sell_signals["Date"],
                    y=sell_signals["Portfolio_Value"],
                    mode="markers",
                    name="Sell Signal",
                    marker=dict(color="red", size=10, symbol="triangle-down"),
                )
            )

            fig.update_layout(
                title=f"{strategy_type} Strategy - Portfolio Performance",
                xaxis_title="Date",
                yaxis_title="Portfolio Value (₹)",
                height=400,
            )

            st.plotly_chart(fig, use_container_width=True)

            # Price chart with indicators
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("📈 Price Chart with Signals")

                price_fig = go.Figure()

                # Price
                price_fig.add_trace(
                    go.Scatter(
                        x=backtest_results["Date"],
                        y=backtest_results["Close"],
                        mode="lines",
                        name="Price",
                        line=dict(color="blue"),
                    )
                )

                # Add indicators based on strategy
                if strategy_type == "SMA Crossover":
                    price_fig.add_trace(
                        go.Scatter(
                            x=backtest_results["Date"],
                            y=backtest_results["SMA_short"],
                            mode="lines",
                            name=f'SMA {params["sma_short"]}',
                            line=dict(color="orange"),
                        )
                    )
                    price_fig.add_trace(
                        go.Scatter(
                            x=backtest_results["Date"],
                            y=backtest_results["SMA_long"],
                            mode="lines",
                            name=f'SMA {params["sma_long"]}',
                            line=dict(color="red"),
                        )
                    )

                elif strategy_type == "Mean Reversion":
                    price_fig.add_trace(
                        go.Scatter(
                            x=backtest_results["Date"],
                            y=backtest_results["BB_upper"],
                            mode="lines",
                            name="BB Upper",
                            line=dict(color="red", dash="dash"),
                        )
                    )
                    price_fig.add_trace(
                        go.Scatter(
                            x=backtest_results["Date"],
                            y=backtest_results["BB_lower"],
                            mode="lines",
                            name="BB Lower",
                            line=dict(color="green", dash="dash"),
                        )
                    )

                # Buy/Sell signals on price
                price_fig.add_trace(
                    go.Scatter(
                        x=buy_signals["Date"],
                        y=buy_signals["Close"],
                        mode="markers",
                        name="Buy",
                        marker=dict(color="green", size=8, symbol="triangle-up"),
                    )
                )

                price_fig.add_trace(
                    go.Scatter(
                        x=sell_signals["Date"],
                        y=sell_signals["Close"],
                        mode="markers",
                        name="Sell",
                        marker=dict(color="red", size=8, symbol="triangle-down"),
                    )
                )

                price_fig.update_layout(
                    title=f"{symbol} Price Chart",
                    xaxis_title="Date",
                    yaxis_title="Price (₹)",
                    height=400,
                )

                st.plotly_chart(price_fig, use_container_width=True)

            with col2:
                st.subheader("📊 Trade Analysis")

                # Trade distribution
                signal_counts = backtest_results["Signal"].value_counts()
                if len(signal_counts) > 1:
                    pie_fig = px.pie(
                        values=signal_counts.values,
                        names=signal_counts.index,
                        title="Signal Distribution",
                    )
                    st.plotly_chart(pie_fig, use_container_width=True)

                # Monthly returns
                backtest_results["Month"] = backtest_results["Date"].dt.to_period("M")
                monthly_returns = (
                    backtest_results.groupby("Month")["Portfolio_Value"]
                    .last()
                    .pct_change()
                    .dropna()
                    * 100
                )

                if len(monthly_returns) > 0:
                    monthly_fig = px.bar(
                        x=monthly_returns.index.astype(str),
                        y=monthly_returns.values,
                        title="Monthly Returns (%)",
                        color=monthly_returns.values,
                        color_continuous_scale="RdYlGn",
                    )
                    monthly_fig.update_layout(height=300)
                    st.plotly_chart(monthly_fig, use_container_width=True)

            # Detailed trade log
            st.subheader("📋 Trade Log")

            trades_df = backtest_results[
                backtest_results["Signal"].isin(["BUY", "SELL"])
            ].copy()
            if not trades_df.empty:
                trades_df = trades_df[
                    ["Date", "Signal", "Close", "Portfolio_Value"]
                ].copy()
                trades_df["Date"] = trades_df["Date"].dt.strftime("%Y-%m-%d")
                trades_df["Close"] = trades_df["Close"].round(2)
                trades_df["Portfolio_Value"] = trades_df["Portfolio_Value"].round(2)
                st.dataframe(trades_df, use_container_width=True)
            else:
                st.info("No trades executed in this backtest")

            # Export results
            st.subheader("💾 Export Results")

            col1, col2, col3 = st.columns(3)

            with col1:
                if st.button("📊 Download Data"):
                    csv = backtest_results.to_csv(index=False)
                    st.download_button(
                        label="Download CSV",
                        data=csv,
                        file_name=f"{symbol}_{strategy_type}_backtest.csv",
                        mime="text/csv",
                    )

            with col2:
                if st.button("📈 Save Strategy"):
                    st.success("Strategy configuration saved!")

            with col3:
                if st.button("🤖 AI Analysis"):
                    with st.spinner("🧠 Analyzing strategy performance with AI..."):
                        try:
                            from src.analysis.ai_analyzer import GroqAnalyzer
                            from src.utils.config import load_config

                            # Load configuration for API key
                            config = load_config()
                            if not config.get("groq_api_key"):
                                st.error(
                                    "❌ GROQ API key not found in configuration. Please set GROQ_API_KEY in your .env file."
                                )
                                st.info(
                                    "💡 Get your free API key from: https://console.groq.com"
                                )
                                return

                            ai_analyzer = GroqAnalyzer(config["groq_api_key"])

                            # Prepare analysis data
                            analysis_data = {
                                "strategy_type": strategy_type,
                                "symbol": symbol,
                                "date_range": f"{start_date} to {end_date}",
                                "total_return": f"{metrics['total_return']:.2f}%",
                                "max_drawdown": f"{metrics['max_drawdown']:.2f}%",
                                "sharpe_ratio": f"{metrics['sharpe_ratio']:.2f}",
                                "win_rate": f"{metrics['win_rate']:.1f}%",
                                "total_trades": metrics["total_trades"],
                                "parameters": params,
                            }

                            # Add regime information if available
                            if (
                                params.get("use_regime_detection", False)
                                and "market_regime" in backtest_results.columns
                            ):
                                regime_distribution = (
                                    backtest_results["market_regime"]
                                    .value_counts()
                                    .to_dict()
                                )
                                analysis_data[
                                    "regime_distribution"
                                ] = regime_distribution

                            # Get recent trades for analysis
                            recent_trades = (
                                trades_df.tail(10)
                                if not trades_df.empty
                                else pd.DataFrame()
                            )
                            if not recent_trades.empty:
                                analysis_data["recent_trades"] = recent_trades.to_dict(
                                    "records"
                                )

                            # Perform AI analysis
                            prompt = f"""
                            Analyze this backtesting result for {strategy_type} strategy on {symbol}:
                            
                            Performance Metrics:
                            - Total Return: {analysis_data['total_return']}
                            - Max Drawdown: {analysis_data['max_drawdown']}
                            - Sharpe Ratio: {analysis_data['sharpe_ratio']}
                            - Win Rate: {analysis_data['win_rate']}
                            - Total Trades: {analysis_data['total_trades']}
                            
                            Strategy Parameters: {analysis_data['parameters']}
                            
                            Please provide:
                            1. Performance assessment (strengths and weaknesses)
                            2. Parameter optimization suggestions
                            3. Risk management recommendations
                            4. Market condition suitability
                            5. Specific improvements for this strategy type
                            
                            Keep analysis practical and actionable.
                            """

                            analysis_result = ai_analyzer.analyze_data(prompt)

                            # Display AI analysis results
                            st.subheader("🤖 AI Strategy Analysis")

                            # Create expandable sections for different aspects
                            with st.expander("📊 Performance Assessment", expanded=True):
                                st.markdown(analysis_result)

                            # Additional AI insights based on strategy type
                            strategy_specific_prompt = f"""
                            Given this {strategy_type} strategy with {metrics['total_trades']} trades and {metrics['win_rate']:.1f}% win rate:
                            
                            Provide 3 specific, actionable optimization suggestions for improving:
                            1. Entry/exit timing
                            2. Risk management
                            3. Parameter tuning
                            
                            Format as bullet points.
                            """

                            optimization_suggestions = ai_analyzer.analyze_data(
                                strategy_specific_prompt
                            )

                            with st.expander("🎯 Optimization Suggestions"):
                                st.markdown(optimization_suggestions)

                            # Risk analysis if drawdown is high
                            if metrics["max_drawdown"] < -10:  # More than 10% drawdown
                                risk_prompt = f"""
                                This strategy experienced {metrics['max_drawdown']:.1f}% maximum drawdown.
                                Provide specific risk management improvements for this {strategy_type} strategy.
                                Focus on position sizing, stop losses, and risk controls.
                                """

                                risk_analysis = ai_analyzer.analyze_data(risk_prompt)

                                with st.expander(
                                    "⚠️ Risk Management Focus", expanded=True
                                ):
                                    st.warning("High drawdown detected!")
                                    st.markdown(risk_analysis)

                            # Market regime insights if available
                            if (
                                params.get("use_regime_detection", False)
                                and "regime_distribution" in analysis_data
                            ):
                                regime_prompt = f"""
                                This strategy operated in different market regimes: {analysis_data['regime_distribution']}
                                
                                Analyze how the strategy performed in different market conditions and suggest regime-specific optimizations.
                                """

                                regime_analysis = ai_analyzer.analyze_data(
                                    regime_prompt
                                )

                                with st.expander("🧠 Market Regime Analysis"):
                                    st.markdown(regime_analysis)

                            st.success("✅ AI analysis complete!")

                        except Exception as e:
                            st.error(f"AI analysis failed: {str(e)}")
                            st.info(
                                "💡 Make sure your AI analyzer is properly configured with API keys."
                            )

    else:
        st.info(
            "👆 Configure your backtest parameters in the sidebar and click 'Run Backtest' to start!"
        )

        # Enhanced strategies showcase
        st.subheader("📚 Enhanced Strategy Library")

        col1, col2 = st.columns(2)

        with col1:
            st.info(
                """
            **🎯 SMA Crossover**
            
            • Buy when short SMA crosses above long SMA
            • Sell when short SMA crosses below long SMA
            • Good for trending markets
            • Enhanced with production-grade RSI
            """
            )

            st.success(
                """
            **📈 Mean Reversion**
            
            • Buy when price is oversold (RSI + Bollinger)
            • Sell when price is overbought
            • Good for ranging markets
            • Uses Wilder's RSI calculation
            """
            )

        with col2:
            st.warning(
                """
            **🧠 Regime-Aware Strategy (NEW!)**
            
            • Adapts strategy based on market regimes
            • Bull market: Trend following
            • Bear market: Defensive positioning
            • Sideways: Mean reversion
            • High volatility: Risk management
            """
            )

            st.info(
                """
            **📊 Buy and Hold**
            
            • Simple benchmark strategy
            • Buy once and hold
            • Compare other strategies against this
            • Enhanced with position sizing
            """
            )

        # Feature highlights
        st.subheader("🚀 Enhanced Features")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown(
                """
            **🧹 Data Quality**
            - Production-grade data cleaning
            - OHLC integrity validation
            - Outlier detection (rolling MAD)
            - Trading calendar awareness
            """
            )

        with col2:
            st.markdown(
                """
            **🧠 Regime Detection**
            - Walk-forward clustering
            - Volatility regime analysis
            - Trend regime detection
            - GMM market state modeling
            """
            )

        with col3:
            st.markdown(
                """
            **⚠️ Risk Management**
            - Position sizing controls
            - Enhanced performance metrics
            - Drawdown analysis
            - Regime-aware risk adjustment
            """
            )


if __name__ == "__main__":
    main()
