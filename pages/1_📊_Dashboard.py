import sys
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

# --- Path setup (works in Streamlit too) ---
ROOT = Path(__file__).resolve().parent
sys.path.append(str(ROOT.parent / "src"))

st.set_page_config(page_title="Dashboard", page_icon="📊", layout="wide")

# ---------- Data helpers ----------

def _to_float(x, default=0.0):
    try:
        return float(str(x).replace(",", ""))
    except Exception:
        return default

def get_sample_market_data():
    return {
        "NIFTY 50": {"value": 25147.50, "change": 0.75, "volume": "1.2B"},
        "NIFTY BANK": {"value": 52234.80, "change": -0.45, "volume": "800M"},
    }, None

@st.cache_data(show_spinner=False, ttl=30)
def get_real_market_overview():
    """Try live via jugaad-data; otherwise fallback to sample."""
    try:
        from src.data.jugaad_client import JugaadDataClient
        client = JugaadDataClient()
        if not client.nse_live:
            return get_sample_market_data()

        major_stocks = ["RELIANCE", "TCS", "HDFCBANK", "INFY"]
        live_prices = client.get_multiple_live_prices(major_stocks) or {}

        if not live_prices:
            return get_sample_market_data()

        # avg pct change across a few bellwethers
        changes = [_to_float(v.get("pChange", 0)) for v in live_prices.values()]
        avg_change = np.mean(changes) if changes else 0.0

        market_data = {
            "NIFTY 50": {
                # This is a heuristic display proxy; for real index values, query index_df.
                "value": 25000 + (avg_change * 100),
                "change": avg_change,
                "volume": "—",
            },
            "NIFTY BANK": {
                "value": 52000 + (avg_change * 200),
                "change": avg_change * 1.2,
                "volume": "—",
            },
        }
        return market_data, live_prices
    except Exception:
        return get_sample_market_data()

def generate_portfolio_df(seed: int = 42):
    """Toy equity curve (replace with your backtest equity)."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start="2024-01-01", end=datetime.now(), freq="D")
    # simulate drift + noise
    rets = rng.normal(loc=0.0006, scale=0.01, size=len(dates))
    equity = 100000 * (1 + pd.Series(rets, index=dates)).cumprod()
    return pd.DataFrame({"Date": dates, "Portfolio Value": equity.values})

# ---------- UI ----------

def main():
    st.title("📊 Trading Dashboard")

    # Metrics row (hook these to real stats later)
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric("Portfolio Value", "₹1,00,000", "₹2,450 (2.45%)")
    with m2:
        st.metric("Today's P&L", "₹2,450", "2.45%")
    with m3:
        st.metric("Total Trades", "127", "5 today")
    with m4:
        st.metric("Win Rate", "68.5%", "2.3%")

    # Performance + Strategies
    c1, c2 = st.columns([2, 1])
    with c1:
        st.subheader("📈 Portfolio Performance")
        pf = generate_portfolio_df()
        fig = px.line(pf, x="Date", y="Portfolio Value", title=None)
        fig.update_layout(height=420, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        st.subheader("🎯 Active Strategies")
        strategies = [
            {"name": "Momentum Strategy", "status": "🟢 Active", "returns": "+5.2%"},
            {"name": "Mean Reversion", "status": "🟢 Active", "returns": "+3.1%"},
            {"name": "Regime Switch", "status": "🟡 Paused", "returns": "-1.2%"},
            {"name": "News Sentiment", "status": "🟢 Active", "returns": "+2.8%"},
        ]
        for s in strategies:
            st.markdown(
                f"**{s['name']}**  \nStatus: {s['status']}  \nReturns: {s['returns']}"
            )
            st.divider()

    # Market overview
    st.subheader("🌍 Market Overview")
    market_data, live_prices = get_real_market_overview()

    k1, k2, k3 = st.columns(3)
    with k1:
        d = market_data.get("NIFTY 50", {})
        st.metric("NIFTY 50", f"{d.get('value','—'):.2f}" if isinstance(d.get("value"), (int,float)) else d.get("value","—"),
                  f"{d.get('change','—'):+.2f}%")
        st.caption(f"Volume: {d.get('volume','—')}")
    with k2:
        d = market_data.get("NIFTY BANK", {})
        st.metric("NIFTY BANK", f"{d.get('value','—'):.2f}" if isinstance(d.get("value"), (int,float)) else d.get("value","—"),
                  f"{d.get('change','—'):+.2f}%")
        st.caption(f"Volume: {d.get('volume','—')}")
    with k3:
        if live_prices:
            avg_change = np.mean([_to_float(v.get("pChange", 0)) for v in live_prices.values()])
            st.metric("Market Sentiment (proxy)", "—", f"{avg_change:+.2f}% avg across bellwethers")
        else:
            st.info("Using sample data")

    # Live bellwethers (if available)
    if live_prices:
        st.markdown("**Live Bellwethers**")
        rows = []
        for sym, v in live_prices.items():
            rows.append({
                "Symbol": sym,
                "Price": _to_float(v.get("price", 0)),
                "% Chg": _to_float(v.get("pChange", 0)),
                "Open": _to_float(v.get("open", 0)),
                "High": _to_float(v.get("high", 0)),
                "Low": _to_float(v.get("low", 0)),
                "Volume": int(_to_float(v.get("volume", 0))),
                "Time": pd.to_datetime(v.get("timestamp")).strftime("%H:%M:%S") if v.get("timestamp") else "—",
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    # Recent trades (placeholder table)
    st.subheader("📋 Recent Trades")
    trades_df = pd.DataFrame([
        {"Time": "15:25:30", "Symbol": "RELIANCE", "Action": "BUY",  "Qty": 10, "Price": 2456.75, "P&L":  245.60, "Strategy": "Momentum"},
        {"Time": "14:55:15", "Symbol": "TCS",      "Action": "SELL", "Qty":  5, "Price": 3234.50, "P&L":  167.25, "Strategy": "Mean Reversion"},
        {"Time": "14:30:45", "Symbol": "INFY",     "Action": "BUY",  "Qty": 15, "Price": 1567.25, "P&L":  312.15, "Strategy": "AI Signals"},
        {"Time": "13:45:20", "Symbol": "HDFC",     "Action": "SELL", "Qty":  8, "Price": 1654.80, "P&L":  -89.40, "Strategy": "Regime Switch"},
        {"Time": "12:15:10", "Symbol": "ICICIBANK","Action": "BUY",  "Qty": 12, "Price": 1089.30, "P&L":  198.75, "Strategy": "News Sentiment"},
    ])
    st.dataframe(trades_df, use_container_width=True, hide_index=True)

if __name__ == "__main__":
    main()
