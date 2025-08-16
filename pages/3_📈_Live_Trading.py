import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

try:
    from src.data.jugaad_client import JugaadDataClient
    from src.utils.config import load_config
except ImportError:
    st.error(
        "Required modules not found. Please ensure all dependencies are installed."
    )

st.set_page_config(page_title="Live Trading", page_icon="📈", layout="wide")

# ==================== HELPERS ====================

SYMBOL_MAP = {
    "ICICI": "ICICIBANK",
    "HDFC": "HDFCBANK",  # prefer HDFCBANK; drop HDFC if you wish
}
DEFAULT_SYMBOLS = ["RELIANCE", "TCS", "INFY", "HDFCBANK", "ICICIBANK"]


def canonicalize_symbols(symbols):
    out = []
    for s in symbols:
        out.append(SYMBOL_MAP.get(s, s))
    # dedupe while preserving order
    dedup = []
    for s in out:
        if s not in dedup:
            dedup.append(s)
    return dedup


def _to_float(x, default=0.0):
    try:
        return float(str(x).replace(",", ""))
    except Exception:
        return default


def _normalize_live_dict(d):
    # normalize Jugaad/NSE fields to clean floats
    return {
        "symbol": d.get("symbol"),
        "price": _to_float(d.get("price", d.get("lastPrice", 0))),
        "change": _to_float(d.get("change", 0)),
        "pChange": _to_float(d.get("pChange", 0)),
        "open": _to_float(d.get("open", 0)),
        "high": _to_float(d.get("dayHigh", d.get("high", 0))),
        "low": _to_float(d.get("dayLow", d.get("low", 0))),
        "volume": int(_to_float(d.get("totalTradedVolume", d.get("volume", 0)))),
        "timestamp": d.get("timestamp", datetime.now()),
    }


@st.cache_data(ttl=10, show_spinner=False)
def fetch_live_data(symbols):
    # returns normalized dict or None
    try:
        from src.data.jugaad_client import JugaadDataClient  # keep import consistent

        client = JugaadDataClient()
        live = client.get_multiple_live_prices(symbols) or {}
        if not live:
            return None
        clean = {
            sym: _normalize_live_dict({**v, "symbol": sym}) for sym, v in live.items()
        }
        return clean
    except Exception:
        return None


def process_limit_orders(live_data):
    """Execute any PENDING LIMIT orders that have reached price."""
    if not live_data:
        return
    for order in st.session_state.portfolio["orders"]:
        if order["status"] != "PENDING" or order.get("type") not in ("BUY", "SELL"):
            continue
        if order.get("price") is None:  # only LIMIT here
            continue
        sym = order["symbol"]
        cur = live_data.get(sym, {}).get("price")
        if not cur:
            continue
        # simple touch logic
        if order["type"] == "BUY" and cur <= order["price"]:
            execute_order(order, cur)
        elif order["type"] == "SELL" and cur >= order["price"]:
            execute_order(order, cur)


def initialize_session_state():
    """Initialize session state variables"""
    if "portfolio" not in st.session_state:
        st.session_state.portfolio = {
            "cash": 100000,  # Starting with 1 lakh
            "positions": {},
            "orders": [],
            "trade_history": [],
        }

    if "data_client" not in st.session_state:
        st.session_state.data_client = JugaadDataClient()


def generate_simulated_data(symbols):
    """Generate simulated live data for demo"""
    data = {}
    base_prices = {
        "RELIANCE": 2456.75,
        "TCS": 3234.50,
        "INFY": 1567.25,
        "HDFCBANK": 1654.80,
        "ICICIBANK": 1089.30,
    }

    for symbol in symbols:
        if symbol in base_prices:
            base_price = base_prices[symbol]
            change = np.random.normal(0, 0.01) * base_price
            data[symbol] = {
                "symbol": symbol,
                "price": base_price + change,
                "change": change,
                "pChange": (change / base_price) * 100,
                "open": base_price,
                "high": base_price + abs(change) * 1.5,
                "low": base_price - abs(change) * 1.5,
                "volume": np.random.randint(100000, 1000000),
                "timestamp": datetime.now(),
            }
    return data


def place_order(symbol, quantity, order_type, price=None):
    """Place a virtual order"""
    order = {
        "id": len(st.session_state.portfolio["orders"]) + 1,
        "symbol": symbol,
        "quantity": quantity,
        "type": order_type,
        "price": price,
        "timestamp": datetime.now(),
        "status": "PENDING",
    }
    st.session_state.portfolio["orders"].append(order)
    return order


def execute_order(order, market_price):
    """Execute a virtual order"""
    symbol = order["symbol"]
    quantity = order["quantity"]
    order_type = order["type"]

    if order_type == "BUY":
        total_cost = quantity * market_price
        if st.session_state.portfolio["cash"] >= total_cost:
            st.session_state.portfolio["cash"] -= total_cost
            if symbol in st.session_state.portfolio["positions"]:
                st.session_state.portfolio["positions"][symbol]["quantity"] += quantity
                # Update average price
                current_qty = (
                    st.session_state.portfolio["positions"][symbol]["quantity"]
                    - quantity
                )
                current_avg = st.session_state.portfolio["positions"][symbol][
                    "avg_price"
                ]
                new_avg = ((current_qty * current_avg) + (quantity * market_price)) / (
                    current_qty + quantity
                )
                st.session_state.portfolio["positions"][symbol]["avg_price"] = new_avg
            else:
                st.session_state.portfolio["positions"][symbol] = {
                    "quantity": quantity,
                    "avg_price": market_price,
                }

            # Add to trade history
            trade = {
                "timestamp": datetime.now(),
                "symbol": symbol,
                "type": "BUY",
                "quantity": quantity,
                "price": market_price,
                "total": total_cost,
            }
            st.session_state.portfolio["trade_history"].append(trade)
            order["status"] = "EXECUTED"
            return True
        else:
            order["status"] = "REJECTED - Insufficient funds"
            return False

    elif order_type == "SELL":
        if (
            symbol in st.session_state.portfolio["positions"]
            and st.session_state.portfolio["positions"][symbol]["quantity"] >= quantity
        ):
            st.session_state.portfolio["cash"] += quantity * market_price
            st.session_state.portfolio["positions"][symbol]["quantity"] -= quantity

            if st.session_state.portfolio["positions"][symbol]["quantity"] == 0:
                del st.session_state.portfolio["positions"][symbol]

            # Add to trade history
            trade = {
                "timestamp": datetime.now(),
                "symbol": symbol,
                "type": "SELL",
                "quantity": quantity,
                "price": market_price,
                "total": quantity * market_price,
            }
            st.session_state.portfolio["trade_history"].append(trade)
            order["status"] = "EXECUTED"
            return True
        else:
            order["status"] = "REJECTED - Insufficient shares"
            return False


def main():
    st.title("📈 Live Trading")
    st.markdown("Trade with virtual money using real-time market data")

    # Initialize session state
    initialize_session_state()

    # Get symbols to watch - use canonicalized symbols
    symbols = canonicalize_symbols(["RELIANCE", "TCS", "INFY", "HDFC", "ICICI"])

    # Get live data with fallback
    live_data = fetch_live_data(symbols)
    if live_data is None:
        st.warning(
            "⚠️ Live feed unavailable (market closed or connection). Using simulated ticks."
        )
        live_data = generate_simulated_data(symbols)

    # Execute any pending LIMIT orders
    process_limit_orders(live_data)

    # Header with portfolio summary
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Cash Available", f"₹{st.session_state.portfolio['cash']:,.2f}")

    # Calculate portfolio value (guarded)
    portfolio_value = st.session_state.portfolio["cash"]
    for symbol, position in st.session_state.portfolio["positions"].items():
        if live_data and symbol in live_data:
            portfolio_value += position["quantity"] * live_data[symbol]["price"]

    with col2:
        st.metric("Portfolio Value", f"₹{portfolio_value:,.2f}")

    with col3:
        total_pnl = portfolio_value - 100000
        st.metric("Total P&L", f"₹{total_pnl:,.2f}", f"{(total_pnl/100000)*100:.2f}%")

    with col4:
        active_positions = len(st.session_state.portfolio["positions"])
        st.metric("Active Positions", active_positions)

    st.markdown("---")

    # Main content
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("📊 Market Watch")

        # Auto-refresh button
        if st.button("🔄 Refresh Data"):
            st.rerun()

        # Display live market data with proper formatting
        if live_data:
            rows = []
            for symbol in symbols:
                d = live_data.get(symbol)
                if not d:
                    continue
                rows.append(
                    {
                        "Symbol": symbol,
                        "Price": d["price"],
                        "Change": d["change"],
                        "Change %": d["pChange"],
                        "High": d["high"],
                        "Low": d["low"],
                        "Volume": d["volume"],
                    }
                )
            df = pd.DataFrame(rows)
            st.dataframe(
                df,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Price": st.column_config.NumberColumn(format="₹%.2f"),
                    "Change": st.column_config.NumberColumn(format="%.2f"),
                    "Change %": st.column_config.NumberColumn(format="%.2f%%"),
                    "High": st.column_config.NumberColumn(format="₹%.2f"),
                    "Low": st.column_config.NumberColumn(format="₹%.2f"),
                    "Volume": st.column_config.NumberColumn(format="%,d"),
                },
            )

        # Order placement
        st.subheader("📝 Place Order")

        order_col1, order_col2, order_col3 = st.columns(3)

        with order_col1:
            selected_symbol = st.selectbox("Select Symbol", symbols)
            order_type = st.selectbox("Order Type", ["BUY", "SELL"])

        with order_col2:
            quantity = st.number_input("Quantity", min_value=1, value=1)
            order_mode = st.selectbox("Order Mode", ["MARKET", "LIMIT"])

        with order_col3:
            if order_mode == "LIMIT":
                limit_price = st.number_input(
                    "Limit Price", min_value=0.01, value=100.0
                )
            else:
                limit_price = None

            if st.button("Place Order", type="primary"):
                if live_data and selected_symbol in live_data:
                    market_price = live_data[selected_symbol]["price"]
                    order = place_order(
                        selected_symbol, quantity, order_type, limit_price
                    )

                    if order_mode == "MARKET":
                        success = execute_order(order, market_price)
                        if success:
                            st.success(f"Order executed successfully!")
                        else:
                            st.error(f"Order failed: {order['status']}")
                    else:
                        st.info(
                            "Limit order placed. Will execute when price reaches target."
                        )

                    st.rerun()

        # Position details with proper formatting
        st.subheader("📋 Current Positions")

        if st.session_state.portfolio["positions"]:
            position_rows = []
            for symbol, pos in st.session_state.portfolio["positions"].items():
                d = live_data.get(symbol) if live_data else None
                if not d:
                    continue
                cur = d["price"]
                value = pos["quantity"] * cur
                cost = pos["quantity"] * pos["avg_price"]
                pnl = value - cost
                pnl_pct = (pnl / cost * 100) if cost > 0 else 0.0
                position_rows.append(
                    {
                        "Symbol": symbol,
                        "Quantity": pos["quantity"],
                        "Avg Price": pos["avg_price"],
                        "Current Price": cur,
                        "Value": value,
                        "P&L": pnl,
                        "P&L %": pnl_pct,
                    }
                )
            if position_rows:
                pdf = pd.DataFrame(position_rows)
                st.dataframe(
                    pdf,
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "Avg Price": st.column_config.NumberColumn(format="₹%.2f"),
                        "Current Price": st.column_config.NumberColumn(format="₹%.2f"),
                        "Value": st.column_config.NumberColumn(format="₹%,.2f"),
                        "P&L": st.column_config.NumberColumn(format="₹%,.2f"),
                        "P&L %": st.column_config.NumberColumn(format="%.2f%%"),
                    },
                )
        else:
            st.info("No open positions")

    with col2:
        st.subheader("📈 Quick Actions")

        # Quick buy/sell buttons for popular stocks
        st.write("**One-Click Trading:**")

        for symbol in symbols[:3]:
            if live_data and symbol in live_data:
                price = live_data[symbol]["price"]
                col_buy, col_sell = st.columns(2)

                with col_buy:
                    if st.button(f"Buy {symbol}", key=f"buy_{symbol}"):
                        order = place_order(symbol, 1, "BUY")
                        execute_order(order, price)
                        st.rerun()

                with col_sell:
                    if st.button(f"Sell {symbol}", key=f"sell_{symbol}"):
                        if symbol in st.session_state.portfolio["positions"]:
                            order = place_order(symbol, 1, "SELL")
                            execute_order(order, price)
                            st.rerun()

        st.markdown("---")

        # Recent orders
        st.subheader("📋 Recent Orders")

        if st.session_state.portfolio["orders"]:
            recent_orders = st.session_state.portfolio["orders"][-5:]  # Last 5 orders
            for order in reversed(recent_orders):
                with st.container():
                    st.write(f"**{order['type']} {order['symbol']}**")
                    st.write(f"Qty: {order['quantity']} | Status: {order['status']}")
                    st.write(f"Time: {order['timestamp'].strftime('%H:%M:%S')}")
                    st.markdown("---")
        else:
            st.info("No orders placed yet")

        # Trade history
        st.subheader("📊 Trade History")

        if st.session_state.portfolio["trade_history"]:
            recent_trades = st.session_state.portfolio["trade_history"][-5:]
            for trade in reversed(recent_trades):
                color = "green" if trade["type"] == "BUY" else "red"
                st.markdown(
                    f"""
                <div style="padding: 5px; border-left: 3px solid {color};">
                <strong>{trade['type']} {trade['symbol']}</strong><br>
                Qty: {trade['quantity']} @ ₹{trade['price']:.2f}<br>
                Total: ₹{trade['total']:.2f}<br>
                Time: {trade['timestamp'].strftime('%H:%M:%S')}
                </div>
                """,
                    unsafe_allow_html=True,
                )
                st.markdown("---")
        else:
            st.info("No trades executed yet")

        # Reset portfolio button
        if st.button("🔄 Reset Portfolio", help="Reset to initial ₹1,00,000"):
            st.session_state.portfolio = {
                "cash": 100000,
                "positions": {},
                "orders": [],
                "trade_history": [],
            }
            st.success("Portfolio reset successfully!")
            st.rerun()

    # Optional auto-refresh (every 10 seconds)
    st.rerun()


if __name__ == "__main__":
    main()
