import sys
import time
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
    # DEPRECATED: This function was causing JSON parsing errors
    # Using fetch_live_data_simplified instead
    return None
    # returns normalized dict or None
    # try:
    #     from src.data.jugaad_client import JugaadDataClient  # keep import consistent

    #     client = JugaadDataClient()
    #     live = client.get_multiple_live_prices(symbols) or {}
    #     if not live:
    #         return None
    #     clean = {
    #         sym: _normalize_live_dict({**v, "symbol": sym}) for sym, v in live.items()
    #     }
    #     return clean
    # except Exception:
    #     return None


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


def get_replay_data(symbols, replay_date, current_time_index=0):
    """
    Replay historical data from a specific date using real jugaad-data
    
    Args:
        symbols: List of stock symbols
        replay_date: Date to replay (datetime.date)
        current_time_index: Current index in the replay (0 = market open)
    
    Returns:
        Dict with live-like data for the replay timestamp
    """
    try:
        # Import the correct jugaad-data function
        from jugaad_data.nse import stock_df
        from datetime import date
        
        replay_data = {}
        
        # Convert datetime.date to date object if needed
        if isinstance(replay_date, datetime):
            replay_date = replay_date.date()
        
        print(f"🔄 Fetching historical data for {len(symbols)} symbols on {replay_date}")
        
        for symbol in symbols:
            try:
                # Use the correct jugaad-data API
                # Get historical data for the specific date (single day)
                df = stock_df(
                    symbol=symbol, 
                    from_date=replay_date,
                    to_date=replay_date,
                    series="EQ"
                )
                
                if df is not None and not df.empty:
                    # jugaad-data returns columns: DATE, SERIES, OPEN, HIGH, LOW, PREV. CLOSE, LTP, CLOSE, VWAP, etc.
                    row = df.iloc[0]  # Get the single day's data
                    
                    # Extract the actual price values
                    open_price = float(row['OPEN'])
                    high_price = float(row['HIGH']) 
                    low_price = float(row['LOW'])
                    close_price = float(row['CLOSE'])
                    prev_close = float(row['PREV. CLOSE'])
                    volume = int(row.get('VOLUME', 0))
                    
                    # Simulate intraday progression based on time_index (0-100)
                    progress = current_time_index / 100.0
                    
                    # Simulate realistic intraday price movement
                    if progress == 0:
                        # Market open
                        current_price = open_price
                    elif progress >= 1.0:
                        # Market close
                        current_price = close_price
                    else:
                        # Interpolate between open and close with some randomness
                        base_progression = open_price + (close_price - open_price) * progress
                        
                        # Add some intraday volatility (within high-low range)
                        volatility_range = high_price - low_price
                        random_factor = (np.random.random() - 0.5) * 0.3  # ±15% of daily range
                        volatility_adjustment = volatility_range * random_factor
                        
                        current_price = base_progression + volatility_adjustment
                        # Ensure price stays within high-low bounds
                        current_price = max(low_price, min(high_price, current_price))
                    
                    # Calculate change from previous close
                    change = current_price - prev_close
                    pchange = (change / prev_close * 100) if prev_close != 0 else 0
                    
                    # Simulate realistic volume progression
                    simulated_volume = int(volume * (progress + 0.1))  # Volume builds throughout day
                    
                    replay_data[symbol] = {
                        "symbol": symbol,
                        "price": float(current_price),
                        "change": float(change),
                        "pChange": float(pchange),
                        "open": float(open_price),
                        "high": float(high_price),
                        "low": float(low_price),
                        "volume": simulated_volume,
                        "timestamp": datetime.combine(replay_date, datetime.min.time()),
                        "prev_close": float(prev_close),
                        "close": float(close_price),  # End-of-day close price
                    }
                    
                    print(f"✅ {symbol}: ₹{current_price:.2f} (Open: ₹{open_price:.2f}, Close: ₹{close_price:.2f})")
                else:
                    print(f"⚠️ {symbol}: No data available for {replay_date}")
                
            except Exception as e:
                print(f"❌ {symbol}: {str(e)[:100]}...")  # Truncate long error messages
                continue
        
        if replay_data:
            print(f"✅ Successfully loaded replay data for {len(replay_data)} symbols")
            return replay_data
        else:
            print("❌ No replay data could be loaded")
            return None
        
    except ImportError as e:
        print(f"❌ jugaad-data not available: {e}")
        return None
    except Exception as e:
        print(f"❌ Error in replay system: {e}")
        return None


def fetch_live_data_simplified(symbols):
    """
    Simplified live data fetching with better error handling
    
    Args:
        symbols: List of stock symbols
    
    Returns:
        Dict with live data or None if failed
    """
    try:
        # Try yfinance first (more reliable for live data)
        import yfinance as yf
        live_data = {}
        
        for symbol in symbols:
            try:
                # Get live data from yfinance
                ticker = yf.Ticker(f"{symbol}.NS")
                info = ticker.info
                
                if info and 'regularMarketPrice' in info:
                    current_price = info.get('regularMarketPrice', 0)
                    prev_close = info.get('previousClose', current_price)
                    open_price = info.get('regularMarketOpen', current_price)
                    high_price = info.get('dayHigh', current_price)
                    low_price = info.get('dayLow', current_price)
                    volume = info.get('regularMarketVolume', 0)
                    
                    change = current_price - prev_close
                    p_change = (change / prev_close * 100) if prev_close else 0
                    
                    live_data[symbol] = {
                        "symbol": symbol,
                        "price": float(current_price),
                        "change": float(change),
                        "pChange": float(p_change),
                        "open": float(open_price),
                        "high": float(high_price),
                        "low": float(low_price),
                        "volume": int(volume),
                        "timestamp": datetime.now(),
                    }
                    print(f"✅ Live data: {symbol} = ₹{current_price:.2f}")
                
            except Exception as e:
                print(f"⚠️ {symbol} live data failed: {str(e)[:50]}...")
                continue
        
        return live_data if live_data else None
        
    except Exception as e:
        print(f"❌ Live data system error: {e}")
        return None


def generate_simulated_data(symbols):
    """Generate simulated live data for demo using recent real prices as base"""
    data = {}
    
    # Try to get recent real prices first
    try:
        import yfinance as yf
        base_prices = {}
        
        for symbol in symbols:
            try:
                ticker = yf.Ticker(f"{symbol}.NS")
                hist = ticker.history(period="5d")
                if not hist.empty:
                    base_prices[symbol] = float(hist['Close'].iloc[-1])
            except:
                pass
    except:
        pass
    
    # Fallback to hardcoded recent prices if yfinance fails
    fallback_prices = {
        "RELIANCE": 2456.75,
        "TCS": 3234.50,
        "INFY": 1567.25,
        "HDFCBANK": 1654.80,
        "ICICIBANK": 1089.30,
    }
    
    # Use real prices if available, fallback otherwise
    if not base_prices:
        base_prices = fallback_prices

    for symbol in symbols:
        if symbol in base_prices:
            base_price = base_prices[symbol]
            # More realistic intraday volatility (0.5% to 2%)
            volatility = np.random.uniform(0.005, 0.02)
            change = np.random.normal(0, volatility) * base_price
            
            data[symbol] = {
                "symbol": symbol,
                "price": base_price + change,
                "change": change,
                "pChange": (change / base_price) * 100,
                "open": base_price * np.random.uniform(0.995, 1.005),  # Realistic gap
                "high": base_price + abs(change) * np.random.uniform(1.2, 2.0),
                "low": base_price - abs(change) * np.random.uniform(1.2, 2.0),
                "volume": np.random.randint(100000, 2000000),  # More realistic volume
                "timestamp": datetime.now(),
            }
        else:
            # For unknown symbols, generate basic simulated data
            base_price = np.random.uniform(500, 3000)
            change = np.random.normal(0, 0.01) * base_price
            data[symbol] = {
                "symbol": symbol,
                "price": base_price,
                "change": change,
                "pChange": (change / base_price) * 100,
                "open": base_price,
                "high": base_price + abs(change) * 1.5,
                "low": base_price - abs(change) * 1.5,
                "volume": np.random.randint(50000, 500000),
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

    # Initialize replay state if not exists
    if "replay_mode" not in st.session_state:
        st.session_state.replay_mode = False
        st.session_state.replay_date = None
        st.session_state.replay_time_index = 0
        st.session_state.replay_auto_play = False

    # Get symbols to watch - use canonicalized symbols
    symbols = canonicalize_symbols(["RELIANCE", "TCS", "INFY", "HDFC", "ICICI"])

    # Try to get live data first (simplified approach)
    live_data = fetch_live_data_simplified(symbols)
    is_real_data = live_data is not None
    data_source = "Real Live Data"
    
    # If no live data, show replay/simulation options
    if live_data is None:
        st.sidebar.markdown("## 📊 Data Source Options")
        
        # Market status check
        try:
            from src.data.jugaad_client import JugaadDataClient
            client = JugaadDataClient()
            market_status = client.get_market_status()
            if market_status.get("status") == "closed":
                st.sidebar.info(f"🕒 {market_status.get('message', 'Market is closed')}")
        except:
            pass
        
        # Data source selection
        data_mode = st.sidebar.radio(
            "Choose Data Source:",
            ["📺 Historical Replay", "🎲 Simulated Data"],
            index=0 if st.session_state.replay_mode else 1
        )
        
        if data_mode == "📺 Historical Replay":
            st.session_state.replay_mode = True
            
            # Date selection for replay
            st.sidebar.markdown("### 📅 Replay Configuration")
            
            # Default to last trading day (not weekend)
            default_date = datetime.now().date()
            while default_date.weekday() >= 5:  # Weekend
                default_date -= timedelta(days=1)
            default_date -= timedelta(days=1)  # Previous trading day
            
            replay_date = st.sidebar.date_input(
                "Select Date to Replay:",
                value=default_date,
                max_value=datetime.now().date() - timedelta(days=1),
                min_value=datetime.now().date() - timedelta(days=365),
                help="Choose a past trading day to replay real market data"
            )
            
            # Time progression controls
            col1, col2 = st.sidebar.columns(2)
            with col1:
                if st.button("⏮️ Reset"):
                    st.session_state.replay_time_index = 0
                    st.rerun()
            
            with col2:
                auto_play = st.checkbox("▶️ Auto Play", value=st.session_state.replay_auto_play)
                st.session_state.replay_auto_play = auto_play
            
            # Manual time control
            if not auto_play:
                time_step = st.sidebar.slider(
                    "Time Progress:",
                    min_value=0,
                    max_value=100,
                    value=st.session_state.replay_time_index,
                    help="Simulate time progression through the trading day"
                )
                st.session_state.replay_time_index = time_step
            else:
                # Auto-increment time for auto-play
                st.session_state.replay_time_index = (st.session_state.replay_time_index + 1) % 100
                time.sleep(2)  # Auto-play speed
                st.rerun()
            
            # Get replay data
            live_data = get_replay_data(symbols, replay_date, st.session_state.replay_time_index)
            
            if live_data:
                data_source = f"Historical Replay - {replay_date.strftime('%d %b %Y')}"
                st.sidebar.success(f"✅ Replaying {replay_date.strftime('%d %b %Y')}")
                
                # Show replay progress
                progress = st.session_state.replay_time_index / 100
                st.sidebar.progress(progress, text=f"Trading Day Progress: {progress:.0%}")
                
            else:
                st.sidebar.error("❌ No historical data available for selected date")
                # Fallback to simulated data
                live_data = generate_simulated_data(symbols)
                data_source = "Simulated Data (Replay Failed)"
        
        else:
            # Simulated data mode
            st.session_state.replay_mode = False
            live_data = generate_simulated_data(symbols)
            data_source = "Simulated Data"
            
            st.sidebar.warning("⚠️ Using synthetic data with realistic volatility")
    
    # If still no data, use fallback
    if live_data is None:
        live_data = generate_simulated_data(symbols)
        data_source = "Simulated Data (Fallback)"

    # Execute any pending LIMIT orders
    process_limit_orders(live_data)

    # Data source indicator
    if is_real_data:
        data_source_color = "🟢"
    elif "Replay" in data_source:
        data_source_color = "🔵" 
    else:
        data_source_color = "🟡"
    
    st.sidebar.markdown(f"{data_source_color} **Data Source**: {data_source}")
    
    if is_real_data:
        st.sidebar.success("✅ Connected to live market feed")
    elif "Replay" in data_source:
        st.sidebar.info("🔵 Using historical market replay")
    else:
        st.sidebar.warning("⚠️ Using simulated market data")

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
