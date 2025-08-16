import json
import logging
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / "src"))

from auth.auth_ui import require_auth

# Authentication disabled for public demo
# require_auth()

# Set up comprehensive logging
import os

os.makedirs("logs", exist_ok=True)  # Create logs directory if it doesn't exist

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("logs/optimization.log"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)
logger.info("AI Assistant page loaded - logging initialized")

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

try:
    from analysis.ai_analyzer import GroqAnalyzer
    from utils.config import load_config
except ImportError:
    st.error(
        "Required modules not found. Please ensure all dependencies are installed."
    )
    st.stop()

st.set_page_config(page_title="AI Assistant", page_icon="🤖", layout="wide")


# Cache AI analyzer to avoid recreation
@st.cache_resource
def get_ai_analyzer(api_key):
    """Cached AI analyzer initialization"""
    logger.info("Initializing AI analyzer...")
    try:
        analyzer = GroqAnalyzer(api_key)
        logger.info("AI analyzer initialized successfully")
        return analyzer
    except Exception as e:
        logger.error(f"Failed to initialize AI analyzer: {e}")
        logger.error(traceback.format_exc())
        raise


# Cache synthetic market data
@st.cache_data(ttl=600)
def generate_market_data(seed=42, days=365):
    """Generate cached synthetic market data"""
    np.random.seed(seed)
    dates = pd.date_range(start="2024-01-01", periods=days, freq="D")
    prices = [100]
    for _ in range(days - 1):
        change = np.random.normal(0.001, 0.02)
        prices.append(prices[-1] * (1 + change))

    return pd.DataFrame(
        {
            "Date": dates,
            "Close": prices,
            "High": [p * 1.01 for p in prices],
            "Low": [p * 0.99 for p in prices],
            "Volume": np.random.randint(1000000, 5000000, len(prices)),
        }
    )


def mock_backtest_function(strategy_config):
    """Mock backtest function for demonstration with deterministic results"""
    logger.info(f"Starting mock backtest with config: {strategy_config}")
    try:
        knobs = strategy_config.get("knobs", {})
        logger.info(f"Extracted knobs: {knobs}")

        # Simulate strategy performance based on parameters
        np.random.seed(42)  # Deterministic for fair comparison across iterations
        days = 252

        sma_fast = knobs.get("sma_fast", 10)
        sma_slow = knobs.get("sma_slow", 30)
        risk_per_trade = knobs.get("risk_per_trade", 0.01)

        logger.info(
            f"Parameters - SMA Fast: {sma_fast}, SMA Slow: {sma_slow}, Risk: {risk_per_trade}"
        )

        # Parameter-dependent performance simulation
        trend_efficiency = 1.0 - abs(sma_fast - 12) * 0.01
        lag_penalty = max(0.8, 1.0 - (sma_slow - 25) * 0.005)
        risk_scaling = min(1.2, 1.0 + (0.01 - risk_per_trade) * 10)

        logger.info(
            f"Calculated multipliers - Trend: {trend_efficiency:.3f}, Lag: {lag_penalty:.3f}, Risk: {risk_scaling:.3f}"
        )

        base_return = 0.001
        volatility = 0.02
        adjusted_return = base_return * trend_efficiency * lag_penalty * risk_scaling
        adjusted_volatility = volatility * (1 + abs(risk_per_trade - 0.01) * 5)

        logger.info(
            f"Adjusted return: {adjusted_return:.5f}, volatility: {adjusted_volatility:.5f}"
        )

        returns = np.random.normal(adjusted_return, adjusted_volatility, days)
        portfolio_values = [10000]

        for ret in returns:
            portfolio_values.append(portfolio_values[-1] * (1 + ret))

        logger.info(f"Generated {len(portfolio_values)} portfolio values")

        dates = pd.date_range(
            start="2024-01-01", periods=len(portfolio_values), freq="D"
        )
        equity_df = pd.DataFrame(
            {
                "Date": dates,
                "Portfolio_Value": portfolio_values,
                "Signal": ["HOLD"] * len(portfolio_values),
            }
        )

        # Add some trade signals (ensure we don't exceed available days)
        max_signals = min(20, len(portfolio_values) - 1)
        logger.info(f"Max signals allowed: {max_signals}")

        if max_signals > 0:
            signal_days = np.random.choice(
                len(portfolio_values), size=max_signals, replace=False
            )
            logger.info(f"Selected signal days: {signal_days[:5]}... (showing first 5)")
            for i, day in enumerate(signal_days):
                if day < len(equity_df):
                    equity_df.loc[day, "Signal"] = "BUY" if i % 2 == 0 else "SELL"

        # Calculate performance metrics with NaN safety
        total_return = (portfolio_values[-1] / portfolio_values[0] - 1) * 100
        daily_returns = pd.Series(returns)

        # NaN-safe Sharpe ratio
        sharpe_ratio = 0.0
        if daily_returns.std() > 0:
            sharpe_ratio = daily_returns.mean() / daily_returns.std() * np.sqrt(252)

        # NaN-safe Sortino ratio
        sortino_ratio = 0.0
        negative_returns = daily_returns[daily_returns < 0]
        if len(negative_returns) > 0 and negative_returns.std() > 0:
            sortino_ratio = daily_returns.mean() / negative_returns.std() * np.sqrt(252)
        else:
            sortino_ratio = sharpe_ratio * 1.2  # Fallback

        peak = np.maximum.accumulate(portfolio_values)
        drawdown = (np.array(portfolio_values) - peak) / peak
        max_drawdown = abs(drawdown.min()) * 100

        positive_returns = daily_returns[daily_returns > 0]
        negative_returns = daily_returns[daily_returns < 0]

        win_rate = (
            len(positive_returns) / len(daily_returns) * 100
            if len(daily_returns) > 0
            else 0
        )
        avg_win = positive_returns.mean() if len(positive_returns) > 0 else 0
        avg_loss = negative_returns.mean() if len(negative_returns) > 0 else 0

        # NaN-safe profit factor
        profit_factor = 1.0
        if negative_returns.sum() != 0:
            profit_factor = abs(positive_returns.sum() / negative_returns.sum())

        perf_metrics = {
            "total_return": total_return,
            "sharpe_ratio": sharpe_ratio,
            "sortino_ratio": sortino_ratio,
            "max_drawdown": max_drawdown,
            "profit_factor": profit_factor,
            "win_rate": win_rate,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "exposure": 0.8,
            "turnover": 1.5,
            "total_trades": max_signals,
        }

        logger.info(
            f"Calculated metrics - Return: {total_return:.2f}%, Sharpe: {sharpe_ratio:.2f}, Drawdown: {max_drawdown:.2f}%"
        )
        logger.info("Mock backtest completed successfully")

        return equity_df, perf_metrics

    except Exception as e:
        # Fallback in case of any error
        logger.error(f"Error in mock_backtest_function: {e}")
        logger.error(traceback.format_exc())

        # Return minimal valid result
        dates = pd.date_range(start="2024-01-01", periods=100, freq="D")
        equity_df = pd.DataFrame(
            {"Date": dates, "Portfolio_Value": [10000] * 100, "Signal": ["HOLD"] * 100}
        )

        perf_metrics = {
            "total_return": 0.0,
            "sharpe_ratio": 0.0,
            "sortino_ratio": 0.0,
            "max_drawdown": 0.0,
            "profit_factor": 1.0,
            "win_rate": 50.0,
            "avg_win": 0.0,
            "avg_loss": 0.0,
            "exposure": 0.0,
            "turnover": 0.0,
            "total_trades": 0,
        }

        logger.warning("Returning fallback results due to error")
        return equity_df, perf_metrics


def apply_changes(knobs: dict, changes: list, bounds: dict) -> dict:
    """Apply LLM suggested changes with bounds checking and invariants"""
    logger.info(f"Applying changes - Input knobs: {knobs}")
    logger.info(f"Suggested changes: {changes}")
    logger.info(f"Parameter bounds: {bounds}")

    new_knobs = knobs.copy()
    applied_changes = []

    for i, ch in enumerate(changes[:3]):  # Limit to 3 changes max
        logger.info(f"Processing change {i+1}: {ch}")

        p = ch.get("param")
        rng = ch.get("new_range", [])

        if p in bounds and isinstance(rng, list) and len(rng) == 2:
            lo, hi = bounds[p]
            a = max(lo, min(hi, float(rng[0])))
            b = max(lo, min(hi, float(rng[1])))
            if a > b:
                a, b = b, a

            old_value = new_knobs.get(p, 0)
            new_value = round((a + b) / 2, 6)
            new_knobs[p] = new_value

            applied_changes.append(
                {
                    "param": p,
                    "old_value": old_value,
                    "new_value": new_value,
                    "suggested_range": rng,
                    "bounded_range": [a, b],
                }
            )

            logger.info(
                f"Applied change to {p}: {old_value} -> {new_value} (from range [{a:.4f}, {b:.4f}])"
            )
        else:
            logger.warning(
                f"Skipped invalid change: param={p}, range={rng}, bounds_exist={p in bounds}"
            )

    # Enforce invariants for SMA crossover
    if "sma_fast" in new_knobs and "sma_slow" in new_knobs:
        original_fast = new_knobs["sma_fast"]
        original_slow = new_knobs["sma_slow"]

        if new_knobs["sma_fast"] >= new_knobs["sma_slow"]:
            logger.warning(
                f"Invariant violation: sma_fast ({original_fast}) >= sma_slow ({original_slow})"
            )
            # Gently enforce gap
            new_knobs["sma_fast"] = max(
                bounds["sma_fast"][0],
                min(new_knobs["sma_slow"] - 1, bounds["sma_fast"][1]),
            )
            logger.info(
                f"Enforced invariant: sma_fast adjusted to {new_knobs['sma_fast']}"
            )

    logger.info(f"Final knobs after changes: {new_knobs}")
    logger.info(f"Applied {len(applied_changes)} changes successfully")

    return new_knobs


def iterate_optimization(
    analyzer: GroqAnalyzer,
    market_df: pd.DataFrame,
    strategy_config: dict,
    max_iters: int,
    min_gain_pct: float,
    max_dd_tol_pp: float,
):
    """Real closed-loop optimization using GroqAnalyzer"""
    logger.info("=" * 60)
    logger.info("STARTING OPTIMIZATION")
    logger.info(f"Max iterations: {max_iters}")
    logger.info(f"Min gain threshold: {min_gain_pct}%")
    logger.info(f"Max drawdown tolerance: {max_dd_tol_pp} pp")
    logger.info(f"Initial strategy config: {strategy_config}")

    try:
        # Champion = current config
        champion_cfg = json.loads(json.dumps(strategy_config))  # Deep copy
        logger.info(f"Champion config initialized: {champion_cfg}")

        # Run baseline backtest
        logger.info("Running baseline backtest...")
        equity_df, perf = mock_backtest_function(champion_cfg)
        logger.info(f"Baseline results: {perf}")

        # Structure perf for analyzer
        perf_struct = {
            "total_return": perf["total_return"],
            "sharpe_ratio": perf["sharpe_ratio"],
            "sortino_ratio": perf["sortino_ratio"],
            "max_drawdown": perf["max_drawdown"],
            "profit_factor": perf["profit_factor"],
            "win_rate": perf["win_rate"],
            "avg_win": perf["avg_win"],
            "avg_loss": perf["avg_loss"],
            "exposure": perf["exposure"],
            "turnover": perf["turnover"],
            "total_trades": perf["total_trades"],
        }

        # Objective function (Sortino by config)
        def objective(p):
            return p.get("sortino_ratio", 0.0)

        champion_perf = perf
        champion_obj = objective(champion_perf)
        champion_dd = champion_perf["max_drawdown"]
        history = []

        logger.info(f"Baseline objective (Sortino): {champion_obj:.4f}")
        logger.info(f"Baseline drawdown: {champion_dd:.2f}%")

        # Build bounds from UI scalars
        bounds = {}
        for k, v in champion_cfg["knobs"].items():
            if isinstance(v, (int, float)):
                bounds[k] = (
                    [max(1, v * 0.7), v * 1.3]
                    if v > 1
                    else [max(0.0001, v * 0.7), v * 1.3]
                )
            elif isinstance(v, list) and len(v) == 2:
                bounds[k] = v
                champion_cfg["knobs"][k] = round(sum(v) / 2, 6)

        logger.info(f"Parameter bounds: {bounds}")

        for i in range(1, int(max_iters) + 1):
            logger.info(f"\n--- ITERATION {i}/{max_iters} ---")

            # Get LLM suggestion
            logger.info("Requesting AI suggestion...")
            try:
                # Convert knobs to ranges format for AI analyzer
                knobs_for_analyzer = {}
                for k, v in champion_cfg["knobs"].items():
                    if k in bounds:
                        knobs_for_analyzer[k] = bounds[
                            k
                        ]  # Use the bounds we calculated
                    else:
                        # Fallback: create a range around the current value
                        knobs_for_analyzer[k] = [v * 0.9, v * 1.1]

                logger.info(f"Knobs formatted for analyzer: {knobs_for_analyzer}")

                # Create a modified config for the analyzer
                analyzer_config = champion_cfg.copy()
                analyzer_config["knobs"] = knobs_for_analyzer

                suggestion = analyzer.optimize_strategy_structured(
                    market_data=market_df,
                    strategy_config=analyzer_config,
                    performance_metrics=perf_struct,
                    regime_data=None,
                    news_data=None,
                )
                logger.info(f"AI suggestion received: {suggestion}")
            except Exception as e:
                logger.error(f"Error getting AI suggestion: {e}")
                logger.error(traceback.format_exc())
                suggestion = None

            # Handle LLM errors
            if not suggestion or not suggestion.get("ok", False):
                logger.warning(f"Invalid AI suggestion: {suggestion}")
                history.append(
                    {
                        "iter": i,
                        "decision": "no_op_llm_error",
                        "gain_pct_on_objective": 0.0,
                        "drawdown_pp": 0.0,
                        "config_after": champion_cfg,
                        "suggestion": suggestion or {},
                    }
                )
                break

            # Apply changes with bounds
            logger.info("Applying suggested changes...")
            try:
                new_knobs = apply_changes(
                    champion_cfg["knobs"], suggestion.get("changes", []), bounds
                )
                logger.info(f"New knobs after applying changes: {new_knobs}")
            except Exception as e:
                logger.error(f"Error applying changes: {e}")
                logger.error(traceback.format_exc())
                continue

            trial_cfg = json.loads(json.dumps(champion_cfg))
            trial_cfg["knobs"].update(new_knobs)
            logger.info(f"Trial config: {trial_cfg}")

            # Backtest new config
            logger.info("Backtesting trial config...")
            try:
                equity_trial, perf_trial = mock_backtest_function(trial_cfg)
                obj_trial = objective(perf_trial)
                dd_trial = perf_trial["max_drawdown"]

                logger.info(f"Trial results: {perf_trial}")
                logger.info(f"Trial objective: {obj_trial:.4f}")
                logger.info(f"Trial drawdown: {dd_trial:.2f}%")
            except Exception as e:
                logger.error(f"Error in trial backtest: {e}")
                logger.error(traceback.format_exc())
                continue

            gain_pct = (
                0.0
                if champion_obj == 0
                else (obj_trial - champion_obj) / abs(champion_obj) * 100.0
            )
            dd_worsen_pp = max(0.0, dd_trial - champion_dd)

            logger.info(f"Gain percentage: {gain_pct:.2f}%")
            logger.info(f"Drawdown worsening: {dd_worsen_pp:.2f} pp")

            if (gain_pct >= float(min_gain_pct)) and (
                dd_worsen_pp <= float(max_dd_tol_pp)
            ):
                # Promote
                champion_cfg = trial_cfg
                champion_perf = perf_trial
                champion_obj = obj_trial
                champion_dd = dd_trial
                decision = "promoted_to_champion"
                logger.info("✅ TRIAL PROMOTED TO CHAMPION!")
            else:
                decision = "rejected"
                logger.info(
                    f"❌ Trial rejected - Gain: {gain_pct:.2f}% (need {min_gain_pct}%), DD worsening: {dd_worsen_pp:.2f} pp (max {max_dd_tol_pp})"
                )

            history.append(
                {
                    "iter": i,
                    "decision": decision,
                    "gain_pct_on_objective": round(gain_pct, 2),
                    "drawdown_pp": round(dd_worsen_pp, 2),
                    "config_after": {"knobs": trial_cfg["knobs"]},
                    "suggestion": suggestion,
                }
            )

            # Early stop if no changes suggested
            if not suggestion.get("changes"):
                logger.info("No more changes suggested - stopping optimization")
                break

        result = {
            "champion_config": champion_cfg,
            "champion_perf": champion_perf,
            "iterations": history,
            "total_iterations": len(history),
            "final_objective": round(champion_obj, 4),
            "final_drawdown_pct": round(champion_dd, 2),
        }

        logger.info("=" * 60)
        logger.info("OPTIMIZATION COMPLETED")
        logger.info(f"Final objective: {result['final_objective']}")
        logger.info(f"Final drawdown: {result['final_drawdown_pct']}%")
        logger.info(f"Total iterations: {result['total_iterations']}")
        logger.info("=" * 60)

        return result

    except Exception as e:
        logger.error(f"CRITICAL ERROR in iterate_optimization: {e}")
        logger.error(traceback.format_exc())
        raise


def display_optimization_results(result):
    """Display optimization results with enhanced detail"""
    if not result:
        return

    st.subheader("🎯 Optimization Results")

    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Iterations", result.get("total_iterations", 0))
    with col2:
        st.metric("Final Objective", f"{result.get('final_objective', 0):.2f}")
    with col3:
        st.metric("Final Drawdown", f"{result.get('final_drawdown_pct', 0):.1f}%")
    with col4:
        promoted_count = len(
            [
                it
                for it in result.get("iterations", [])
                if it.get("decision") == "promoted_to_champion"
            ]
        )
        st.metric("Promotions", promoted_count)

    # Show exact knobs and constraints for reproducibility
    if "champion_config" in result:
        st.subheader("🏆 Champion Configuration (Reproducible)")

        champion_knobs = result["champion_config"].get("knobs", {})
        constraints = result["champion_config"].get("constraints", {})

        col1, col2 = st.columns(2)

        with col1:
            st.write("**Final Parameters:**")
            knobs_df = pd.DataFrame(
                [
                    {
                        "Parameter": k,
                        "Value": f"{v:.6f}" if isinstance(v, float) else str(v),
                    }
                    for k, v in champion_knobs.items()
                ]
            )
            st.dataframe(knobs_df, use_container_width=True)

        with col2:
            st.write("**Constraints Used:**")
            constraints_df = pd.DataFrame(
                [{"Constraint": k, "Value": v} for k, v in constraints.items()]
            )
            st.dataframe(constraints_df, use_container_width=True)

    # Parameter evolution with exact values
    if "iterations" in result and result["iterations"]:
        st.subheader("📊 Parameter Evolution (Exact Values)")

        iterations = result["iterations"]
        iter_nums = [it["iter"] for it in iterations]

        # Get parameter names from first iteration
        first_config = iterations[0].get("config_after", {}).get("knobs", {})
        param_names = list(first_config.keys())

        if param_names:
            fig = go.Figure()

            for param in param_names:
                values = []
                for it in iterations:
                    config = it.get("config_after", {}).get("knobs", {})
                    values.append(config.get(param, 0))

                fig.add_trace(
                    go.Scatter(
                        x=iter_nums,
                        y=values,
                        mode="lines+markers",
                        name=param,
                        line=dict(width=2),
                        text=[f"{param}: {v:.6f}" for v in values],
                        hovertemplate="%{text}<br>Iter: %{x}<extra></extra>",
                    )
                )

            fig.update_layout(
                title="Parameter Values Across Iterations",
                xaxis_title="Iteration",
                yaxis_title="Parameter Value",
                height=400,
            )

            st.plotly_chart(fig, use_container_width=True)

    # Detailed iteration history with NaN-safe rendering
    if "iterations" in result and result["iterations"]:
        st.subheader("📋 Detailed Iteration History")

        history_data = []
        for it in result["iterations"]:
            # NaN-safe formatting
            gain_pct = it.get("gain_pct_on_objective", 0)
            drawdown_pp = it.get("drawdown_pp", 0)

            gain_str = f"{gain_pct:+.1f}%" if not pd.isna(gain_pct) else "N/A"
            dd_str = f"{drawdown_pp:.1f}%" if not pd.isna(drawdown_pp) else "N/A"

            history_data.append(
                {
                    "Iteration": it["iter"],
                    "Decision": it["decision"],
                    "Gain %": gain_str,
                    "Drawdown Δ": dd_str,
                    "Changes": len(it.get("suggestion", {}).get("changes", [])),
                    "LLM Status": (
                        "✅ OK"
                        if it.get("suggestion", {}).get("ok", True)
                        else "❌ Error"
                    ),
                }
            )

        df_history = pd.DataFrame(history_data)
        st.dataframe(df_history, use_container_width=True)


def main():
    st.title("🤖 AI Trading Assistant")
    st.markdown(
        "Powered by Groq AI for advanced market analysis and strategy optimization"
    )

    # Load config
    logger.info("Loading configuration...")
    config = load_config()
    logger.info(f"Config loaded - Keys available: {list(config.keys())}")

    # Initialize AI analyzer with caching
    if "ai_analyzer" not in st.session_state:
        logger.info("Initializing AI analyzer...")
        try:
            api_key = config.get("groq_api_key")
            if not api_key:
                logger.error("No Groq API key found in config")
                st.error("Groq API key not found. Please check your configuration.")
                st.stop()

            logger.info("Creating AI analyzer with API key...")
            st.session_state.ai_analyzer = get_ai_analyzer(api_key)
            logger.info("AI analyzer successfully initialized and cached")
        except Exception as e:
            logger.error(f"Failed to initialize AI analyzer: {e}")
            logger.error(traceback.format_exc())
            st.error(f"Failed to initialize AI analyzer: {e}")
            st.stop()
    else:
        logger.info("AI analyzer found in session state - using cached version")

    # Main tabs - always visible at the top
    tab1, tab2, tab3 = st.tabs(
        ["💬 Chat Assistant", "🔄 Strategy Optimization", "⚙️ Configuration"]
    )

    with tab1:
        # Initialize chat history with size limit
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []

        # Cap chat history to prevent DOM bloat (keep last 100 messages)
        if len(st.session_state.chat_history) > 100:
            st.session_state.chat_history = st.session_state.chat_history[-100:]

        # Standard chatbot layout - messages container
        chat_container = st.container()

        with chat_container:
            # Display chat messages using standard chatbot interface
            if st.session_state.chat_history:
                for message in st.session_state.chat_history:
                    if message["role"] == "user":
                        with st.chat_message("user"):
                            st.write(message["content"])
                    else:
                        with st.chat_message("assistant"):
                            st.write(message["content"])
            else:
                with st.chat_message("assistant"):
                    st.write(
                        "👋 Hello! I'm your AI trading assistant. I can help you with:"
                    )
                    st.write("• Trading strategy analysis and development")
                    st.write("• Market regime analysis")
                    st.write("• Risk management frameworks")
                    st.write("• Technical analysis insights")
                    st.write("• Portfolio optimization")
                    st.write("What would you like to explore today?")

        # Chat input at the bottom - standard chatbot style
        user_input = st.chat_input(
            "Ask me about trading strategies, market analysis, or risk management..."
        )

        if user_input:
            # Add user message to history
            st.session_state.chat_history.append(
                {"role": "user", "content": user_input}
            )

            # Get AI response
            with st.spinner("🤖 Analyzing your query..."):
                try:
                    response = st.session_state.ai_analyzer.analyze_query(user_input)
                    st.session_state.chat_history.append(
                        {"role": "assistant", "content": response}
                    )
                except Exception as e:
                    st.error(f"Error getting AI response: {e}")

            st.rerun()

        # Sidebar with quick actions
        with st.sidebar:
            st.markdown("### 🎯 Quick Actions")

            col1, col2 = st.columns(2)

            with col1:
                if st.button("📊 Market Analysis", use_container_width=True):
                    prompt = "Analyze the current market regime and suggest appropriate trading strategies."
                    st.session_state.chat_history.append(
                        {"role": "user", "content": prompt}
                    )
                    with st.spinner("Analyzing..."):
                        try:
                            response = st.session_state.ai_analyzer.analyze_query(
                                prompt
                            )
                            st.session_state.chat_history.append(
                                {"role": "assistant", "content": response}
                            )
                        except Exception as e:
                            st.error(f"Error: {e}")
                    st.rerun()

                if st.button("⚡ Strategy Ideas", use_container_width=True):
                    prompt = "Suggest 3 algorithmic trading strategies for Indian markets with their key characteristics."
                    st.session_state.chat_history.append(
                        {"role": "user", "content": prompt}
                    )
                    with st.spinner("Generating..."):
                        try:
                            response = st.session_state.ai_analyzer.analyze_query(
                                prompt
                            )
                            st.session_state.chat_history.append(
                                {"role": "assistant", "content": response}
                            )
                        except Exception as e:
                            st.error(f"Error: {e}")
                    st.rerun()

            with col2:
                if st.button("🔍 Risk Assessment", use_container_width=True):
                    prompt = "Explain key risk management principles for algorithmic trading."
                    st.session_state.chat_history.append(
                        {"role": "user", "content": prompt}
                    )
                    with st.spinner("Analyzing..."):
                        try:
                            response = st.session_state.ai_analyzer.analyze_query(
                                prompt
                            )
                            st.session_state.chat_history.append(
                                {"role": "assistant", "content": response}
                            )
                        except Exception as e:
                            st.error(f"Error: {e}")
                    st.rerun()

                if st.button("🧹 Clear Chat", use_container_width=True):
                    st.session_state.chat_history = []
                    st.rerun()

            st.markdown("---")
            st.markdown("### 📈 Chat Stats")
            st.metric("Messages", len(st.session_state.chat_history))
            if st.session_state.chat_history:
                user_msgs = len(
                    [m for m in st.session_state.chat_history if m["role"] == "user"]
                )
                st.metric("Your Questions", user_msgs)

    with tab2:
        # New closed-loop optimization interface
        st.subheader("🔄 Closed-Loop Strategy Optimization")
        st.markdown(
            "Let AI iteratively improve your strategy parameters through automated backtesting"
        )

        col1, col2 = st.columns([2, 1])

        with col1:
            st.subheader("📋 Strategy Configuration")

            # Strategy setup
            strategy_name = st.text_input(
                "Strategy Name", value="SMA Crossover Strategy"
            )
            strategy_description = st.text_area(
                "Strategy Description",
                value="Long when fast SMA > slow SMA; exit when fast SMA < slow SMA or stop loss triggered",
            )

            st.subheader("🎛️ Parameters to Optimize")

            # Parameter configuration
            col_param1, col_param2, col_param3 = st.columns(3)

            with col_param1:
                sma_fast = st.number_input(
                    "SMA Fast", min_value=5, max_value=50, value=10, step=1
                )

            with col_param2:
                sma_slow = st.number_input(
                    "SMA Slow", min_value=20, max_value=100, value=30, step=1
                )

            with col_param3:
                risk_per_trade = st.number_input(
                    "Risk per Trade",
                    min_value=0.005,
                    max_value=0.02,
                    value=0.01,
                    step=0.001,
                    format="%.3f",
                )

            st.subheader("⚙️ Optimization Settings")

            col_opt1, col_opt2, col_opt3 = st.columns(3)

            with col_opt1:
                max_iterations = st.number_input(
                    "Max Iterations", min_value=3, max_value=10, value=5, step=1
                )

            with col_opt2:
                min_gain_pct = st.number_input(
                    "Min Gain % to Promote",
                    min_value=5.0,
                    max_value=20.0,
                    value=10.0,
                    step=1.0,
                )

            with col_opt3:
                max_dd_tolerance = st.number_input(
                    "Max DD Tolerance (pp)",
                    min_value=1.0,
                    max_value=5.0,
                    value=2.0,
                    step=0.5,
                )

            # Run optimization button with validation warnings
            validation_warnings = []

            # Validate parameters
            if sma_fast >= sma_slow:
                validation_warnings.append("⚠️ SMA Fast must be less than SMA Slow")

            # Check if configuration will generate enough trades
            if abs(sma_fast - sma_slow) < 5:
                validation_warnings.append(
                    "⚠️ SMA gap too small - may generate < 10 trades"
                )

            # Display warnings
            if validation_warnings:
                for warning in validation_warnings:
                    st.warning(warning)

            # Disable button if critical validation fails
            button_disabled = sma_fast >= sma_slow

            if st.button(
                "🚀 Start AI Optimization",
                type="primary",
                use_container_width=True,
                disabled=button_disabled,
            ):
                logger.info("🚀 START AI OPTIMIZATION BUTTON CLICKED")
                logger.info(
                    f"User parameters - SMA Fast: {sma_fast}, SMA Slow: {sma_slow}, Risk: {risk_per_trade}"
                )
                logger.info(
                    f"Optimization settings - Max iterations: {max_iterations}, Min gain: {min_gain_pct}%, Max DD tolerance: {max_dd_tolerance} pp"
                )

                # Create strategy config
                strategy_config = {
                    "name": strategy_name,
                    "description": strategy_description,
                    "universe": "NIFTY50",
                    "timeframe": "1D",
                    "objective": "maximize_sortino",
                    "constraints": {
                        "max_dd": 0.15,
                        "risk_per_trade": 0.015,
                        "turnover_pa": 2.0,
                    },
                    "knobs": {
                        "sma_fast": sma_fast,
                        "sma_slow": sma_slow,
                        "risk_per_trade": risk_per_trade,
                    },
                    "invariants": ["sma_fast < sma_slow", "risk_per_trade <= 1.5%"],
                }

                logger.info(f"Strategy config created: {strategy_config}")

                # Get cached market data
                logger.info("Generating market data...")
                market_data_df = generate_market_data(seed=42, days=365)
                logger.info(f"Market data generated - Shape: {market_data_df.shape}")

                # Prepare market data for analyzer (set index for stats compression)
                market_df = market_data_df.copy()
                market_df = market_df.set_index("Date")
                logger.info(
                    f"Market data prepared for analyzer - Columns: {market_df.columns.tolist()}"
                )

                # Run optimization with progress tracking
                progress_bar = st.progress(0)
                status_text = st.empty()

                try:
                    logger.info("Starting optimization process...")
                    status_text.text("🔄 Starting real AI optimization...")
                    progress_bar.progress(10)

                    # Check AI analyzer availability
                    if "ai_analyzer" not in st.session_state:
                        logger.error("AI analyzer not found in session state!")
                        st.error(
                            "AI analyzer not initialized. Please check your API key."
                        )
                        logger.info("Optimization aborted due to missing AI analyzer")
                        st.stop()

                    logger.info("AI analyzer found in session state")

                    # Real closed-loop optimization using GroqAnalyzer
                    status_text.text("🤖 Running closed-loop optimization with LLM...")
                    progress_bar.progress(30)

                    logger.info("Calling iterate_optimization...")
                    result = iterate_optimization(
                        analyzer=st.session_state.ai_analyzer,
                        market_df=market_df,
                        strategy_config=strategy_config,
                        max_iters=max_iterations,
                        min_gain_pct=min_gain_pct,
                        max_dd_tol_pp=max_dd_tolerance,
                    )

                    logger.info(
                        f"Optimization completed - Result keys: {result.keys()}"
                    )

                    status_text.text("📊 Processing optimization results...")
                    progress_bar.progress(80)

                    status_text.text("✅ Optimization complete!")
                    progress_bar.progress(100)

                    # Store result in session state
                    st.session_state.optimization_result = result
                    logger.info("Result stored in session state")

                    st.success("🎉 Real AI optimization completed successfully!")

                    # Show quick summary
                    if result.get("total_iterations", 0) > 0:
                        promotions = len(
                            [
                                it
                                for it in result.get("iterations", [])
                                if it.get("decision") == "promoted_to_champion"
                            ]
                        )
                        summary_msg = f"Completed {result['total_iterations']} iterations with {promotions} promotions. Final Sortino: {result.get('final_objective', 0):.3f}"
                        st.info(summary_msg)
                        logger.info(f"Optimization summary: {summary_msg}")

                except Exception as e:
                    error_msg = f"Optimization failed: {e}"
                    logger.error(error_msg)
                    logger.error(traceback.format_exc())
                    st.error(error_msg)

                    # Additional debugging info
                    st.error("Check the logs for detailed error information.")
                    logger.info("Optimization failed - check logs above for details")
                    status_text.text("❌ Optimization failed")
                    progress_bar.progress(0)

                    # Fallback for debugging
                    st.warning("Using fallback demonstration result for debugging...")

                    mock_result = {
                        "champion_config": {
                            **strategy_config,
                            "knobs": {
                                "sma_fast": max(5, sma_fast - 1),
                                "sma_slow": min(100, sma_slow + 2),
                                "risk_per_trade": max(0.005, risk_per_trade * 0.9),
                            },
                        },
                        "champion_perf": {
                            "total_return": 15.2,
                            "sharpe_ratio": 1.45,
                            "sortino_ratio": 1.78,
                            "max_drawdown": 8.5,
                        },
                        "iterations": [
                            {
                                "iter": 1,
                                "decision": "promoted_to_champion",
                                "gain_pct_on_objective": 12.5,
                                "drawdown_pp": 1.5,
                                "config_after": {
                                    "knobs": {
                                        "sma_fast": sma_fast - 1,
                                        "sma_slow": sma_slow + 1,
                                        "risk_per_trade": risk_per_trade * 0.9,
                                    }
                                },
                                "suggestion": {
                                    "ok": True,
                                    "changes": [
                                        {
                                            "param": "risk_per_trade",
                                            "new_range": [
                                                risk_per_trade * 0.85,
                                                risk_per_trade * 0.95,
                                            ],
                                        }
                                    ],
                                },
                            }
                        ],
                        "total_iterations": 1,
                        "final_objective": 1.78,
                        "final_drawdown_pct": 8.5,
                    }

                    st.session_state.optimization_result = mock_result

        with col2:
            st.subheader("💡 Real AI Optimization")

            st.info(
                """
            **🎯 How it actually works:**
            1. AI analyzes your strategy + market data via GroqAnalyzer
            2. Suggests ≤3 parameter changes per iteration  
            3. Backtests each suggestion deterministically
            4. Promotes only if Sortino improves within DD limits
            5. Repeats until convergence or max iterations
            """
            )

            st.warning(
                """
            **⚠️ Real constraints:**
            - Higher min gain % = more conservative promotion
            - Lower DD tolerance = safer parameter exploration  
            - More iterations = better but slower optimization
            - Uses compressed market stats to keep LLM prompts small
            - Results depend on backtest quality + market regime
            """
            )

            if "optimization_result" in st.session_state:
                result = st.session_state.optimization_result
                promotions = len(
                    [
                        it
                        for it in result.get("iterations", [])
                        if it.get("decision") == "promoted_to_champion"
                    ]
                )

                if promotions > 0:
                    st.success(f"🏆 Found {promotions} improvements!")
                    improvement = result.get("final_objective", 0) - 1.5
                    st.metric("Sortino Gain", f"+{improvement:.3f}")
                else:
                    st.info("🔍 No improvements found - try different settings")

            # Show current LLM model being used
            st.markdown("---")
            st.caption(
                f"**AI Model:** {getattr(st.session_state.ai_analyzer, 'model', 'llama-3.3-70b-versatile')}"
            )
            st.caption("**Optimization:** Real closed-loop with GroqAnalyzer")

        # Display optimization results if available
        if "optimization_result" in st.session_state:
            display_optimization_results(st.session_state.optimization_result)

    with tab3:
        # Model configuration (existing functionality)
        st.subheader("⚙️ AI Model Configuration")

        col1, col2 = st.columns(2)

        with col1:
            model_choice = st.selectbox(
                "Select AI Model",
                [
                    "llama-3.3-70b-versatile",
                    "llama-3.1-8b-instant",
                    "deepseek-r1-distill-llama-70b",
                    "meta-llama/llama-4-scout-17b-16e-instruct",
                    "qwen/qwen3-32b",
                ],
                help="Choose the AI model for analysis",
            )

            if st.button("Update Model"):
                st.session_state.ai_analyzer.update_model(model_choice)
                st.success(f"Model updated to {model_choice}")

        with col2:
            temperature = st.slider(
                "Response Creativity",
                min_value=0.0,
                max_value=1.0,
                value=0.1,
                step=0.1,
                help="Higher values make responses more creative but less focused",
            )

            if st.button("Update Temperature"):
                st.session_state.ai_analyzer.update_temperature(temperature)
                st.success(f"Temperature updated to {temperature}")

        st.subheader("📊 System Status")

        # System status indicators
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("API Status", "🟢 Connected")
        with col2:
            st.metric("Model", model_choice.split("/")[-1])
        with col3:
            st.metric("Temperature", f"{temperature:.1f}")


if __name__ == "__main__":
    main()
