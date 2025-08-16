import sys
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

# --- paths ---
ROOT = Path(__file__).resolve().parent
sys.path.append(str(ROOT.parent / "src"))

st.set_page_config(page_title="Strategy Builder", page_icon="🔧", layout="wide")
st.title("🔧 Strategy Builder")
st.caption("educational prototype • no investment advice • paper testing only")

# ===================== utils =====================

def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    roll_up = up.ewm(alpha=1/period, adjust=False).mean()
    roll_down = down.ewm(alpha=1/period, adjust=False).mean()
    rs = roll_up / roll_down.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def bollinger_bands(series: pd.Series, period: int = 20, stds: float = 2.0):
    ma = series.rolling(period, min_periods=period).mean()
    sd = series.rolling(period, min_periods=period).std()
    return ma, ma + stds * sd, ma - stds * sd

def max_drawdown(equity: pd.Series) -> float:
    peak = equity.cummax()
    dd = equity/peak - 1.0
    return float(dd.min())

def sharpe(returns: pd.Series, periods_per_year=252) -> float:
    if returns.std() == 0 or len(returns) == 0: return 0.0
    return float((returns.mean() / returns.std()) * np.sqrt(periods_per_year))

def sortino(returns: pd.Series, periods_per_year=252) -> float:
    downside = returns[returns < 0]
    if downside.std() == 0 or len(returns) == 0: return 0.0
    return float((returns.mean() / downside.std()) * np.sqrt(periods_per_year))

def profit_factor(returns: pd.Series) -> float:
    gains = returns[returns > 0].sum()
    losses = -returns[returns < 0].sum()
    return float(gains / losses) if losses > 1e-12 else 1.0

def simulate_prices(n_days: int = 240, seed: int = 42, start=1000.0):
    rng = np.random.default_rng(seed)
    rets = rng.normal(0.0006, 0.012, n_days)  # mild drift, daily vol ~1.2%
    dates = pd.date_range("2024-01-01", periods=n_days, freq="B")  # Business days only
    price = start * (1 + pd.Series(rets, index=dates)).cumprod()
    vol = pd.Series(rng.lognormal(12, 0.3, n_days).astype(int), index=dates)  # Reduced mean from 16 to 12
    return pd.DataFrame({"Date": dates, "Close": price.values, "Volume": vol.values})

# ===================== signal generators =====================

def signal_momentum(df: pd.DataFrame, period: int, threshold_pct: float, vol_mult: float = 1.2):
    px = df["Close"]
    mom = px.pct_change(periods=period) * 100
    vol_ok = df["Volume"] > df["Volume"].rolling(20).mean() * vol_mult
    buy  = (mom > threshold_pct) & vol_ok
    sell = (mom < -threshold_pct) & vol_ok
    sig = pd.Series("HOLD", index=df.index)
    sig[buy] = "BUY"
    sig[sell] = "SELL"
    return sig

def signal_mean_rev(df: pd.DataFrame, rsi_period: int, rsi_lo: int, rsi_hi: int, bb_period: int = 20, bb_std: float = 2.0):
    px = df["Close"]
    r = rsi(px, rsi_period)
    mid, upper, lower = bollinger_bands(px, bb_period, bb_std)
    buy  = (r < rsi_lo) & (px < lower)
    sell = (r > rsi_hi) & (px > upper)
    sig = pd.Series("HOLD", index=df.index)
    sig[buy] = "BUY"
    sig[sell] = "SELL"
    return sig

def signal_breakout(df: pd.DataFrame, lookback: int, vol_mult: float):
    hi = df["Close"].rolling(lookback).max()
    lo = df["Close"].rolling(lookback).min()
    vol_ok = df["Volume"] > df["Volume"].rolling(20).mean() * vol_mult
    buy  = (df["Close"] > hi.shift(1)) & vol_ok
    sell = (df["Close"] < lo.shift(1)) & vol_ok
    sig = pd.Series("HOLD", index=df.index)
    sig[buy] = "BUY"
    sig[sell] = "SELL"
    return sig

# ===================== backtester (no look-ahead) =====================

def run_backtest(df: pd.DataFrame,
                 signal: pd.Series,
                 stop_loss_pct: float,
                 take_profit_pct: float,
                 max_pos_frac: float,
                 cost_bps: float = 0.0):
    """
    Simple daily close-to-close backtest:
    position[t] in {-1,0,1}; P&L uses position[t-1] * ret[t].
    Stop/TP checked on close vs entry; when hit → flat next bar.
    Transaction costs applied on position changes.
    """
    df = df.copy()
    df["Signal"] = signal.fillna("HOLD")
    ret = df["Close"].pct_change().fillna(0.0)

    position = np.zeros(len(df), dtype=int)
    entry_px = np.full(len(df), np.nan)
    curr_pos = 0
    last_entry = np.nan
    exit_next = 0  # 0=none, 1=flat next bar

    for i in range(1, len(df)):
        # Handle exit_next flag (flat next bar after stop/TP trigger)
        if exit_next:
            curr_pos = 0
            last_entry = np.nan
            exit_next = 0
        
        sig = df.at[df.index[i], "Signal"]
        px = df.at[df.index[i], "Close"]

        # check stop / tp if in a position (trigger exit next bar)
        if curr_pos != 0 and not np.isnan(last_entry):
            if curr_pos == 1:
                if px <= last_entry * (1 - stop_loss_pct) or px >= last_entry * (1 + take_profit_pct):
                    exit_next = 1
            elif curr_pos == -1:
                if px >= last_entry * (1 + stop_loss_pct) or px <= last_entry * (1 - take_profit_pct):
                    exit_next = 1

        # new signals set direction, persisted until opposite or stop/tp
        if not exit_next:  # Only take new signals if not exiting next bar
            if sig == "BUY":
                if curr_pos != 1:
                    curr_pos = 1
                    last_entry = px
            elif sig == "SELL":
                if curr_pos != -1:
                    curr_pos = -1
                    last_entry = px
            # else HOLD: maintain

        position[i] = curr_pos
        entry_px[i] = last_entry

    # Calculate position changes for cost calculation
    changes = np.r_[0, np.abs(np.diff(position))]
    
    # strategy returns: apply previous position to current market return
    strat_ret = pd.Series(position[:-1], index=df.index[1:]) * ret.iloc[1:]
    
    # Apply transaction costs on position changes
    if cost_bps > 0:
        cost_series = pd.Series(changes[1:], index=df.index[1:]) * (cost_bps / 1e4)
        strat_ret = strat_ret - cost_series
    
    # position sizing (fraction of capital)
    strat_ret *= max_pos_frac

    equity = 100000 * (1 + strat_ret).cumprod()
    out = pd.DataFrame({
        "Date": df["Date"].iloc[1:].values,
        "Close": df["Close"].iloc[1:].values,
        "Signal": df["Signal"].iloc[1:].values,
        "Position": position[1:],
        "EntryPrice": entry_px[1:],
        "Return": strat_ret.values,
        "Portfolio_Value": equity.values
    }).set_index("Date")

    # metrics
    years = max(1e-9, len(out) / 252)
    total_ret = equity.iloc[-1] / equity.iloc[0] - 1
    
    # Calculate round-trips per year
    round_trips_per_year = (changes.sum() / 2) * (252 / len(out))
    
    metrics = {
        "total_return": round(total_ret * 100, 2),
        "cagr": round(((1 + total_ret) ** (1/years) - 1) * 100, 2),
        "sharpe_ratio": round(sharpe(out["Return"]), 2),
        "sortino_ratio": round(sortino(out["Return"]), 2),
        "max_drawdown": round(max_drawdown(equity) * 100, 2),
        "profit_factor": round(profit_factor(out["Return"]), 2),
        "win_rate": round((out["Return"] > 0).mean() * 100, 1),
        "turnover": round((np.abs(np.diff(position)) > 0).mean() * 252, 2),
        "round_trips_per_year": round(round_trips_per_year, 2),
        "total_trades": int((np.abs(np.diff(position)) > 0).sum())
    }
    return out.reset_index(), metrics

# ===================== Universal AI Optimization Functions =====================

def universal_iterate_optimization(ai_analyzer, strategy_config, current_params, market_data, max_iterations=5, min_gain_threshold=5.0, max_dd_tolerance=3.0):
    """
    Universal optimization algorithm that works with any strategy type
    """
    import json
    
    try:
        # Guard against Custom strategies
        if current_params.get("strategy_type") == "Custom":
            st.warning("⚠️ Custom strategies are not supported for AI optimization. Please select a predefined strategy type (Momentum, Mean Reversion, or Breakout) to use the AI optimizer.")
            return {"improvement_found": False, "reason": "Custom strategy not supported"}
        
        # Initialize tracking
        champion_config = strategy_config.copy()
        champion_params = current_params.copy()
        iterations = []
        
        # Run baseline backtest with actual strategy
        baseline_results = run_actual_backtest(champion_params, market_data)
        champion_perf = baseline_results
        baseline_objective = baseline_results.get('sortino_ratio', 0)
        baseline_drawdown = baseline_results.get('max_drawdown', 100)
        
        st.info(f"🎯 **Baseline Performance**: Sortino {baseline_objective:.3f}, Return {baseline_results.get('total_return', 0):.2f}%, DD {baseline_drawdown:.2f}%")
        
        # Create status placeholder for cleaner UI updates
        status_placeholder = st.empty()
        
        for iteration in range(max_iterations):
            status_placeholder.info(f"🔄 **Iteration {iteration + 1}/{max_iterations}** - Requesting AI suggestions...")
            
            # Get AI suggestion using universal prompt with JSON format
            suggestion = get_universal_ai_suggestion(
                ai_analyzer, 
                strategy_config, 
                champion_params, 
                champion_perf,
                market_data
            )
            
            if not suggestion.get('ok', False) or not suggestion.get('parameter_suggestions'):
                status_placeholder.warning(f"❌ No suggestions from AI in iteration {iteration + 1}")
                break
            
            # Limit to ≤3 edits per round (high-impact fix)
            raw_suggestions = suggestion.get('parameter_suggestions', [])
            suggestions = raw_suggestions[:3]  # Enforce budget
            
            if not suggestions:
                status_placeholder.warning(f"❌ No valid suggestions in iteration {iteration + 1}")
                break
            
            # Apply suggested parameter changes
            trial_params = apply_universal_changes(
                champion_params.copy(),
                suggestions,
                strategy_config['type']
            )
            
            # Early stop if no effective change (high-impact fix)
            if trial_params == champion_params:
                status_placeholder.info("✋ No effective parameter change; stopping optimization.")
                break
            
            status_placeholder.info(f"🧪 **Testing new parameters**: {trial_params}")
            
            # Run trial backtest with new parameters
            trial_results = run_actual_backtest(trial_params, market_data)
            trial_objective = trial_results.get('sortino_ratio', 0)
            trial_drawdown = trial_results.get('max_drawdown', 100)
            
            # Calculate improvement with proper handling for zero baseline (high-impact fix)
            if baseline_objective != 0:
                gain_pct = ((trial_objective - baseline_objective) / abs(baseline_objective) * 100)
            else:
                gain_pct = 0  # Will use different acceptance criteria below
                
            dd_worsening = trial_drawdown - baseline_drawdown
            
            # Accept improvements with special handling for baseline Sortino ≤ 0 (high-impact fix)
            if baseline_objective <= 0:
                accepted = (trial_objective > baseline_objective) and (dd_worsening <= max_dd_tolerance)
            else:
                accepted = (gain_pct >= min_gain_threshold) and (dd_worsening <= max_dd_tolerance)
            
            # Track iteration with delta information for rejected trials too
            iteration_data = {
                'iteration': iteration + 1,
                'objective': trial_objective,
                'total_return': trial_results.get('total_return', 0),
                'max_drawdown': trial_drawdown,
                'gain_pct': gain_pct,
                'dd_worsening': dd_worsening,
                'accepted': accepted,
                'parameters': trial_params.copy(),
                'ai_reasoning': suggestion.get('reasoning', 'No reasoning provided'),
                'objective_delta': trial_objective - baseline_objective,
                'suggestions_applied': len(suggestions)
            }
            iterations.append(iteration_data)
            
            if accepted:
                status_placeholder.success(f"✅ **Improvement Found!** Sortino: {trial_objective:.3f} (Δ{iteration_data['objective_delta']:+.3f}), DD: {trial_drawdown:.2f}% (+{dd_worsening:.1f}pp)")
                champion_params = trial_params
                champion_perf = trial_results
                baseline_objective = trial_objective
                baseline_drawdown = trial_drawdown
            else:
                if baseline_objective <= 0:
                    status_placeholder.warning(f"❌ **Trial Rejected** - Sortino: {trial_objective:.3f} (need > {baseline_objective:.3f}), DD worsening: {dd_worsening:.1f}pp (max {max_dd_tolerance})")
                else:
                    status_placeholder.warning(f"❌ **Trial Rejected** - Gain: {gain_pct:.1f}% (need {min_gain_threshold}%), DD worsening: {dd_worsening:.1f}pp (max {max_dd_tolerance})")
        
        # Clear status placeholder
        status_placeholder.empty()
        
        return {
            'champion_params': champion_params,
            'champion_perf': champion_perf,
            'iterations': iterations,
            'total_iterations': len(iterations),
            'final_objective': baseline_objective,
            'final_drawdown_pct': baseline_drawdown,
            'improvement_found': any(iter_data['accepted'] for iter_data in iterations)
        }
        
    except Exception as e:
        st.error(f"Universal optimization error: {e}")
        return None

def get_universal_ai_suggestion(ai_analyzer, strategy_config, current_params, current_perf, market_data):
    """
    Get AI suggestions for any strategy type using structured JSON output
    """
    import json
    
    try:
        # Get parameter bounds for the strategy type
        param_bounds = get_parameter_bounds(current_params, strategy_config['type'])
        
        # Create a universal prompt with parameter bounds
        strategy_description = f"""
Strategy Type: {strategy_config['type']}
Current Parameters and Allowed Ranges:
{format_params_with_bounds_for_ai(current_params, param_bounds, strategy_config['type'])}

Current Performance:
- Sortino Ratio: {current_perf.get('sortino_ratio', 0):.3f}
- Total Return: {current_perf.get('total_return', 0):.2f}%
- Max Drawdown: {current_perf.get('max_drawdown', 0):.2f}%
- Sharpe Ratio: {current_perf.get('sharpe_ratio', 0):.3f}
- Win Rate: {current_perf.get('win_rate', 0):.1f}%

Suggest ≤3 parameter improvements to increase Sortino ratio while controlling drawdown.
Stay within the allowed ranges shown above.
"""
        
        # JSON schema for structured output
        schema = """Return ONLY JSON in this format:
{"suggestions":[{"parameter":"<exact_parameter_name>","proposed":<numeric_value>,"why":"<brief_reason>"}]}"""
        
        # Use structured JSON mode for reliable parsing
        messages = [
            {
                "role": "system", 
                "content": "You optimize trading strategy parameters. Provide ≤3 specific changes. Stay within plausible bounds. Focus on risk-adjusted returns."
            },
            {
                "role": "user",
                "content": strategy_description + "\n\n" + schema
            }
        ]
        
        response = ai_analyzer.client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=messages,
            temperature=0.1,
            max_tokens=800,
            response_format={"type": "json_object"}
        )
        
        ai_response_text = response.choices[0].message.content
        
        try:
            # Parse structured JSON response
            data = json.loads(ai_response_text)
            parameter_suggestions = []
            
            for s in data.get("suggestions", []):
                if "parameter" in s and "proposed" in s:
                    parameter_suggestions.append({
                        "parameter": s["parameter"],
                        "suggested_value": s["proposed"],
                        "reasoning": s.get("why", "AI optimization suggestion")
                    })
            
            return {
                'ok': True,
                'parameter_suggestions': parameter_suggestions,
                'reasoning': ai_response_text,
                'raw_response': ai_response_text,
                'structured_data': data
            }
            
        except json.JSONDecodeError:
            # Fallback to regex parsing if JSON fails
            st.warning("⚠️ JSON parsing failed, using fallback text parsing")
            parameter_suggestions = parse_ai_parameter_suggestions(ai_response_text, strategy_config['type'])
            
            return {
                'ok': True,
                'parameter_suggestions': parameter_suggestions,
                'reasoning': ai_response_text,
                'raw_response': ai_response_text,
                'fallback_used': True
            }
        
    except Exception as e:
        st.error(f"Error getting AI suggestion: {e}")
        return {'ok': False, 'parameter_suggestions': [], 'reasoning': str(e)}

def get_parameter_bounds(current_params, strategy_type):
    """Get parameter bounds for a strategy type"""
    bounds = {}
    
    if strategy_type == "Momentum":
        bounds = {
            'momentum_period': [5, 60],
            'momentum_threshold': [0.1, 5.0],
            'vol_mult': [1.0, 3.0],
            'max_pos_pct': [10, 100],
            'stop_loss': [1, 20],
            'take_profit': [2, 40]
        }
    elif strategy_type == "Mean Reversion":
        bounds = {
            'rsi_period': [5, 30],
            'rsi_lo': [5, 40],     # Consistent naming (was rsi_oversold)
            'rsi_hi': [60, 95],    # Consistent naming (was rsi_overbought)
            'bb_period': [10, 40],
            'bb_std': [1.0, 3.5],
            'max_pos_pct': [10, 100],
            'stop_loss': [1, 20],
            'take_profit': [2, 40]
        }
    elif strategy_type == "Breakout":
        bounds = {
            'brk_period': [10, 100],
            'vol_mult': [1.0, 3.0],
            'max_pos_pct': [10, 100],
            'stop_loss': [1, 20],
            'take_profit': [2, 40]
        }
    
    return bounds

def format_params_with_bounds_for_ai(params, bounds, strategy_type):
    """Format parameters with bounds for AI prompt"""
    formatted = []
    
    for param, value in params.items():
        if param in bounds:
            min_val, max_val = bounds[param]
            formatted.append(f"- {param}: {value} (range: {min_val}-{max_val})")
        else:
            formatted.append(f"- {param}: {value}")
    
    return "\n".join(formatted)

def format_params_for_ai(params, strategy_type):
    """Format parameters in a human-readable way for AI"""
    if strategy_type == "Momentum":
        return f"""
- Momentum Period: {params.get('momentum_period', 20)} days
- Momentum Threshold: {params.get('momentum_threshold', 1.0)}%
- Volume Multiplier: {params.get('vol_mult', 1.2)}x
- Position Size: {params.get('max_pos_pct', 50)}%
- Stop Loss: {params.get('stop_loss', 3)}%
- Take Profit: {params.get('take_profit', 6)}%
"""
    elif strategy_type == "Mean Reversion":
        return f"""
- RSI Period: {params.get('rsi_period', 14)}
- RSI Oversold: {params.get('rsi_lo', 30)}
- RSI Overbought: {params.get('rsi_hi', 70)}
- Bollinger Band Period: {params.get('bb_period', 20)}
- Bollinger Band Std Dev: {params.get('bb_std', 2.0)}
- Position Size: {params.get('max_pos_pct', 50)}%
- Stop Loss: {params.get('stop_loss', 3)}%
- Take Profit: {params.get('take_profit', 6)}%
"""
    elif strategy_type == "Breakout":
        return f"""
- Breakout Lookback: {params.get('brk_period', 20)} days
- Volume Multiplier: {params.get('vol_mult', 1.5)}x
- Position Size: {params.get('max_pos_pct', 50)}%
- Stop Loss: {params.get('stop_loss', 3)}%
- Take Profit: {params.get('take_profit', 6)}%
"""
    else:
        return str(params)

def parse_ai_parameter_suggestions(ai_response, strategy_type):
    """Parse AI response to extract parameter suggestions"""
    suggestions = []
    
    # Look for parameter suggestions in the AI response
    import re
    
    if strategy_type == "Momentum":
        # Look for momentum-specific parameters
        period_match = re.search(r'momentum.*?period.*?(\d+)', ai_response.lower())
        threshold_match = re.search(r'threshold.*?(\d+\.?\d*)', ai_response.lower())
        volume_match = re.search(r'volume.*?(\d+\.?\d*)', ai_response.lower())
        
        if period_match:
            suggestions.append({
                'parameter': 'momentum_period',
                'suggested_value': int(period_match.group(1)),
                'reasoning': 'AI suggested momentum period adjustment'
            })
        
        if threshold_match:
            suggestions.append({
                'parameter': 'momentum_threshold',
                'suggested_value': float(threshold_match.group(1)),
                'reasoning': 'AI suggested threshold adjustment'
            })
            
        if volume_match:
            suggestions.append({
                'parameter': 'vol_mult',
                'suggested_value': float(volume_match.group(1)),
                'reasoning': 'AI suggested volume filter adjustment'
            })
    
    elif strategy_type == "Mean Reversion":
        # Look for mean reversion parameters
        rsi_period_match = re.search(r'rsi.*?period.*?(\d+)', ai_response.lower())
        oversold_match = re.search(r'oversold.*?(\d+)', ai_response.lower())
        overbought_match = re.search(r'overbought.*?(\d+)', ai_response.lower())
        bb_period_match = re.search(r'bollinger.*?period.*?(\d+)', ai_response.lower())
        
        if rsi_period_match:
            suggestions.append({
                'parameter': 'rsi_period',
                'suggested_value': int(rsi_period_match.group(1)),
                'reasoning': 'AI suggested RSI period adjustment'
            })
        
        if oversold_match:
            suggestions.append({
                'parameter': 'rsi_lo',
                'suggested_value': int(oversold_match.group(1)),
                'reasoning': 'AI suggested oversold threshold adjustment'
            })
            
        if overbought_match:
            suggestions.append({
                'parameter': 'rsi_hi',
                'suggested_value': int(overbought_match.group(1)),
                'reasoning': 'AI suggested overbought threshold adjustment'
            })
    
    elif strategy_type == "Breakout":
        # Look for breakout parameters
        lookback_match = re.search(r'lookback.*?(\d+)', ai_response.lower())
        volume_match = re.search(r'volume.*?(\d+\.?\d*)', ai_response.lower())
        
        if lookback_match:
            suggestions.append({
                'parameter': 'brk_period',
                'suggested_value': int(lookback_match.group(1)),
                'reasoning': 'AI suggested lookback period adjustment'
            })
            
        if volume_match:
            suggestions.append({
                'parameter': 'vol_mult',
                'suggested_value': float(volume_match.group(1)),
                'reasoning': 'AI suggested volume filter adjustment'
            })
    
    # If no specific suggestions found, create some smart defaults based on performance
    if not suggestions:
        suggestions = create_smart_default_suggestions(strategy_type)
    
    return suggestions

def create_smart_default_suggestions(strategy_type):
    """Create intelligent default suggestions when AI doesn't provide specific ones"""
    suggestions = []
    
    if strategy_type == "Momentum":
        suggestions = [
            {'parameter': 'momentum_period', 'suggested_value': 25, 'reasoning': 'Increase period for smoother signals'},
            {'parameter': 'momentum_threshold', 'suggested_value': 1.5, 'reasoning': 'Increase threshold to reduce noise'}
        ]
    elif strategy_type == "Mean Reversion":
        suggestions = [
            {'parameter': 'rsi_period', 'suggested_value': 16, 'reasoning': 'Adjust RSI period for better sensitivity'},
            {'parameter': 'rsi_lo', 'suggested_value': 25, 'reasoning': 'Lower oversold threshold for earlier signals'}
        ]
    elif strategy_type == "Breakout":
        suggestions = [
            {'parameter': 'brk_period', 'suggested_value': 25, 'reasoning': 'Increase lookback for stronger breakouts'},
            {'parameter': 'vol_mult', 'suggested_value': 1.8, 'reasoning': 'Higher volume filter for quality signals'}
        ]
    
    return suggestions

def apply_universal_changes(current_params, suggestions, strategy_type):
    """Apply parameter suggestions with consistent bounds checking"""
    new_params = current_params.copy()
    bounds = get_parameter_bounds(current_params, strategy_type)
    
    for suggestion in suggestions:
        param = suggestion['parameter']
        new_value = suggestion['suggested_value']
        
        # Apply bounds checking using the same bounds we showed the AI
        if param in bounds:
            min_val, max_val = bounds[param]
            if isinstance(new_value, (int, float)):
                if param in ['momentum_period', 'rsi_period', 'bb_period', 'brk_period', 'rsi_lo', 'rsi_hi', 'max_pos_pct', 'stop_loss', 'take_profit']:
                    new_value = int(max(min_val, min(max_val, new_value)))
                else:
                    new_value = float(max(min_val, min(max_val, new_value)))
                
                new_params[param] = new_value
            else:
                st.warning(f"⚠️ Invalid value type for {param}: {new_value}")
        else:
            st.warning(f"⚠️ Unknown parameter: {param}")
    
    return new_params

def run_actual_backtest(params, market_data):
    """Run actual backtest using the current strategy with given parameters"""
    try:
        # Use the global variables from sidebar
        strategy_type = params.get('strategy_type', 'Momentum')
        
        # Generate signals based on strategy type and parameters
        if strategy_type == "Momentum":
            signals = signal_momentum(
                market_data,
                period=int(params.get('momentum_period', 20)),
                threshold_pct=float(params.get('momentum_threshold', 1.0)),
                vol_mult=float(params.get('vol_mult', 1.2))
            )
        elif strategy_type == "Mean Reversion":
            signals = signal_mean_rev(
                market_data,
                rsi_period=int(params.get('rsi_period', 14)),
                rsi_lo=int(params.get('rsi_lo', 30)),
                rsi_hi=int(params.get('rsi_hi', 70)),
                bb_period=int(params.get('bb_period', 20)),
                bb_std=float(params.get('bb_std', 2.0))
            )
        elif strategy_type == "Breakout":
            signals = signal_breakout(
                market_data,
                lookback=int(params.get('brk_period', 20)),
                vol_mult=float(params.get('vol_mult', 1.5))
            )
        else:
            # Default to hold signals
            signals = pd.Series("HOLD", index=market_data.index)
        
        # Run backtest
        bt_df, metrics = run_backtest(
            market_data,
            signals,
            stop_loss_pct=float(params.get('stop_loss', 3))/100.0,
            take_profit_pct=float(params.get('take_profit', 6))/100.0,
            max_pos_frac=float(params.get('max_pos_pct', 50))/100.0
        )
        
        return metrics
        
    except Exception as e:
        st.error(f"Backtest error: {e}")
        # Return fallback metrics
        return {
            'total_return': 0,
            'sharpe_ratio': 0,
            'sortino_ratio': 0,
            'max_drawdown': 100,
            'profit_factor': 1,
            'win_rate': 50,
            'turnover': 1,
            'total_trades': 0
        }

# ===================== UI: sidebar =====================

with st.sidebar:
    st.header("⚙️ Strategy Configuration")
    strategy_name = st.text_input("Strategy Name", value="My Strategy")
    strategy_type = st.selectbox("Strategy Type", ["Momentum", "Mean Reversion", "Breakout", "Custom"])
    timeframe = st.selectbox("Timeframe", ["1day", "4hour", "1hour", "15min"], index=0)  # display only; backtest is daily here
    
    # General lookback window (used as default for strategy-specific periods)
    default_lookback = st.slider("Default Lookback Window", 5, 100, 20, help="Default period for strategy calculations")
    
    seed = st.number_input("Random Seed (for demo data)", value=42, step=1)

    st.subheader("⚠️ Risk Management")
    max_pos_pct = st.slider("Max Position Size (%)", 1, 100, 50)
    stop_loss = st.slider("Stop Loss (%)", 1, 20, 3)
    take_profit = st.slider("Take Profit (%)", 2, 40, 6)
    
    # Add basic cost/slippage
    cost_bps = st.slider("Transaction Costs (bps)", 0.0, 50.0, 0.0, 0.5, help="Cost per trade in basis points")

    # strategy-specific (use default_lookback as starting point)
    if strategy_type == "Momentum":
        momentum_period = st.slider("Momentum Period (days)", 5, 60, max(5, min(60, default_lookback)))
        momentum_threshold = st.slider("Momentum Threshold (%)", 0.1, 5.0, 1.0)
        vol_mult = st.slider("Volume Filter ×", 1.0, 3.0, 1.2, 0.1)
    elif strategy_type == "Mean Reversion":
        rsi_period = st.slider("RSI Period", 5, 30, max(5, min(30, default_lookback)))
        rsi_lo = st.slider("RSI Oversold", 5, 40, 30)
        rsi_hi = st.slider("RSI Overbought", 60, 95, 70)
        bb_period = st.slider("BB Period", 10, 40, max(10, min(40, default_lookback)))
        bb_std = st.slider("BB Std Dev", 1.0, 3.5, 2.0, 0.1)
    elif strategy_type == "Breakout":
        brk_period = st.slider("Breakout Lookback", 10, 100, max(10, min(100, default_lookback)))
        vol_mult = st.slider("Volume Filter ×", 1.0, 3.0, 1.5, 0.1)

# ===================== Data + signals =====================

data = simulate_prices(n_days=300, seed=int(seed))
if strategy_type == "Momentum":
    signals = signal_momentum(data, momentum_period, momentum_threshold, vol_mult)
elif strategy_type == "Mean Reversion":
    signals = signal_mean_rev(data, rsi_period, rsi_lo, rsi_hi, bb_period, bb_std)
elif strategy_type == "Breakout":
    signals = signal_breakout(data, brk_period, vol_mult)
else:
    signals = pd.Series("HOLD", index=data.index)

# ===================== Backtest =====================

bt_df, metrics = run_backtest(
    data,
    signals,
    stop_loss_pct=stop_loss/100.0,
    take_profit_pct=take_profit/100.0,
    max_pos_frac=max_pos_pct/100.0
)

# ===================== Layout: charts & metrics =====================

col1, col2 = st.columns([2, 1], gap="large")

with col1:
    st.subheader("📈 Backtest Results")
    fig = go.Figure()
    # Portfolio
    fig.add_trace(go.Scatter(
        x=bt_df["Date"], y=bt_df["Portfolio_Value"], name="Portfolio Value", mode="lines", line=dict(width=3)
    ))
    # Price (secondary)
    fig.add_trace(go.Scatter(
        x=bt_df["Date"], y=bt_df["Close"], name="Price", mode="lines", yaxis="y2", line=dict(width=1)
    ))
    # markers
    buys = bt_df[bt_df["Signal"] == "BUY"]
    sells = bt_df[bt_df["Signal"] == "SELL"]
    fig.add_trace(go.Scatter(
        x=buys["Date"], y=buys["Close"], mode="markers", name="BUY", yaxis="y2",
        marker=dict(color="green", size=8, symbol="triangle-up")
    ))
    fig.add_trace(go.Scatter(
        x=sells["Date"], y=sells["Close"], mode="markers", name="SELL", yaxis="y2",
        marker=dict(color="red", size=8, symbol="triangle-down")
    ))

    fig.update_layout(
        xaxis=dict(domain=[0.0, 1.0]),
        yaxis=dict(title="Portfolio (₹)"),
        yaxis2=dict(title="Price (₹)", overlaying="y", side="right"),
        height=520, margin=dict(l=10, r=10, t=10, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0)
    )
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("Show Signal & Equity Data"):
        st.dataframe(bt_df.tail(200), use_container_width=True)

with col2:
    st.subheader("📊 Performance Metrics")
    m = metrics
    st.metric("Total Return", f"{m['total_return']:.2f}%")
    st.metric("CAGR", f"{m['cagr']:.2f}%")
    st.metric("Max Drawdown", f"{m['max_drawdown']:.2f}%")
    st.metric("Sharpe", f"{m['sharpe_ratio']:.2f}")
    st.metric("Sortino", f"{m['sortino_ratio']:.2f}")
    st.metric("Profit Factor", f"{m['profit_factor']:.2f}")
    st.metric("Win Rate", f"{m['win_rate']:.1f}%")
    st.metric("Turnover (×/yr)", f"{m['turnover']:.2f}")

    # Signal distribution
    sig_counts = bt_df["Signal"].value_counts()
    fig_pie = px.pie(values=sig_counts.values, names=sig_counts.index, title="Signal Distribution")
    fig_pie.update_layout(height=300, margin=dict(l=10, r=10, t=40, b=10))
    st.plotly_chart(fig_pie, use_container_width=True)

    # Actions
    st.subheader("🎯 Actions")
    # export a "manifest" for your GroqAnalyzer
    manifest = {
        "name": strategy_name,
        "type": strategy_type,
        "timeframe": timeframe,
        "knobs": (
            {
                "momentum_period": [max(5, int(momentum_period*0.7)), int(momentum_period*1.3)],
                "momentum_threshold": [round(momentum_threshold*0.7, 3), round(momentum_threshold*1.3, 3)],
                "vol_mult": [max(1.0, round(vol_mult*0.8, 2)), round(vol_mult*1.2, 2)],
            } if strategy_type == "Momentum" else
            {
                "rsi_period": [max(5, int(rsi_period*0.7)), int(rsi_period*1.3)],
                "rsi_lo": [max(5, int(rsi_lo*0.8)), int(rsi_lo*1.1)],     # Consistent naming
                "rsi_hi": [int(rsi_hi*0.9), min(95, int(rsi_hi*1.05))],   # Consistent naming
                "bb_period": [max(10, int(bb_period*0.8)), int(bb_period*1.2)],
                "bb_std": [round(max(1.0, bb_std*0.8),2), round(bb_std*1.2,2)],
            } if strategy_type == "Mean Reversion" else
            {
                "brk_period": [max(10, int(brk_period*0.8)), int(brk_period*1.2)],
                "vol_mult": [max(1.0, round(vol_mult*0.8,2)), round(vol_mult*1.2,2)],
            }
        ),
        "invariants": ["no_lookahead", "apply_costs_externally", "close_to_close_returns"],
        "performance": m
    }
    
    # Use proper JSON encoding instead of pd.Series().to_json()
    import json
    st.download_button(
        "📦 Export Manifest JSON", 
        data=json.dumps(manifest, separators=(",", ":")),
        file_name="strategy_manifest.json", 
        mime="application/json", 
        use_container_width=True
    )

    # Universal AI Optimization System - Works with any strategy type
    if st.button("🤖 AI Optimize (Universal)", use_container_width=True):
        try:
            from src.analysis.ai_analyzer import GroqAnalyzer
            from src.utils.config import load_config
            
            # Load configuration
            config = load_config()
            api_key = config.get("groq_api_key")
            
            if not api_key:
                st.error("⚠️ GROQ_API_KEY not found in configuration. Please set it in .env file.")
            else:
                with st.spinner("🔄 Starting Universal AI Optimization..."):
                    # Initialize AI analyzer
                    if 'ai_analyzer' not in st.session_state:
                        st.session_state.ai_analyzer = GroqAnalyzer(api_key=api_key)
                    
                    # Create universal strategy configuration
                    strategy_config = {
                        "name": strategy_name,
                        "type": strategy_type,
                        "description": f"{strategy_type} strategy with universal AI optimization",
                        "timeframe": timeframe,
                        "objective": "maximize_sortino"
                    }
                    
                    # Collect current parameters from sidebar
                    current_params = {
                        "strategy_type": strategy_type,
                        "max_pos_pct": max_pos_pct,
                        "stop_loss": stop_loss,
                        "take_profit": take_profit
                    }
                    
                    # Add strategy-specific parameters
                    if strategy_type == "Momentum":
                        current_params.update({
                            "momentum_period": momentum_period,
                            "momentum_threshold": momentum_threshold,
                            "vol_mult": vol_mult
                        })
                    elif strategy_type == "Mean Reversion":
                        current_params.update({
                            "rsi_period": rsi_period,
                            "rsi_lo": rsi_lo,
                            "rsi_hi": rsi_hi,
                            "bb_period": bb_period,
                            "bb_std": bb_std
                        })
                    elif strategy_type == "Breakout":
                        current_params.update({
                            "brk_period": brk_period,
                            "vol_mult": vol_mult
                        })
                    
                    st.info(f"🎯 **Optimizing {strategy_type} Strategy** with parameters: {current_params}")
                    
                    # Optimization controls
                    col_opt1, col_opt2 = st.columns(2)
                    with col_opt1:
                        min_gain = st.selectbox("Min Improvement Required", [2.0, 5.0, 10.0, 15.0], index=1, help="Lower = more experimental changes")
                    with col_opt2:
                        max_dd_tolerance = st.selectbox("Max Drawdown Tolerance", [2.0, 3.0, 5.0, 8.0], index=1, help="Higher = riskier improvements allowed")
                    
                    st.markdown(f"**🎚️ Optimization Settings:** Requiring **{min_gain}%** improvement with max **{max_dd_tolerance}pp** drawdown increase")
                    
                    # Run universal optimization
                    optimization_result = universal_iterate_optimization(
                        ai_analyzer=st.session_state.ai_analyzer,
                        strategy_config=strategy_config,
                        current_params=current_params,
                        market_data=data,
                        max_iterations=5,
                        min_gain_threshold=min_gain,
                        max_dd_tolerance=max_dd_tolerance
                    )
                    
                    if optimization_result and optimization_result.get('improvement_found'):
                        st.success("🎉 **Universal AI Optimization Completed with Improvements!**")
                        
                        # Display comprehensive results
                        with st.expander("📊 Optimization Results", expanded=True):
                            col_a, col_b = st.columns(2)
                            
                            with col_a:
                                st.markdown("**📈 Original Strategy:**")
                                original_perf = run_actual_backtest(current_params, data)
                                st.metric("Sortino Ratio", f"{original_perf['sortino_ratio']:.3f}")
                                st.metric("Total Return", f"{original_perf['total_return']:.2f}%")
                                st.metric("Max Drawdown", f"{original_perf['max_drawdown']:.2f}%")
                                st.metric("Win Rate", f"{original_perf['win_rate']:.1f}%")
                            
                            with col_b:
                                st.markdown("**🚀 Optimized Strategy:**")
                                final_perf = optimization_result.get('champion_perf', {})
                                improvement = ((final_perf.get('sortino_ratio', 0) - original_perf['sortino_ratio']) / original_perf['sortino_ratio'] * 100) if original_perf['sortino_ratio'] > 0 else 0
                                st.metric("Sortino Ratio", f"{final_perf.get('sortino_ratio', 0):.3f}", f"+{improvement:.1f}%")
                                st.metric("Total Return", f"{final_perf.get('total_return', 0):.2f}%")
                                st.metric("Max Drawdown", f"{final_perf.get('max_drawdown', 0):.2f}%")
                                st.metric("Win Rate", f"{final_perf.get('win_rate', 0):.1f}%")
                        
                        with st.expander("🔧 Optimized Parameters", expanded=True):
                            st.markdown("**🎯 Recommended Parameter Changes:**")
                            
                            champion_params = optimization_result.get('champion_params', {})
                            
                            if strategy_type == "Momentum":
                                col1, col2, col3 = st.columns(3)
                                with col1:
                                    old_period = current_params.get('momentum_period', 20)
                                    new_period = champion_params.get('momentum_period', old_period)
                                    change = ((new_period - old_period) / old_period * 100) if old_period != 0 else 0
                                    st.metric("Momentum Period", f"{new_period} days", f"{change:+.1f}%")
                                
                                with col2:
                                    old_threshold = current_params.get('momentum_threshold', 1.0)
                                    new_threshold = champion_params.get('momentum_threshold', old_threshold)
                                    change = ((new_threshold - old_threshold) / old_threshold * 100) if old_threshold != 0 else 0
                                    st.metric("Threshold", f"{new_threshold:.2f}%", f"{change:+.1f}%")
                                
                                with col3:
                                    old_vol = current_params.get('vol_mult', 1.2)
                                    new_vol = champion_params.get('vol_mult', old_vol)
                                    change = ((new_vol - old_vol) / old_vol * 100) if old_vol != 0 else 0
                                    st.metric("Volume Filter", f"{new_vol:.2f}x", f"{change:+.1f}%")
                            
                            elif strategy_type == "Mean Reversion":
                                col1, col2, col3 = st.columns(3)
                                with col1:
                                    old_rsi = current_params.get('rsi_period', 14)
                                    new_rsi = champion_params.get('rsi_period', old_rsi)
                                    change = ((new_rsi - old_rsi) / old_rsi * 100) if old_rsi != 0 else 0
                                    st.metric("RSI Period", f"{new_rsi}", f"{change:+.1f}%")
                                
                                with col2:
                                    old_oversold = current_params.get('rsi_lo', 30)
                                    new_oversold = champion_params.get('rsi_lo', old_oversold)
                                    st.metric("RSI Oversold", f"{new_oversold}")
                                
                                with col3:
                                    old_overbought = current_params.get('rsi_hi', 70)
                                    new_overbought = champion_params.get('rsi_hi', old_overbought)
                                    st.metric("RSI Overbought", f"{new_overbought}")
                            
                            elif strategy_type == "Breakout":
                                col1, col2 = st.columns(2)
                                with col1:
                                    old_period = current_params.get('brk_period', 20)
                                    new_period = champion_params.get('brk_period', old_period)
                                    change = ((new_period - old_period) / old_period * 100) if old_period != 0 else 0
                                    st.metric("Lookback Period", f"{new_period} days", f"{change:+.1f}%")
                                
                                with col2:
                                    old_vol = current_params.get('vol_mult', 1.5)
                                    new_vol = champion_params.get('vol_mult', old_vol)
                                    change = ((new_vol - old_vol) / old_vol * 100) if old_vol != 0 else 0
                                    st.metric("Volume Filter", f"{new_vol:.2f}x", f"{change:+.1f}%")
                            
                            # Risk management parameters
                            st.markdown("**⚠️ Risk Management:**")
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                old_pos = current_params.get('max_pos_pct', 50)
                                new_pos = champion_params.get('max_pos_pct', old_pos)
                                st.metric("Position Size", f"{new_pos}%")
                            with col2:
                                old_sl = current_params.get('stop_loss', 3)
                                new_sl = champion_params.get('stop_loss', old_sl)
                                st.metric("Stop Loss", f"{new_sl}%")
                            with col3:
                                old_tp = current_params.get('take_profit', 6)
                                new_tp = champion_params.get('take_profit', old_tp)
                                st.metric("Take Profit", f"{new_tp}%")
                                
                            st.success("💡 **How to Apply**: Adjust the sliders in the sidebar to match these optimized values!")
                        
                        with st.expander("📈 Optimization Journey", expanded=True):
                            iterations = optimization_result.get('iterations', [])
                            st.markdown(f"**🎯 Total Iterations:** {optimization_result.get('total_iterations', 0)}")
                            
                            if iterations:
                                # Show detailed journey with deltas for both accepted and rejected
                                journey_df = pd.DataFrame([
                                    {
                                        "Iteration": iter_data.get('iteration', i+1),
                                        "Sortino": f"{iter_data.get('objective', 0):.3f}",
                                        "Sortino Δ": f"{iter_data.get('objective_delta', 0):+.3f}",
                                        "Return %": f"{iter_data.get('total_return', 0):.2f}%",
                                        "Drawdown %": f"{iter_data.get('max_drawdown', 0):.2f}%",
                                        "DD Δ": f"{iter_data.get('dd_worsening', 0):+.1f}pp",
                                        "Suggestions": iter_data.get('suggestions_applied', 0),
                                        "Status": "✅ Accepted" if iter_data.get('accepted', False) else "❌ Rejected"
                                    }
                                    for i, iter_data in enumerate(iterations)
                                ])
                                st.dataframe(journey_df, use_container_width=True)
                                
                                # Show AI reasoning for all iterations (not just accepted ones)
                                st.markdown("**🤖 AI Reasoning for All Iterations:**")
                                for i, iter_data in enumerate(iterations):
                                    status_emoji = "✅" if iter_data.get('accepted', False) else "❌"
                                    with st.expander(f"{status_emoji} Iteration {iter_data.get('iteration', i+1)} - AI Analysis"):
                                        st.markdown(f"**Suggestions Applied:** {iter_data.get('suggestions_applied', 0)}")
                                        st.markdown(f"**Objective Delta:** {iter_data.get('objective_delta', 0):+.3f}")
                                        st.markdown(f"**Reasoning:** {iter_data.get('ai_reasoning', 'No reasoning provided')}")
                                        if not iter_data.get('accepted', False):
                                            st.markdown("**Why Rejected:** Performance improvement below threshold or drawdown increased too much")
                    
                    else:
                        st.warning("🔍 **No Significant Improvements Found**")
                        st.info(f"""
                        **Analysis Results:**
                        - Completed {optimization_result.get('total_iterations', 0) if optimization_result else 0} optimization iterations
                        - Your current {strategy_type} strategy parameters appear to be well-tuned
                        
                        **💡 Suggestions:**
                        - Try lowering the improvement threshold (e.g., 2-5%)
                        - Experiment with different market data (change random seed)
                        - Consider testing other strategy types
                        - Manual parameter exploration might reveal hidden opportunities
                        """)
                        
                        # Show what was analyzed
                        if optimization_result:
                            with st.expander("🔍 What the AI Analyzed"):
                                st.markdown("**Current Strategy Configuration:**")
                                st.json(strategy_config)
                                st.markdown("**Current Parameters:**")
                                st.json(current_params)
                                
                                if optimization_result.get('iterations'):
                                    st.markdown("**AI Attempts:**")
                                    for i, iter_data in enumerate(optimization_result['iterations']):
                                        st.markdown(f"**Iteration {i+1}:** {iter_data.get('ai_reasoning', 'No reasoning')}")
                
        except Exception as e:
            st.error(f"❌ Universal AI optimization error: {str(e)}")
            st.error("Please ensure all dependencies are installed and configured correctly.")
