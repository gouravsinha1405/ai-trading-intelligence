import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# --- paths ---
ROOT = Path(__file__).resolve().parent
sys.path.append(str(ROOT.parent / "src"))

from data.jugaad_client import JugaadDataClient
from utils.mobile_ui import (
    inject_mobile_css,
    mobile_friendly_columns,
    responsive_chart_config,
)
from utils.config import load_config
from analysis.ai_analyzer import GroqAnalyzer

st.set_page_config(
    page_title="Strategy Builder",
    page_icon="🔧",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Inject mobile CSS
inject_mobile_css()

# ===================== Initialize AI Analyzer =====================
# Load config and initialize AI analyzer
if "ai_analyzer" not in st.session_state:
    try:
        config = load_config()
        api_key = config.get("groq_api_key")
        if api_key:
            st.session_state.ai_analyzer = GroqAnalyzer(api_key)
        else:
            st.session_state.ai_analyzer = None
    except Exception as e:
        st.session_state.ai_analyzer = None

# ===================== Main Page Header =====================
st.title("🔧 AI-Powered Strategy Builder")
st.markdown("""
**Build, optimize, and backtest trading strategies with AI assistance**
- 🎯 Choose from pre-built strategies or generate custom ones with AI  
- ⚡ Multi-objective AI optimization for better performance
- 📊 Real market data backtesting with comprehensive analytics
""")

st.divider()
st.caption("educational prototype • no investment advice • paper testing only")

# ===================== utils =====================


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    roll_up = up.ewm(alpha=1 / period, adjust=False).mean()
    roll_down = down.ewm(alpha=1 / period, adjust=False).mean()
    rs = roll_up / roll_down.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def bollinger_bands(series: pd.Series, period: int = 20, stds: float = 2.0):
    ma = series.rolling(period, min_periods=period).mean()
    sd = series.rolling(period, min_periods=period).std()
    return ma, ma + stds * sd, ma - stds * sd


def universal_iterate_optimization(
    ai_analyzer, strategy_config, current_params, market_data, 
    max_iterations=5, min_gain_threshold=10.0, max_dd_tolerance=20.0
):
    """Universal AI optimization that works with any strategy type"""
    try:
        best_result = None
        iteration_count = 0
        
        for iteration in range(max_iterations):
            iteration_count += 1
            
            # Create context for AI
            market_summary = f"""
            Data: {len(market_data)} periods
            Current Price: {market_data['Close'].iloc[-1]:.2f}
            Price Range: {market_data['Close'].min():.2f} - {market_data['Close'].max():.2f}
            Volatility: {market_data['Close'].pct_change().std() * 100:.2f}%
            """
            
            # Get AI suggestions
            prompt = f"""
            Strategy Type: {strategy_config['type']}
            Current Parameters: {current_params}
            Market Data: {market_summary}
            
            Optimize these parameters for better performance.
            Return only a JSON object with improved parameter values.
            """
            
            ai_response = ai_analyzer.analyze_strategy(
                strategy_data={"config": strategy_config, "params": current_params},
                context=prompt
            )
            
            if ai_response and "suggestions" in ai_response:
                suggested_params = ai_response["suggestions"]
                
                # Validate suggestions based on strategy type
                validated_params = validate_strategy_params(suggested_params, strategy_config['type'])
                
                if validated_params:
                    best_result = {
                        "champion_params": validated_params,
                        "iteration": iteration_count,
                        "improvement_found": True,
                        "ai_reasoning": ai_response.get("reasoning", "AI optimization completed"),
                        "suggestions": validated_params
                    }
                    break
        
        if not best_result:
            best_result = {
                "champion_params": current_params,
                "iteration": iteration_count,
                "improvement_found": False,
                "ai_reasoning": "No significant improvements found",
                "suggestions": current_params
            }
        
        return best_result
        
    except Exception as e:
        st.error(f"Universal optimization error: {str(e)}")
        return None


def validate_strategy_params(params, strategy_type):
    """Enhanced parameter validation with risk constraints"""
    try:
        validated = {}
        
        # Common parameters for all strategies
        if "max_pos_pct" in params:
            validated["max_pos_pct"] = max(1, min(100, float(params["max_pos_pct"])))
        if "stop_loss" in params:
            validated["stop_loss"] = max(1, min(50, float(params["stop_loss"])))
        if "take_profit" in params:
            validated["take_profit"] = max(1, min(100, float(params["take_profit"])))
        
        # Enhanced constraint: take_profit >= 1.2 * stop_loss
        if "take_profit" in validated and "stop_loss" in validated:
            min_tp = validated["stop_loss"] * 1.2
            if validated["take_profit"] < min_tp:
                validated["take_profit"] = min_tp
        
        # Risk per trade constraint: max_pos_pct * stop_loss <= 2.0%
        if "max_pos_pct" in validated and "stop_loss" in validated:
            risk_per_trade = (validated["max_pos_pct"] / 100) * (validated["stop_loss"] / 100) * 100
            if risk_per_trade > 2.0:
                # Scale down position size to maintain 2% risk
                validated["max_pos_pct"] = min(validated["max_pos_pct"], (2.0 / (validated["stop_loss"] / 100)))
        
        # Strategy-specific validation with bounds checking
        if strategy_type == "Momentum":
            if "momentum_period" in params:
                validated["momentum_period"] = max(5, min(100, int(params["momentum_period"])))
            if "momentum_threshold" in params:
                validated["momentum_threshold"] = max(0.1, min(10.0, float(params["momentum_threshold"])))
            if "vol_mult" in params:
                validated["vol_mult"] = max(0.5, min(5.0, float(params["vol_mult"])))
                
        elif strategy_type == "Mean Reversion":
            if "rsi_period" in params:
                validated["rsi_period"] = max(5, min(50, int(params["rsi_period"])))
            if "rsi_lo" in params:
                validated["rsi_lo"] = max(10, min(40, float(params["rsi_lo"])))
            if "rsi_hi" in params:
                validated["rsi_hi"] = max(60, min(90, float(params["rsi_hi"])))
            if "bb_period" in params:
                validated["bb_period"] = max(10, min(50, int(params["bb_period"])))
            if "bb_std" in params:
                validated["bb_std"] = max(1.0, min(3.0, float(params["bb_std"])))
                
        elif strategy_type == "Breakout":
            if "brk_period" in params:
                validated["brk_period"] = max(10, min(100, int(params["brk_period"])))
            if "vol_mult" in params:
                validated["vol_mult"] = max(0.5, min(5.0, float(params["vol_mult"])))
        
        return validated if validated else None
        
    except Exception as e:
        st.error(f"Parameter validation error: {str(e)}")
        return None


def validate_and_fix_ai_code(ai_analyzer, strategy_code, market_data_sample):
    """
    Use GroqAI to validate and fix variable alignment issues in generated strategy code
    """
    import json
    
    try:
        # Get actual data structure information
        data_info = {
            "columns": list(market_data_sample.columns),
            "data_types": {col: str(dtype) for col, dtype in market_data_sample.dtypes.items()},
            "sample_values": {col: market_data_sample[col].iloc[-1] if len(market_data_sample) > 0 else "N/A" 
                            for col in market_data_sample.columns},
            "index_type": str(type(market_data_sample.index)),
            "length": len(market_data_sample)
        }
        
        # Available functions in our environment
        available_functions = [
            "pd.Series", "pd.DataFrame", "np.sqrt", "np.mean", "np.std", "np.abs", "np.max", "np.min",
            ".rolling()", ".ewm()", ".pct_change()", ".shift()", ".diff()", ".cumsum()", ".fillna()",
            ".dropna()", ".clip()", ".replace()", ".where()", ".iloc", ".loc", ".index"
        ]
        
        validation_prompt = f"""You are a code validator for trading strategies. Your job is to check and fix variable alignment issues.

**Actual Data Structure:**
```
Columns: {data_info['columns']}
Data Types: {data_info['data_types']}
Sample Values: {data_info['sample_values']}
Index: {data_info['index_type']}
Length: {data_info['length']} rows
```

**Available Functions:**
{', '.join(available_functions)}

**Strategy Code to Validate:**
```python
{strategy_code}
```

**Common Issues to Fix:**
1. Column name mismatches (e.g., 'close' vs 'Close', 'volume' vs 'Volume')
2. Missing columns (add fallback logic)
3. Undefined functions (replace with available alternatives)
4. Index alignment issues
5. Data type mismatches

**Requirements:**
- All column references must match exactly: {data_info['columns']}
- Use .get() for optional columns with fallbacks
- Ensure pandas Series output with proper index
- Handle missing data gracefully
- Use only available functions

Return ONLY valid Python code that will work with the actual data structure."""

        messages = [
            {
                "role": "system", 
                "content": "You are an expert Python code validator specializing in trading strategy compatibility. Fix variable alignment issues and ensure code works with the actual data structure."
            },
            {"role": "user", "content": validation_prompt}
        ]

        response = ai_analyzer.client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=messages,
            temperature=0.1,
            max_tokens=2000,
        )

        fixed_code = response.choices[0].message.content
        
        # Clean up the response (remove markdown code blocks if present)
        if "```python" in fixed_code:
            fixed_code = fixed_code.split("```python")[1].split("```")[0]
        elif "```" in fixed_code:
            fixed_code = fixed_code.split("```")[1].split("```")[0]
            
        return {
            "ok": True,
            "fixed_code": fixed_code.strip(),
            "original_code": strategy_code,
            "data_info": data_info
        }
        
    except Exception as e:
        return {
            "ok": False,
            "error": str(e),
            "fixed_code": strategy_code,  # Return original as fallback
            "original_code": strategy_code
        }


def safe_execute_ai_strategy(ai_strategy, market_data, params, ai_analyzer=None):
    """
    Safely execute AI-generated strategy with validation and error handling
    """
    try:
        # Step 1: Validate and fix code using GroqAI if available
        original_code = ai_strategy["code"]
        fixed_code = original_code
        
        if ai_analyzer:
            validation_result = validate_and_fix_ai_code(ai_analyzer, original_code, market_data.head(10))
            if validation_result["ok"]:
                fixed_code = validation_result["fixed_code"]
                st.info("🔧 **Code Validation**: AI has optimized the strategy code for your data structure")
            else:
                st.warning(f"⚠️ **Code Validation Warning**: {validation_result.get('error', 'Unknown validation error')}")
        
        # Step 2: Create enhanced safe execution environment
        exec_globals = {
            'pd': pd,
            'np': np,
            'len': len, 'max': max, 'min': min, 'int': int, 'float': float,
            'range': range, 'enumerate': enumerate, 'abs': abs, 'round': round,
            'sum': sum, 'any': any, 'all': all,
            '__builtins__': {}  # Restrict built-ins for security
        }
        
        # Step 3: Execute the (potentially fixed) code
        exec(fixed_code, exec_globals)
        ai_strategy_func = exec_globals.get('ai_generated_strategy')
        
        if not ai_strategy_func:
            raise ValueError("Function 'ai_generated_strategy' not found in generated code")
        
        # Step 4: Prepare parameters with validation
        ai_params = {}
        if "params" in ai_strategy:
            for param, details in ai_strategy["params"].items():
                if isinstance(details, dict) and "default" in details:
                    ai_params[param] = details["default"]
                else:
                    ai_params[param] = details
        
        # Override with provided parameters
        for key, value in params.items():
            if key in ai_params:
                ai_params[key] = value
        
        # Step 5: Execute strategy function with error handling
        signals = ai_strategy_func(market_data, ai_params)
        
        # Step 6: Validate output format
        if not isinstance(signals, pd.Series):
            if hasattr(signals, '__iter__') and not isinstance(signals, str):
                signals = pd.Series(signals, index=market_data.index)
            else:
                raise ValueError(f"Strategy function returned invalid type: {type(signals)}")
        
        # Step 7: Validate signal values
        valid_signals = ['BUY', 'SELL', 'HOLD']
        invalid_signals = signals[~signals.isin(valid_signals)]
        if len(invalid_signals) > 0:
            st.warning(f"⚠️ **Signal Validation**: Found invalid signals, converting to HOLD: {invalid_signals.unique()}")
            signals = signals.where(signals.isin(valid_signals), 'HOLD')
        
        # Step 8: Ensure proper index alignment
        if not signals.index.equals(market_data.index):
            st.warning("⚠️ **Index Alignment**: Realigning strategy signals with market data")
            signals = signals.reindex(market_data.index, fill_value='HOLD')
        
        return {
            "ok": True,
            "signals": signals,
            "fixed_code": fixed_code,
            "original_code": original_code,
            "params_used": ai_params,
            "validation_applied": ai_analyzer is not None
        }
        
    except Exception as e:
        error_details = str(e)
        
        # Provide helpful error context
        if "KeyError" in error_details:
            missing_col = error_details.split("'")[1] if "'" in error_details else "unknown"
            error_context = f"Missing column '{missing_col}'. Available columns: {list(market_data.columns)}"
        elif "NameError" in error_details:
            undefined_var = error_details.split("'")[1] if "'" in error_details else "unknown"
            error_context = f"Undefined variable/function '{undefined_var}'. Check imports and function definitions."
        elif "AttributeError" in error_details:
            error_context = f"Method/attribute error: {error_details}"
        else:
            error_context = error_details
        
        return {
            "ok": False,
            "error": error_context,
            "signals": pd.Series("HOLD", index=market_data.index),  # Safe fallback
            "fixed_code": ai_strategy.get("code", ""),
            "original_code": ai_strategy.get("code", ""),
            "params_used": params
        }


# ===================== Enhanced Strategy Proposal System =====================


def calculate_composite_objective(trial_results, baseline_results, optimization_mode="balanced"):
    """
    Calculate composite objective score based on multiple criteria
    
    optimization_mode options:
    - "growth": Prioritize CAGR with quality constraints
    - "balanced": Balance growth and risk-adjusted returns  
    - "quality": Prioritize Sortino with growth secondary
    - "conservative": Focus on risk metrics
    """
    
    # Extract metrics
    trial_cagr = trial_results.get("cagr", 0)
    trial_sortino = trial_results.get("sortino_ratio", 0)
    trial_return = trial_results.get("total_return", 0)
    trial_dd = trial_results.get("max_drawdown", 100)
    trial_sharpe = trial_results.get("sharpe_ratio", 0)
    trial_profit_factor = trial_results.get("profit_factor", 1)
    
    baseline_cagr = baseline_results.get("cagr", 0)
    baseline_sortino = baseline_results.get("sortino_ratio", 0)
    baseline_return = baseline_results.get("total_return", 0)
    
    # Quality gates - must pass these first
    quality_passed = True
    quality_reasons = []
    
    # Gate 1: Profitability (unless baseline is also unprofitable)
    if trial_return < -0.5 and baseline_return > 0:
        quality_passed = False
        quality_reasons.append("Turns profitable strategy into losing one")
    
    # Gate 2: Risk-adjusted returns (unless baseline is also poor)
    if trial_sortino < 0.3 and baseline_sortino > 0.8:
        quality_passed = False
        quality_reasons.append("Severely degrades risk-adjusted returns")
    
    # Gate 3: Drawdown protection
    if trial_dd > 15.0:  # Absolute drawdown limit
        quality_passed = False
        quality_reasons.append(f"Excessive drawdown: {trial_dd:.1f}%")
    
    if not quality_passed:
        return {
            "score": -1000,  # Heavily penalize quality failures
            "passed_gates": False,
            "reasons": quality_reasons,
            "breakdown": {"quality_penalty": -1000}
        }
    
    # Calculate composite score based on optimization mode
    if optimization_mode == "growth":
        # Primary: CAGR, Constraints: Sortino ≥ 0.5, DD ≤ 12%
        primary_score = trial_cagr * 10  # Scale CAGR to be primary driver
        
        # Quality bonuses/penalties
        sortino_bonus = max(0, (trial_sortino - 0.5) * 50) if trial_sortino >= 0.5 else -100
        dd_penalty = max(0, (trial_dd - 12.0) * -20) if trial_dd > 12.0 else 0
        consistency_bonus = min(trial_profit_factor * 10, 30)  # Cap at 30
        
        score = primary_score + sortino_bonus + dd_penalty + consistency_bonus
        
        breakdown = {
            "primary_cagr": primary_score,
            "sortino_bonus": sortino_bonus,
            "dd_penalty": dd_penalty,
            "consistency_bonus": consistency_bonus
        }
        
    elif optimization_mode == "balanced":
        # Balance growth and quality
        cagr_component = trial_cagr * 6
        sortino_component = trial_sortino * 40
        dd_component = max(0, (15.0 - trial_dd) * 3)  # Reward low drawdown
        consistency_component = min(trial_profit_factor * 8, 25)
        
        score = cagr_component + sortino_component + dd_component + consistency_component
        
        breakdown = {
            "cagr_component": cagr_component,
            "sortino_component": sortino_component,
            "dd_component": dd_component,
            "consistency_component": consistency_component
        }
        
    elif optimization_mode == "quality":
        # Primary: Sortino, Secondary: CAGR
        primary_score = trial_sortino * 60
        secondary_score = trial_cagr * 3
        dd_bonus = max(0, (10.0 - trial_dd) * 2)
        
        score = primary_score + secondary_score + dd_bonus
        
        breakdown = {
            "primary_sortino": primary_score,
            "secondary_cagr": secondary_score,
            "dd_bonus": dd_bonus
        }
        
    else:  # conservative
        # Focus on risk metrics with modest growth
        risk_score = (trial_sortino * 40) + max(0, (8.0 - trial_dd) * 5)
        growth_score = min(trial_cagr * 2, 20)  # Cap growth component
        
        score = risk_score + growth_score
        
        breakdown = {
            "risk_score": risk_score,
            "growth_score": growth_score
        }
    
    return {
        "score": score,
        "passed_gates": True,
        "reasons": ["All quality gates passed"],
        "breakdown": breakdown,
        "mode": optimization_mode
    }


# ===================== Enhanced Strategy Proposal System =====================


def get_strategy_proposal_from_ai(ai_analyzer, symbol="RELIANCE", timeframe="1D", market_data=None):
    """
    Get a complete new strategy proposal from AI with code generation
    Enhanced with actual data structure awareness
    """
    import json
    
    try:
        # Market context for strategy selection
        regime_stats = calculate_regime_stats(market_data) if market_data is not None else {}
        
        # Get actual data structure for better code generation
        data_structure_info = ""
        if market_data is not None and len(market_data) > 0:
            data_structure_info = f"""
**Actual Data Structure Available:**
- Columns: {list(market_data.columns)}
- Data Types: {dict(market_data.dtypes)}
- Sample Data: {market_data.head(2).to_dict()}
- Index Type: {type(market_data.index)}
- Total Records: {len(market_data)}
"""
        
        prompt = f"""You are a quantitative trading strategist. Design a new trading strategy for {symbol} [{timeframe}].

{data_structure_info}

Market Context:
- Volatility: {regime_stats.get('vol_20d', 15):.1f}% (20-day annualized)
- Trend Strength: {regime_stats.get('adx', 30):.0f} ADX
- Autocorrelation: {regime_stats.get('autocorr1', 0):.3f}
- ATR: {regime_stats.get('atr_pct', 2):.2f}%

**CRITICAL REQUIREMENTS:**
1. Use EXACT column names from the actual data structure above
2. Handle missing columns gracefully with .get() or try/except
3. Use proper pandas syntax compatible with the data types shown
4. Return pandas Series with same index as input DataFrame
5. Use only 'BUY', 'SELL', 'HOLD' as signal values

Design a robust strategy that:
1. Uses common technical indicators (RSI, EMA, ATR, Bollinger Bands, etc.)
2. Has clear entry/exit rules
3. Includes position sizing based on volatility
4. Respects risk limits: risk_per_trade ≤ 2%
5. WORKS WITH THE ACTUAL DATA STRUCTURE SHOWN ABOVE

Return ONLY this JSON format:"""

        schema = """{
  "name": "Strategy_Name",
  "family": "mean_reversion|momentum|breakout|hybrid",
  "timeframe": "1D",
  "warmup_bars": 200,
  "params": {
    "param1": {"default": 14, "bounds": [10, 30]},
    "param2": {"default": 2.0, "bounds": [1.0, 3.0]},
    "risk_per_trade_pct": {"default": 1.0, "bounds": [0.5, 2.0]},
    "max_pos_pct": {"default": 20, "bounds": [10, 50]}
  },
  "math": {
    "indicators": [
      "RSI_t = RSI(Close, period)",
      "EMA_t = EMA(Close, period)"
    ],
    "rules": [
      "Entry: condition1 AND condition2",
      "Exit: condition3 OR condition4"
    ],
    "position_sizing": "size = min(max_pos_pct, risk_per_trade_pct / ATR_pct)"
  },
  "description": "Brief strategy description and market conditions where it works best",
  "data_compatibility_notes": "Any specific notes about data compatibility"
}"""

        messages = [
            {
                "role": "system",
                "content": "You are an expert quantitative strategist specializing in data-compatible trading strategies. Always use exact column names and handle data structure variations gracefully.",
            },
            {"role": "user", "content": prompt + "\n" + schema},
        ]

        response = ai_analyzer.client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=messages,
            temperature=0.2,
            max_tokens=1500,
            response_format={"type": "json_object"},
        )

        ai_response_text = response.choices[0].message.content
        
        try:
            proposal = json.loads(ai_response_text)
            return {
                "ok": True,
                "proposal": proposal,
                "raw_response": ai_response_text,
                "data_aware": True
            }
        except json.JSONDecodeError:
            return {
                "ok": False,
                "error": "Failed to parse strategy proposal JSON",
                "raw_response": ai_response_text
            }
            
    except Exception as e:
        return {
            "ok": False,
            "error": f"Strategy proposal error: {str(e)}"
        }


def generate_strategy_code(proposal):
    """
    Generate executable Python code from strategy proposal
    """
    try:
        if not proposal.get("ok", False):
            return None
            
        strategy_data = proposal["proposal"]
        strategy_name = strategy_data.get("name", "Custom_Strategy")
        family = strategy_data.get("family", "hybrid")
        params = strategy_data.get("params", {})
        
        # Generate basic strategy template based on family
        if family == "momentum":
            code_template = generate_momentum_template(strategy_data)
        elif family == "mean_reversion":
            code_template = generate_mean_reversion_template(strategy_data)
        elif family == "breakout":
            code_template = generate_breakout_template(strategy_data)
        else:  # hybrid
            code_template = generate_hybrid_template(strategy_data)
        
        return {
            "name": strategy_name,
            "code": code_template,
            "params": params,
            "description": strategy_data.get("description", "AI-generated strategy")
        }
        
    except Exception as e:
        st.error(f"Code generation error: {str(e)}")
        return None


def generate_momentum_template(strategy_data):
    """Generate momentum strategy template"""
    return f'''
def ai_generated_strategy(df, params):
    """
    AI-Generated Momentum Strategy: {strategy_data.get("name", "Momentum")}
    {strategy_data.get("description", "")}
    """
    default_params = {{{", ".join([f'"{k}": {v.get("default", 1)}' for k, v in strategy_data.get("params", {}).items()])}}}
    p = {{**default_params, **params}}
    
    close = df['Close']
    returns = close.pct_change()
    momentum = returns.rolling(int(p.get("momentum_period", 20))).sum() * 100
    
    # Volume confirmation
    volume = df.get('Volume', pd.Series(1, index=df.index))
    vol_ma = volume.rolling(20).mean()
    vol_ok = volume > vol_ma * p.get("vol_mult", 1.2)
    
    # Signals
    buy = (momentum > p.get("momentum_threshold", 1.0)) & vol_ok
    sell = (momentum < -p.get("momentum_threshold", 1.0)) & vol_ok
    
    signals = pd.Series("HOLD", index=df.index)
    signals[buy] = "BUY"
    signals[sell] = "SELL"
    
    return signals
'''


def generate_mean_reversion_template(strategy_data):
    """Generate mean reversion strategy template"""
    return f'''
def ai_generated_strategy(df, params):
    """
    AI-Generated Mean Reversion Strategy: {strategy_data.get("name", "MeanRev")}
    {strategy_data.get("description", "")}
    """
    default_params = {{{", ".join([f'"{k}": {v.get("default", 1)}' for k, v in strategy_data.get("params", {}).items()])}}}
    p = {{**default_params, **params}}
    
    close = df['Close']
    
    # RSI calculation
    delta = close.diff()
    gain = (delta.where(delta > 0, 0)).rolling(int(p.get("rsi_period", 14))).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(int(p.get("rsi_period", 14))).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    
    # Bollinger Bands
    bb_period = int(p.get("bb_period", 20))
    bb_std = p.get("bb_std", 2.0)
    bb_mid = close.rolling(bb_period).mean()
    bb_std_dev = close.rolling(bb_period).std()
    bb_upper = bb_mid + (bb_std_dev * bb_std)
    bb_lower = bb_mid - (bb_std_dev * bb_std)
    
    # Mean reversion signals
    buy = (rsi < p.get("rsi_lo", 30)) & (close < bb_lower)
    sell = (rsi > p.get("rsi_hi", 70)) & (close > bb_upper)
    
    signals = pd.Series("HOLD", index=df.index)
    signals[buy] = "BUY"
    signals[sell] = "SELL"
    
    return signals
'''


def generate_breakout_template(strategy_data):
    """Generate breakout strategy template"""
    return f'''
def ai_generated_strategy(df, params):
    """
    AI-Generated Breakout Strategy: {strategy_data.get("name", "Breakout")}
    {strategy_data.get("description", "")}
    """
    default_params = {{{", ".join([f'"{k}": {v.get("default", 1)}' for k, v in strategy_data.get("params", {}).items()])}}}
    p = {{**default_params, **params}}
    
    close = df['Close']
    high = df.get('High', close)
    low = df.get('Low', close)
    volume = df.get('Volume', pd.Series(1, index=df.index))
    
    # Breakout levels
    lookback = int(p.get("brk_period", 20))
    resistance = high.rolling(lookback).max()
    support = low.rolling(lookback).min()
    
    # Volume confirmation
    vol_ma = volume.rolling(20).mean()
    vol_ok = volume > vol_ma * p.get("vol_mult", 1.5)
    
    # Breakout signals
    buy = (close > resistance.shift(1)) & vol_ok
    sell = (close < support.shift(1)) & vol_ok
    
    signals = pd.Series("HOLD", index=df.index)
    signals[buy] = "BUY"
    signals[sell] = "SELL"
    
    return signals
'''


def generate_hybrid_template(strategy_data):
    """Generate a hybrid strategy template (most flexible)"""
    return f'''
def ai_generated_strategy(df, params):
    """
    AI-Generated Strategy: {strategy_data.get("name", "Hybrid")}
    {strategy_data.get("description", "")}
    """
    # Default parameters
    default_params = {{{", ".join([f'"{k}": {v.get("default", 1)}' for k, v in strategy_data.get("params", {}).items()])}}}
    p = {{**default_params, **params}}
    
    # Technical indicators
    close = df['Close']
    rsi = close.rolling(int(p.get("rsi_period", 14))).apply(lambda x: 100 - 100/(1 + x.pct_change().clip(lower=0).mean() / (-x.pct_change().clip(upper=0).mean())), raw=False)
    ema_fast = close.ewm(span=int(p.get("ema_fast", 12))).mean()
    ema_slow = close.ewm(span=int(p.get("ema_slow", 26))).mean()
    
    # ATR for position sizing
    if all(col in df.columns for col in ['High', 'Low']):
        tr = pd.concat([
            (df['High'] - df['Low']),
            (df['High'] - close.shift(1)).abs(),
            (df['Low'] - close.shift(1)).abs()
        ], axis=1).max(axis=1)
        atr = tr.rolling(14).mean()
        atr_pct = atr / close
    else:
        atr_pct = close.pct_change().abs().rolling(14).mean()
    
    # Strategy signals
    buy_signal = (ema_fast > ema_slow) & (rsi < p.get("rsi_entry", 40))
    sell_signal = (ema_fast < ema_slow) | (rsi > p.get("rsi_exit", 70))
    
    # Generate position signals
    position = pd.Series(0, index=df.index)
    current_pos = 0
    
    for i in range(len(df)):
        if current_pos == 0 and buy_signal.iloc[i]:
            current_pos = 1
        elif current_pos == 1 and sell_signal.iloc[i]:
            current_pos = 0
        position.iloc[i] = current_pos
    
    # Convert to signal format
    signals = pd.Series("HOLD", index=df.index)
    signals[position.diff() == 1] = "BUY"
    signals[position.diff() == -1] = "SELL"
    
    return signals
'''


def calculate_regime_stats(market_data):
    """Calculate enhanced market/regime statistics for AI context"""
    try:
        close = market_data['Close']
        high = market_data['High'] if 'High' in market_data.columns else close
        low = market_data['Low'] if 'Low' in market_data.columns else close
        
        # Returns for calculations
        returns = close.pct_change().dropna()
        
        # 20-day volatility (annualized)
        vol_20d = returns.rolling(20).std().iloc[-1] * np.sqrt(252) * 100 if len(returns) >= 20 else 0
        
        # Downside volatility
        downside_returns = returns[returns < 0]
        downside_vol = downside_returns.std() * np.sqrt(252) * 100 if len(downside_returns) > 5 else 0
        
        # Autocorrelation lag-1
        autocorr1 = returns.autocorr(lag=1) if len(returns) > 20 else 0
        
        # ATR percentage
        if len(market_data) >= 14:
            tr1 = high - low
            tr2 = (high - close.shift(1)).abs()
            tr3 = (low - close.shift(1)).abs()
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            atr = tr.rolling(14).mean().iloc[-1]
            atr_pct = (atr / close.iloc[-1]) * 100 if close.iloc[-1] > 0 else 0
        else:
            atr_pct = 0
        
        # MA20 slope (normalized)
        if len(close) >= 20:
            ma20 = close.rolling(20).mean()
            ma_slope = ((ma20.iloc[-1] - ma20.iloc[-5]) / ma20.iloc[-5]) if len(ma20) >= 5 and ma20.iloc[-5] > 0 else 0
        else:
            ma_slope = 0
        
        # Simple ADX approximation (directional movement strength)
        if len(market_data) >= 14:
            dm_plus = (high.diff()).clip(lower=0)
            dm_minus = (-low.diff()).clip(lower=0)
            dm_sum = dm_plus + dm_minus
            adx = (dm_sum.rolling(14).mean().iloc[-1] / atr if atr > 0 else 0) * 100
        else:
            adx = 0
        
        return {
            'vol_20d': vol_20d,
            'downside_vol': downside_vol,
            'autocorr1': autocorr1,
            'atr_pct': atr_pct,
            'ma_slope': ma_slope,
            'adx': min(adx, 100)  # Cap ADX at 100
        }
    except Exception as e:
        # Return defaults on error
        return {
            'vol_20d': 0, 'downside_vol': 0, 'autocorr1': 0,
            'atr_pct': 0, 'ma_slope': 0, 'adx': 0
        }


def calculate_performance_breakdown(current_perf, market_data):
    """Calculate detailed performance breakdown for AI context"""
    try:
        breakdown_parts = []
        
        # Add available performance metrics
        if 'profit_factor' in current_perf:
            breakdown_parts.append(f"Profit Factor: {current_perf['profit_factor']:.2f}")
        
        if 'total_trades' in current_perf:
            breakdown_parts.append(f"Total Trades: {current_perf['total_trades']}")
        
        if 'turnover' in current_perf:
            breakdown_parts.append(f"Turnover: {current_perf['turnover']:.1f}/yr")
        
        # Add market context
        if len(market_data) > 0:
            price_change = ((market_data['Close'].iloc[-1] / market_data['Close'].iloc[0]) - 1) * 100
            breakdown_parts.append(f"Market Return: {price_change:.2f}%")
        
        return ", ".join(breakdown_parts) if breakdown_parts else "Limited performance data available"
        
    except Exception as e:
        return "Performance breakdown unavailable"


def get_strategy_logic_description(strategy_type):
    """Get one-liner description of strategy logic"""
    descriptions = {
        "Momentum": "Price momentum > threshold with volume confirmation for entries/exits",
        "Mean Reversion": "RSI(14) 30/70 + Bollinger Band mean reversion with fixed SL/TP",
        "Breakout": "Rolling high/low breakout with volume confirmation and ATR stops",
        "Custom": "User-defined custom strategy logic"
    }
    return descriptions.get(strategy_type, "Unknown strategy logic")


def max_drawdown(equity: pd.Series) -> float:
    peak = equity.cummax()
    dd = equity / peak - 1.0
    return float(dd.min())


def sharpe(returns: pd.Series, periods_per_year=252) -> float:
    if returns.std() == 0 or len(returns) == 0:
        return 0.0
    return float((returns.mean() / returns.std()) * np.sqrt(periods_per_year))


def sortino(returns: pd.Series, periods_per_year=252) -> float:
    downside = returns[returns < 0]
    if downside.std() == 0 or len(returns) == 0:
        return 0.0
    return float((returns.mean() / downside.std()) * np.sqrt(periods_per_year))


def profit_factor(returns: pd.Series) -> float:
    gains = returns[returns > 0].sum()
    losses = -returns[returns < 0].sum()
    return float(gains / losses) if losses > 1e-12 else 1.0


def fetch_real_data(symbol: str, from_date: str, to_date: str):
    """Fetch real market data using JugaadDataClient"""
    try:
        client = JugaadDataClient()
        data = client.get_stock_data(symbol, from_date, to_date)
        
        if data is None or data.empty:
            st.error(f"No data found for symbol {symbol}")
            return None
        
        # Debug: Show what columns we got
        st.info(f"📊 Data columns for {symbol}: {list(data.columns)}")
            
        # More flexible column validation - check for essential columns
        essential_cols = ['Close']  # Only Close is absolutely required
        missing_cols = [col for col in essential_cols if col not in data.columns]
        
        if missing_cols:
            st.error(f"Data for {symbol} missing essential columns: {missing_cols}")
            return None
            
        # Clean and format data
        data = data.copy()
        
        # Ensure Date column exists
        if 'Date' not in data.columns:
            if data.index.name in ['Date', 'DATE'] or isinstance(data.index, pd.DatetimeIndex):
                data = data.reset_index()
                if 'index' in data.columns:
                    data['Date'] = data['index']
                    data.drop(columns=['index'], inplace=True)
        
        if 'Date' in data.columns:
            data['Date'] = pd.to_datetime(data['Date'])
            data = data.sort_values('Date').reset_index(drop=True)
        
        # Add missing OHLC columns if not present (use Close as fallback)
        if 'Open' not in data.columns:
            data['Open'] = data['Close'].shift(1)
            data.loc[0, 'Open'] = data.loc[0, 'Close']
        if 'High' not in data.columns:
            data['High'] = data['Close']
        if 'Low' not in data.columns:
            data['Low'] = data['Close']
        if 'Volume' not in data.columns:
            data['Volume'] = 1000000  # Default volume if not available
            
        return data
        
    except Exception as e:
        st.error(f"Error fetching data for {symbol}: {str(e)}")
        return None


# ===================== signal generators =====================


def signal_momentum(
    df: pd.DataFrame, period: int, threshold_pct: float, vol_mult: float = 1.2
):
    px = df["Close"]
    mom = px.pct_change(periods=period) * 100
    vol_ok = df["Volume"] > df["Volume"].rolling(20).mean() * vol_mult
    buy = (mom > threshold_pct) & vol_ok
    sell = (mom < -threshold_pct) & vol_ok
    sig = pd.Series("HOLD", index=df.index)
    sig[buy] = "BUY"
    sig[sell] = "SELL"
    return sig


def signal_mean_rev(
    df: pd.DataFrame,
    rsi_period: int,
    rsi_lo: int,
    rsi_hi: int,
    bb_period: int = 20,
    bb_std: float = 2.0,
):
    px = df["Close"]
    r = rsi(px, rsi_period)
    mid, upper, lower = bollinger_bands(px, bb_period, bb_std)
    buy = (r < rsi_lo) & (px < lower)
    sell = (r > rsi_hi) & (px > upper)
    sig = pd.Series("HOLD", index=df.index)
    sig[buy] = "BUY"
    sig[sell] = "SELL"
    return sig


def signal_breakout(df: pd.DataFrame, lookback: int, vol_mult: float):
    hi = df["Close"].rolling(lookback).max()
    lo = df["Close"].rolling(lookback).min()
    vol_ok = df["Volume"] > df["Volume"].rolling(20).mean() * vol_mult
    buy = (df["Close"] > hi.shift(1)) & vol_ok
    sell = (df["Close"] < lo.shift(1)) & vol_ok
    sig = pd.Series("HOLD", index=df.index)
    sig[buy] = "BUY"
    sig[sell] = "SELL"
    return sig


# ===================== backtester (no look-ahead) =====================


def run_backtest(
    df: pd.DataFrame,
    signal: pd.Series,
    stop_loss_pct: float,
    take_profit_pct: float,
    max_pos_frac: float,
    cost_bps: float = 0.0,
):
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
                if px <= last_entry * (1 - stop_loss_pct) or px >= last_entry * (
                    1 + take_profit_pct
                ):
                    exit_next = 1
            elif curr_pos == -1:
                if px >= last_entry * (1 + stop_loss_pct) or px <= last_entry * (
                    1 - take_profit_pct
                ):
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
    out = pd.DataFrame(
        {
            "Date": df["Date"].iloc[1:].values,
            "Close": df["Close"].iloc[1:].values,
            "Signal": df["Signal"].iloc[1:].values,
            "Position": position[1:],
            "EntryPrice": entry_px[1:],
            "Return": strat_ret.values,
            "Portfolio_Value": equity.values,
        }
    ).set_index("Date")

    # metrics
    years = max(1e-9, len(out) / 252)
    total_ret = equity.iloc[-1] / equity.iloc[0] - 1

    # Calculate round-trips per year
    round_trips_per_year = (changes.sum() / 2) * (252 / len(out))

    metrics = {
        "total_return": round(total_ret * 100, 2),
        "cagr": round(((1 + total_ret) ** (1 / years) - 1) * 100, 2),
        "sharpe_ratio": round(sharpe(out["Return"]), 2),
        "sortino_ratio": round(sortino(out["Return"]), 2),
        "max_drawdown": round(max_drawdown(equity) * 100, 2),
        "profit_factor": round(profit_factor(out["Return"]), 2),
        "win_rate": round((out["Return"] > 0).mean() * 100, 1),
        "turnover": round((np.abs(np.diff(position)) > 0).mean() * 252, 2),
        "round_trips_per_year": round(round_trips_per_year, 2),
        "total_trades": int((np.abs(np.diff(position)) > 0).sum()),
    }
    return out.reset_index(), metrics


# ===================== Universal AI Optimization Functions =====================


def universal_iterate_optimization(
    ai_analyzer,
    strategy_config,
    current_params,
    market_data,
    max_iterations=5,
    min_gain_threshold=5.0,
    max_dd_tolerance=3.0,
    optimization_mode="balanced",  # New parameter for optimization focus
):
    """
    Universal optimization algorithm with multi-objective support
    """
    import json

    try:
        # Guard against Custom strategies
        if current_params.get("strategy_type") == "Custom":
            st.warning(
                "⚠️ Custom strategies are not supported for AI optimization. Please select a predefined strategy type (Momentum, Mean Reversion, or Breakout) to use the AI optimizer."
            )
            return {
                "improvement_found": False,
                "reason": "Custom strategy not supported",
            }

        # Initialize tracking
        champion_config = strategy_config.copy()
        champion_params = current_params.copy()
        iterations = []

        # Run baseline backtest with actual strategy
        baseline_results = run_actual_backtest(champion_params, market_data)
        champion_perf = baseline_results
        
        # Calculate baseline composite score
        baseline_objective_data = calculate_composite_objective(baseline_results, baseline_results, optimization_mode)
        baseline_objective = baseline_objective_data["score"]
        baseline_drawdown = baseline_results.get("max_drawdown", 100)

        st.info(
            f"🎯 **Baseline Performance** ({optimization_mode.title()} Mode): Score {baseline_objective:.1f}, CAGR {baseline_results.get('cagr', 0):.2f}%, Sortino {baseline_results.get('sortino_ratio', 0):.3f}, DD {baseline_drawdown:.2f}%"
        )

        # Create status placeholder for cleaner UI updates
        status_placeholder = st.empty()

        for iteration in range(max_iterations):
            status_placeholder.info(
                f"🔄 **Iteration {iteration + 1}/{max_iterations}** - Requesting AI suggestions..."
            )

            # Get AI suggestion using enhanced universal prompt with optimization mode context
            suggestion = get_universal_ai_suggestion(
                ai_analyzer,
                strategy_config,
                champion_params,
                champion_perf,
                market_data,
                symbol=st.session_state.get('current_symbol', 'SYMBOL'),
                timeframe="1D",
                optimization_mode=optimization_mode  # Pass optimization mode to AI
            )

            if not suggestion.get("ok", False) or not suggestion.get(
                "parameter_suggestions"
            ):
                status_placeholder.warning(
                    f"❌ No suggestions from AI in iteration {iteration + 1}"
                )
                break

            # Limit to ≤3 edits per round (high-impact fix)
            raw_suggestions = suggestion.get("parameter_suggestions", [])
            suggestions = raw_suggestions[:3]  # Enforce budget

            if not suggestions:
                status_placeholder.warning(
                    f"❌ No valid suggestions in iteration {iteration + 1}"
                )
                break

            # Apply suggested parameter changes
            trial_params = apply_universal_changes(
                champion_params.copy(), suggestions, strategy_config["type"]
            )

            # Early stop if no effective change (high-impact fix)
            if trial_params == champion_params:
                status_placeholder.info(
                    "✋ No effective parameter change; stopping optimization."
                )
                break

            status_placeholder.info(f"🧪 **Testing new parameters**: {trial_params}")

            # Run trial backtest with new parameters
            trial_results = run_actual_backtest(trial_params, market_data)
            
            # Calculate composite objective scores
            trial_objective_data = calculate_composite_objective(trial_results, baseline_results, optimization_mode)
            trial_objective = trial_objective_data["score"]
            trial_drawdown = trial_results.get("max_drawdown", 100)

            # Enhanced acceptance logic using composite scoring
            dd_worsening = trial_drawdown - baseline_drawdown
            score_improvement = trial_objective - baseline_objective
            score_improvement_pct = (score_improvement / abs(baseline_objective) * 100) if baseline_objective != 0 else 0

            # Accept if composite score improves and quality gates pass
            if trial_objective_data["passed_gates"]:
                if score_improvement > 0:
                    # Additional check: ensure meaningful improvement
                    min_score_improvement = 5.0  # Minimum score improvement required
                    accepted = score_improvement >= min_score_improvement
                    rejection_reason = f"Score improvement {score_improvement:.1f} below threshold {min_score_improvement}" if not accepted else ""
                else:
                    accepted = False
                    rejection_reason = f"Composite score decreased by {abs(score_improvement):.1f}"
            else:
                accepted = False
                rejection_reason = "; ".join(trial_objective_data["reasons"])

            # Track iteration with enhanced multi-objective context
            trial_return = trial_results.get("total_return", 0)
            baseline_return = baseline_results.get("total_return", 0)
            trial_cagr = trial_results.get("cagr", 0)
            trial_sortino = trial_results.get("sortino_ratio", 0)
            
            iteration_data = {
                "iteration": iteration + 1,
                "composite_score": trial_objective,
                "score_improvement": score_improvement,
                "score_improvement_pct": score_improvement_pct,
                "sortino": trial_sortino,
                "cagr": trial_cagr,
                "total_return": trial_return,
                "max_drawdown": trial_drawdown,
                "dd_worsening": dd_worsening,
                "accepted": accepted,
                "parameters": trial_params.copy(),
                "ai_reasoning": suggestion.get("reasoning", "No reasoning provided"),
                "return_delta": trial_return - baseline_return,
                "suggestions_applied": len(suggestions),
                "rejection_reason": rejection_reason if not accepted else "Accepted - improved composite score",
                "optimization_mode": optimization_mode,
                "objective_breakdown": trial_objective_data.get("breakdown", {}),
                "quality_gates_passed": trial_objective_data["passed_gates"]
            }
            iterations.append(iteration_data)

            if accepted:
                status_placeholder.success(
                    f"✅ **Improvement Accepted!** Score: {trial_objective:.1f} (+{score_improvement:.1f}), CAGR: {trial_cagr:.2f}%, Sortino: {trial_sortino:.3f}, DD: {trial_drawdown:.2f}%"
                )
                champion_params = trial_params
                champion_perf = trial_results
                baseline_objective = trial_objective
                baseline_drawdown = trial_drawdown
                baseline_results = trial_results  # Update baseline for next iteration
            else:
                status_placeholder.warning(
                    f"❌ **Trial Rejected**: {rejection_reason}"
                )

        # Clear status placeholder
        status_placeholder.empty()

        # Calculate improvement percentage based on initial vs final performance
        improvement_percentage = 0.0
        if baseline_results and champion_perf and baseline_results != champion_perf:
            original_sortino = baseline_results.get("sortino_ratio", 0)
            champion_sortino = champion_perf.get("sortino_ratio", 0)
            if original_sortino > 0:
                improvement_percentage = ((champion_sortino - original_sortino) / original_sortino) * 100

        return {
            "champion_params": champion_params,
            "champion_perf": champion_perf,
            "baseline_params": current_params.copy(),  # Add original baseline parameters
            "baseline_perf": baseline_results,  # Add baseline for comparison
            "iterations": iterations,
            "total_iterations": len(iterations),
            "final_objective": baseline_objective,
            "final_drawdown_pct": baseline_drawdown,
            "improvement_found": any(iter_data["accepted"] for iter_data in iterations),
            "improvement_percentage": improvement_percentage,
        }

    except Exception as e:
        st.error(f"Universal optimization error: {e}")
        return None


def get_universal_ai_suggestion(
    ai_analyzer, strategy_config, current_params, current_perf, market_data, symbol="RELIANCE", timeframe="1D", optimization_mode="balanced"
):
    """
    Enhanced AI suggestions with comprehensive market context and structured output
    """
    import json

    try:
        # Get parameter bounds for the strategy type
        param_bounds = get_parameter_bounds(current_params, strategy_config["type"])
        
        # Calculate enhanced market/regime statistics
        regime_stats = calculate_regime_stats(market_data)
        performance_breakdown = calculate_performance_breakdown(current_perf, market_data)
        
        # Get strategy logic description
        strategy_logic = get_strategy_logic_description(strategy_config["type"])
        
        # Enhanced AI prompt with better context about optimization goals
        optimization_mode_guidance = {
            "growth": "PRIORITIZE CAGR maximization with moderate risk constraints. Target 15-25% CAGR while keeping Sortino > 1.0 and DD < 20%.",
            "balanced": "BALANCE growth and risk equally. Target 10-20% CAGR with Sortino > 1.2 and moderate drawdown protection.",
            "quality": "EMPHASIZE risk-adjusted returns. Focus on Sortino > 1.5, Sharpe > 1.0, with conservative CAGR expectations.",
            "conservative": "MINIMIZE risk metrics. Keep DD < 15%, volatility low, with modest positive returns as secondary goal."
        }
        
        strategy_description = f"""You are optimizing a {strategy_config['type']} strategy on {symbol} [{timeframe}] over {len(market_data)} periods.
Costs: 46 bps each way, slippage included.
Logic summary: {strategy_logic}

**OPTIMIZATION MODE: {optimization_mode.upper()}**
{optimization_mode_guidance.get(optimization_mode, "Balanced approach to optimization.")}

Current params and bounds:
{format_params_with_bounds_for_ai(current_params, param_bounds, strategy_config['type'])}

Current performance:
Sortino {current_perf.get('sortino_ratio', 0):.3f}, Sharpe {current_perf.get('sharpe_ratio', 0):.3f}, PF {current_perf.get('profit_factor', 1):.2f}, Return {current_perf.get('total_return', 0):.2f}%, MaxDD {current_perf.get('max_drawdown', 0):.2f}%, WinRate {current_perf.get('win_rate', 0):.1f}%, Trades {current_perf.get('total_trades', 0)}, Turnover {current_perf.get('turnover', 0):.1f}/yr.

Market/regime stats:
20D vol {regime_stats.get('vol_20d', 0):.2f}%, downside vol {regime_stats.get('downside_vol', 0):.2f}%, ADX14 {regime_stats.get('adx', 0):.1f}, autocorr1 {regime_stats.get('autocorr1', 0):.3f}, ATR% {regime_stats.get('atr_pct', 0):.3f}, MA20 slope {regime_stats.get('ma_slope', 0):.4f}.

Performance breakdown:
{performance_breakdown}

**CRITICAL OPTIMIZATION RULES:**
1. NEVER suggest changes that would make a profitable strategy (Return > 0%) lose money
2. NEVER suggest changes that would make positive Sortino (> 0.1) turn negative
3. REJECT any suggestions that would dramatically worsen drawdown (> 2pp increase)
4. FOCUS on strategies that are BOTH profitable AND have good risk-adjusted returns
5. For poor baseline performance (Sortino < 0 or Return < 0), focus on making it profitable first
6. ALIGN suggestions with the {optimization_mode.upper()} optimization mode priorities

Current baseline assessment: {"GOOD" if current_perf.get('sortino_ratio', 0) > 0.5 else "POOR" if current_perf.get('sortino_ratio', 0) <= 0 else "MEDIOCRE"}

Propose up to 3 parameter sets that IMPROVE the overall strategy quality for {optimization_mode.upper()} mode:
- If baseline is GOOD: Require significant improvement (>10% composite score gain) while maintaining profitability
- If baseline is MEDIOCRE: Modest improvement (>5% composite score gain) while protecting returns  
- If baseline is POOR: Focus on making it profitable and positive composite score first

Optimization constraints:
- MaxDD increase ≤ 2.0 pp
- Turnover change ≤ +3/yr  
- risk_per_trade = max_pos_pct × stop_loss ≤ 2.0%
- Maintain or improve total returns
- Prefer parameter plateaus (robust within ±10%)
- Stay within bounds
- Ensure take_profit_pct ≥ 1.2 × stop_loss_pct"""

        # Enhanced JSON schema for structured output
        schema = """{
  "proposals": [
    {
      "name": "Conservative|Balanced|Aggressive",
      "param_updates": {"param_name": value, ...},
      "expected_effect": {"sortino_delta": "+0.1~+0.2", "dd_delta_pp": "-0.5~0.0", "turnover_delta": "-1/yr"},
      "constraints_ok": true,
      "risk_per_trade_pct": 1.5,
      "notes": ["constraint check", "robustness note"]
    }
  ]
}"""

        # Use structured JSON mode for reliable parsing
        messages = [
            {
                "role": "system",
                "content": "You are an expert quantitative trading strategist. Optimize parameters for risk-adjusted returns. Provide structured proposals with trade-off analysis. Focus on robustness over overfitting.",
            },
            {"role": "user", "content": strategy_description + "\n\nReturn ONLY JSON in this format:\n" + schema},
        ]

        response = ai_analyzer.client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=messages,
            temperature=0.1,
            max_tokens=1200,
            response_format={"type": "json_object"},
        )

        ai_response_text = response.choices[0].message.content

        try:
            # Parse enhanced structured JSON response
            data = json.loads(ai_response_text)
            proposals = data.get("proposals", [])
            
            if not proposals:
                # Fallback to old format if new format fails
                parameter_suggestions = []
                for s in data.get("suggestions", []):
                    if "parameter" in s and "proposed" in s:
                        parameter_suggestions.append({
                            "parameter": s["parameter"],
                            "suggested_value": s["proposed"],
                            "reasoning": s.get("why", "AI optimization suggestion"),
                        })
                
                return {
                    "ok": True,
                    "parameter_suggestions": parameter_suggestions,
                    "reasoning": ai_response_text,
                    "raw_response": ai_response_text,
                    "structured_data": data,
                }
            
            # Process enhanced proposals
            best_proposal = None
            for proposal in proposals:
                if proposal.get("constraints_ok", False):
                    best_proposal = proposal
                    break
            
            # If no constraints-ok proposal, take the first one
            if not best_proposal and proposals:
                best_proposal = proposals[0]
            
            if best_proposal:
                parameter_suggestions = []
                param_updates = best_proposal.get("param_updates", {})
                
                for param, value in param_updates.items():
                    parameter_suggestions.append({
                        "parameter": param,
                        "suggested_value": value,
                        "reasoning": f"{best_proposal.get('name', 'AI')} proposal: {best_proposal.get('expected_effect', {}).get('sortino_delta', 'improvement expected')}",
                    })
                
                return {
                    "ok": True,
                    "parameter_suggestions": parameter_suggestions,
                    "reasoning": f"Selected {best_proposal.get('name', 'best')} proposal. {' '.join(best_proposal.get('notes', []))}",
                    "raw_response": ai_response_text,
                    "structured_data": data,
                    "enhanced_format": True,
                    "all_proposals": proposals
                }
            
            return {
                "ok": False,
                "parameter_suggestions": [],
                "reasoning": "No valid proposals in AI response",
                "raw_response": ai_response_text,
            }

        except json.JSONDecodeError:
            # Fallback to regex parsing if JSON fails
            st.warning("⚠️ JSON parsing failed, using fallback text parsing")
            parameter_suggestions = parse_ai_parameter_suggestions(
                ai_response_text, strategy_config["type"]
            )

            return {
                "ok": True,
                "parameter_suggestions": parameter_suggestions,
                "reasoning": ai_response_text,
                "raw_response": ai_response_text,
                "fallback_used": True,
            }

    except Exception as e:
        st.error(f"Error getting AI suggestion: {e}")
        return {"ok": False, "parameter_suggestions": [], "reasoning": str(e)}


def get_parameter_bounds(current_params, strategy_type):
    """Get parameter bounds for a strategy type"""
    bounds = {}

    if strategy_type == "Momentum":
        bounds = {
            "momentum_period": [5, 60],
            "momentum_threshold": [0.1, 5.0],
            "vol_mult": [1.0, 3.0],
            "max_pos_pct": [10, 100],
            "stop_loss": [1, 20],
            "take_profit": [2, 40],
        }
    elif strategy_type == "Mean Reversion":
        bounds = {
            "rsi_period": [5, 30],
            "rsi_lo": [5, 40],  # Consistent naming (was rsi_oversold)
            "rsi_hi": [60, 95],  # Consistent naming (was rsi_overbought)
            "bb_period": [10, 40],
            "bb_std": [1.0, 3.5],
            "max_pos_pct": [10, 100],
            "stop_loss": [1, 20],
            "take_profit": [2, 40],
        }
    elif strategy_type == "Breakout":
        bounds = {
            "brk_period": [10, 100],
            "vol_mult": [1.0, 3.0],
            "max_pos_pct": [10, 100],
            "stop_loss": [1, 20],
            "take_profit": [2, 40],
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
        period_match = re.search(r"momentum.*?period.*?(\d+)", ai_response.lower())
        threshold_match = re.search(r"threshold.*?(\d+\.?\d*)", ai_response.lower())
        volume_match = re.search(r"volume.*?(\d+\.?\d*)", ai_response.lower())

        if period_match:
            suggestions.append(
                {
                    "parameter": "momentum_period",
                    "suggested_value": int(period_match.group(1)),
                    "reasoning": "AI suggested momentum period adjustment",
                }
            )

        if threshold_match:
            suggestions.append(
                {
                    "parameter": "momentum_threshold",
                    "suggested_value": float(threshold_match.group(1)),
                    "reasoning": "AI suggested threshold adjustment",
                }
            )

        if volume_match:
            suggestions.append(
                {
                    "parameter": "vol_mult",
                    "suggested_value": float(volume_match.group(1)),
                    "reasoning": "AI suggested volume filter adjustment",
                }
            )

    elif strategy_type == "Mean Reversion":
        # Look for mean reversion parameters
        rsi_period_match = re.search(r"rsi.*?period.*?(\d+)", ai_response.lower())
        oversold_match = re.search(r"oversold.*?(\d+)", ai_response.lower())
        overbought_match = re.search(r"overbought.*?(\d+)", ai_response.lower())
        bb_period_match = re.search(r"bollinger.*?period.*?(\d+)", ai_response.lower())

        if rsi_period_match:
            suggestions.append(
                {
                    "parameter": "rsi_period",
                    "suggested_value": int(rsi_period_match.group(1)),
                    "reasoning": "AI suggested RSI period adjustment",
                }
            )

        if oversold_match:
            suggestions.append(
                {
                    "parameter": "rsi_lo",
                    "suggested_value": int(oversold_match.group(1)),
                    "reasoning": "AI suggested oversold threshold adjustment",
                }
            )

        if overbought_match:
            suggestions.append(
                {
                    "parameter": "rsi_hi",
                    "suggested_value": int(overbought_match.group(1)),
                    "reasoning": "AI suggested overbought threshold adjustment",
                }
            )

    elif strategy_type == "Breakout":
        # Look for breakout parameters
        lookback_match = re.search(r"lookback.*?(\d+)", ai_response.lower())
        volume_match = re.search(r"volume.*?(\d+\.?\d*)", ai_response.lower())

        if lookback_match:
            suggestions.append(
                {
                    "parameter": "brk_period",
                    "suggested_value": int(lookback_match.group(1)),
                    "reasoning": "AI suggested lookback period adjustment",
                }
            )

        if volume_match:
            suggestions.append(
                {
                    "parameter": "vol_mult",
                    "suggested_value": float(volume_match.group(1)),
                    "reasoning": "AI suggested volume filter adjustment",
                }
            )

    # If no specific suggestions found, create some smart defaults based on performance
    if not suggestions:
        suggestions = create_smart_default_suggestions(strategy_type)

    return suggestions


def create_smart_default_suggestions(strategy_type):
    """Create intelligent default suggestions when AI doesn't provide specific ones"""
    suggestions = []

    if strategy_type == "Momentum":
        suggestions = [
            {
                "parameter": "momentum_period",
                "suggested_value": 25,
                "reasoning": "Increase period for smoother signals",
            },
            {
                "parameter": "momentum_threshold",
                "suggested_value": 1.5,
                "reasoning": "Increase threshold to reduce noise",
            },
        ]
    elif strategy_type == "Mean Reversion":
        suggestions = [
            {
                "parameter": "rsi_period",
                "suggested_value": 16,
                "reasoning": "Adjust RSI period for better sensitivity",
            },
            {
                "parameter": "rsi_lo",
                "suggested_value": 25,
                "reasoning": "Lower oversold threshold for earlier signals",
            },
        ]
    elif strategy_type == "Breakout":
        suggestions = [
            {
                "parameter": "brk_period",
                "suggested_value": 25,
                "reasoning": "Increase lookback for stronger breakouts",
            },
            {
                "parameter": "vol_mult",
                "suggested_value": 1.8,
                "reasoning": "Higher volume filter for quality signals",
            },
        ]

    return suggestions


def apply_universal_changes(current_params, suggestions, strategy_type):
    """Apply parameter suggestions with consistent bounds checking"""
    new_params = current_params.copy()
    bounds = get_parameter_bounds(current_params, strategy_type)

    for suggestion in suggestions:
        param = suggestion["parameter"]
        new_value = suggestion["suggested_value"]

        # Apply bounds checking using the same bounds we showed the AI
        if param in bounds:
            min_val, max_val = bounds[param]
            if isinstance(new_value, (int, float)):
                if param in [
                    "momentum_period",
                    "rsi_period",
                    "bb_period",
                    "brk_period",
                    "rsi_lo",
                    "rsi_hi",
                    "max_pos_pct",
                    "stop_loss",
                    "take_profit",
                ]:
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
        # Check if AI-generated strategy is active
        if "generated_strategy" in st.session_state and st.session_state["generated_strategy"]:
            # Use AI-generated strategy with enhanced validation
            ai_strategy = st.session_state["generated_strategy"]
            ai_analyzer = st.session_state.get("ai_analyzer", None)
            
            # Use safe execution with validation
            execution_result = safe_execute_ai_strategy(
                ai_strategy, 
                market_data, 
                params,  # Pass optimization parameters
                ai_analyzer
            )
            
            if execution_result["ok"]:
                signals = execution_result["signals"]
            else:
                # Use safe fallback signals
                signals = execution_result["signals"]  # This will be HOLD signals
        else:
            # Use the global variables from sidebar for manual strategies
            strategy_type = params.get("strategy_type", "Momentum")

            # Generate signals based on strategy type and parameters
            if strategy_type == "Momentum":
                signals = signal_momentum(
                    market_data,
                    period=int(params.get("momentum_period", 20)),
                    threshold_pct=float(params.get("momentum_threshold", 1.0)),
                    vol_mult=float(params.get("vol_mult", 1.2)),
                )
            elif strategy_type == "Mean Reversion":
                signals = signal_mean_rev(
                    market_data,
                    rsi_period=int(params.get("rsi_period", 14)),
                    rsi_lo=int(params.get("rsi_lo", 30)),
                    rsi_hi=int(params.get("rsi_hi", 70)),
                    bb_period=int(params.get("bb_period", 20)),
                    bb_std=float(params.get("bb_std", 2.0)),
                )
            elif strategy_type == "Breakout":
                signals = signal_breakout(
                    market_data,
                    lookback=int(params.get("brk_period", 20)),
                    vol_mult=float(params.get("vol_mult", 1.5)),
                )
            else:
                # Default to hold signals
                signals = pd.Series("HOLD", index=market_data.index)

        # Run backtest
        bt_df, metrics = run_backtest(
            market_data,
            signals,
            stop_loss_pct=float(params.get("stop_loss", 3)) / 100.0,
            take_profit_pct=float(params.get("take_profit", 6)) / 100.0,
            max_pos_frac=float(params.get("max_pos_pct", 50)) / 100.0,
        )

        return metrics

    except Exception as e:
        st.error(f"Backtest error: {e}")
        # Return fallback metrics
        return {
            "total_return": 0,
            "sharpe_ratio": 0,
            "sortino_ratio": 0,
            "max_drawdown": 100,
            "profit_factor": 1,
            "win_rate": 50,
            "turnover": 1,
            "total_trades": 0,
        }


# ===================== Initialize Session State =====================

# Initialize optimization results in session state
if "optimization_result" not in st.session_state:
    st.session_state.optimization_result = None
if "optimization_suggestions" not in st.session_state:
    st.session_state.optimization_suggestions = None
if "optimization_champion_params" not in st.session_state:
    st.session_state.optimization_champion_params = None

# ===================== UI: sidebar =====================

with st.sidebar:
    st.header("⚙️ Strategy Configuration")
    
    # Data Selection
    st.subheader("📊 Data Selection")
    
    # Popular Indian stocks for easy selection
    popular_stocks = ["RELIANCE", "TCS", "HDFCBANK", "INFY", "HINDUNILVR", "ITC", "SBIN", "BAJFINANCE", "LT", "ASIANPAINT"]
    
    # Initialize symbol in session state if not exists
    if "selected_symbol" not in st.session_state:
        st.session_state.selected_symbol = "RELIANCE"
    
    col1, col2 = st.columns([2, 1])
    with col1:
        symbol = st.text_input("Stock Symbol", value=st.session_state.selected_symbol, help="Enter NSE stock symbol (e.g., RELIANCE, TCS, INFY)")
        # Update session state when text input changes
        if symbol != st.session_state.selected_symbol:
            st.session_state.selected_symbol = symbol
    with col2:
        quick_selected = st.selectbox("Quick Select", [""] + popular_stocks, key="quick_stock")
        if quick_selected and quick_selected != st.session_state.selected_symbol:
            st.session_state.selected_symbol = quick_selected
            st.rerun()
    
    # Use the session state value as the final symbol
    symbol = st.session_state.selected_symbol
    
    # Store symbol in session state for AI optimization
    st.session_state.current_symbol = symbol
    
    st.subheader("📅 Data Period Selection")
    
    # Add helpful info about data period selection
    with st.expander("ℹ️ How to Choose the Right Data Period", expanded=False):
        st.markdown("""
        **Recommended data periods for different purposes:**
        
        🔸 **Quick Strategy Testing**: 3-6 months (faster loading, good for initial testing)
        🔸 **Robust Backtesting**: 1-2 years (captures seasonal patterns and market cycles)  
        🔸 **Production Strategies**: 2-3 years (comprehensive market condition testing)
        
        **Performance Impact:**
        - **3 months**: ~65 trading days - Very fast loading and plotting
        - **6 months**: ~125 trading days - Good balance of speed and reliability  
        - **1 year**: ~250 trading days - Comprehensive but slower
        - **2+ years**: 500+ trading days - Most robust but slowest
        
        💡 **Tip**: Start with 6 months for strategy development, then extend for final validation.
        """)
    
    # Date range selection - Optimized for better performance
    max_date = datetime.now().date()
    min_date = max_date - timedelta(days=365*3)  # 3 years back (maximum)
    
    # Provide quick presets for common backtesting periods
    col1, col2 = st.columns([3, 1])
    with col1:
        # Default to 6 months for faster loading and better UX
        default_from = max_date - timedelta(days=180)  # 6 months back (optimized)
        from_date = st.date_input("From Date", value=default_from, min_value=min_date, max_value=max_date)
        to_date = st.date_input("To Date", value=max_date, min_value=from_date, max_value=max_date)
    
    with col2:
        st.markdown("**Quick Periods:**")
        if st.button("📅 3 Months", help="Last 3 months of data"):
            st.session_state.quick_period = "3M"
            st.rerun()
        if st.button("📅 6 Months", help="Last 6 months of data"):
            st.session_state.quick_period = "6M"
            st.rerun()
        if st.button("📅 1 Year", help="Last 1 year of data"):
            st.session_state.quick_period = "1Y"
            st.rerun()
    
    # Apply quick period selection
    if "quick_period" in st.session_state:
        if st.session_state.quick_period == "3M":
            from_date = max_date - timedelta(days=90)
            to_date = max_date
        elif st.session_state.quick_period == "6M":
            from_date = max_date - timedelta(days=180)
            to_date = max_date
        elif st.session_state.quick_period == "1Y":
            from_date = max_date - timedelta(days=365)
            to_date = max_date
        # Clear the quick period after applying
        del st.session_state.quick_period
    
    st.subheader("🎯 Strategy Settings")
    
    # Check if AI strategy is active and show management controls
    if "generated_strategy" in st.session_state and st.session_state["generated_strategy"]:
        ai_strategy = st.session_state["generated_strategy"]
        st.success(f"🤖 **Active AI Strategy**: {ai_strategy['name']}")
        
        col1, col2 = st.columns([2, 1])
        with col1:
            st.info(f"📝 {ai_strategy.get('description', 'AI-generated strategy')}")
        with col2:
            if st.button("🗑️ Clear AI Strategy", help="Return to manual strategy selection"):
                del st.session_state["generated_strategy"]
                st.success("✅ Cleared AI strategy. Now using manual selection.")
                st.rerun()
        
        # Override manual inputs when AI strategy is active
        strategy_name = ai_strategy['name']
        strategy_type = "AI Generated"
        
        # Show AI strategy parameters (read-only)
        if "params" in ai_strategy:
            with st.expander("⚙️ **AI Strategy Parameters** (Auto-configured)", expanded=False):
                for param, details in ai_strategy["params"].items():
                    if isinstance(details, dict) and "default" in details:
                        bounds = details.get("bounds", [0, 100])
                        st.metric(
                            param.replace("_", " ").title(),
                            f"{details['default']} (range: {bounds[0]}-{bounds[1]})"
                        )
    else:
        # Manual strategy selection (original logic)
        strategy_name = st.text_input("Strategy Name", value="My Strategy")
        strategy_type = st.selectbox(
            "Strategy Type", ["Momentum", "Mean Reversion", "Breakout", "Custom"]
        )
    
    # AI Strategy Generator Section
    st.markdown("---")
    with st.expander("🤖 **AI Strategy Generator** - Let AI Design a New Strategy", expanded=False):
        st.info("🚀 **New Feature**: Let AI create a completely new trading strategy tailored to your market conditions!")
        
        col1, col2 = st.columns([2, 1])
        with col1:
            st.markdown("""
            **What the AI Strategy Generator does:**
            - Analyzes current market regime (volatility, trend, momentum)
            - Designs a new strategy with proper entry/exit rules
            - Generates executable Python code
            - Includes risk management and position sizing
            - Provides parameter bounds for optimization
            """)
        
        with col2:
            if st.button("🎯 Generate New Strategy", key="generate_strategy"):
                if "ai_analyzer" in st.session_state:
                    with st.spinner("🧠 AI is designing your strategy..."):
                        # Get market data for context
                        market_data = fetch_real_data(symbol, from_date.strftime('%Y-%m-%d'), to_date.strftime('%Y-%m-%d'))
                        
                        if market_data is not None:
                            proposal = get_strategy_proposal_from_ai(
                                st.session_state.ai_analyzer, 
                                symbol=symbol, 
                                timeframe="1D",
                                market_data=market_data
                            )
                            
                            if proposal.get("ok", False):
                                st.success("✨ **Strategy Generated Successfully!**")
                                strategy_data = proposal["proposal"]
                                
                                st.markdown(f"### 📊 **{strategy_data.get('name', 'New Strategy')}**")
                                st.markdown(f"**Family:** {strategy_data.get('family', 'hybrid').title()}")
                                st.markdown(f"**Description:** {strategy_data.get('description', 'AI-generated strategy')}")
                                
                                # Show strategy math/rules
                                if "math" in strategy_data:
                                    math_info = strategy_data["math"]
                                    
                                    st.markdown("**📐 Strategy Rules:**")
                                    for rule in math_info.get("rules", []):
                                        st.markdown(f"- {rule}")
                                    
                                    st.markdown("**📊 Technical Indicators:**")
                                    for indicator in math_info.get("indicators", []):
                                        st.markdown(f"- {indicator}")
                                    
                                    if "position_sizing" in math_info:
                                        st.markdown(f"**💰 Position Sizing:** {math_info['position_sizing']}")
                                
                                # Show parameters with bounds
                                if "params" in strategy_data:
                                    st.markdown("**⚙️ Strategy Parameters:**")
                                    params_df = pd.DataFrame([
                                        {
                                            "Parameter": param,
                                            "Default": details.get("default", "N/A"),
                                            "Range": f"{details.get('bounds', [0, 100])[0]} - {details.get('bounds', [0, 100])[1]}"
                                        }
                                        for param, details in strategy_data["params"].items()
                                    ])
                                    st.dataframe(params_df, use_container_width=True)
                                
                                # Generate and show code
                                code_result = generate_strategy_code(proposal)
                                if code_result:
                                    st.markdown("**🐍 Generated Python Code:**")
                                    st.code(code_result["code"], language="python")
                                    
                                    # Option to use this strategy
                                    if st.button("🚀 Use This Strategy", key="use_generated"):
                                        st.session_state["generated_strategy"] = code_result
                                        st.success("✅ **Strategy Activated!** The AI strategy is now being used for backtesting.")
                                        st.info("📈 **Next Steps:** The page will refresh and you'll see the AI strategy in action. Check the 'Strategy Settings' section to confirm it's active.")
                                        st.rerun()
                                
                                # Store proposal for potential use
                                st.session_state["last_ai_proposal"] = proposal
                                
                            else:
                                st.error(f"❌ Strategy generation failed: {proposal.get('error', 'Unknown error')}")
                                if "raw_response" in proposal:
                                    with st.expander("Debug: Raw AI Response"):
                                        st.text(proposal["raw_response"])
                        else:
                            st.error("❌ Could not fetch market data for strategy generation")
                else:
                    st.warning("⚠️ AI Analyzer not available. Please check your Groq API key.")
    
    st.markdown("---")
    
    timeframe = st.selectbox(
        "Timeframe", ["1day", "4hour", "1hour", "15min"], index=0
    )  # display only; backtest is daily here

    # General lookback window (used as default for strategy-specific periods)
    default_lookback = st.slider(
        "Default Lookback Window",
        5,
        100,
        20,
        help="Default period for strategy calculations",
    )

    st.subheader("⚠️ Risk Management")
    max_pos_pct = st.slider("Max Position Size (%)", 1, 100, 50)
    stop_loss = st.slider("Stop Loss (%)", 1, 20, 3)
    take_profit = st.slider("Take Profit (%)", 2, 40, 6)

    # Add basic cost/slippage
    cost_bps = st.slider(
        "Transaction Costs (bps)",
        0.0,
        50.0,
        0.0,
        0.5,
        help="Cost per trade in basis points",
    )

    # strategy-specific (use default_lookback as starting point)
    if strategy_type == "Momentum":
        momentum_period = st.slider(
            "Momentum Period (days)", 5, 60, max(5, min(60, default_lookback))
        )
        momentum_threshold = st.slider("Momentum Threshold (%)", 0.1, 5.0, 1.0)
        vol_mult = st.slider("Volume Filter ×", 1.0, 3.0, 1.2, 0.1)
    elif strategy_type == "Mean Reversion":
        rsi_period = st.slider("RSI Period", 5, 30, max(5, min(30, default_lookback)))
        rsi_lo = st.slider("RSI Oversold", 5, 40, 30)
        rsi_hi = st.slider("RSI Overbought", 60, 95, 70)
        bb_period = st.slider("BB Period", 10, 40, max(10, min(40, default_lookback)))
        bb_std = st.slider("BB Std Dev", 1.0, 3.5, 2.0, 0.1)
    elif strategy_type == "Breakout":
        brk_period = st.slider(
            "Breakout Lookback", 10, 100, max(10, min(100, default_lookback))
        )
        vol_mult = st.slider("Volume Filter ×", 1.0, 3.0, 1.5, 0.1)

    # ===================== AI OPTIMIZATION CONTROLS =====================
    st.markdown("---")
    st.subheader("🤖 AI Strategy Optimization")
    
    # Optimization mode selection
    optimization_mode = st.selectbox(
        "Optimization Mode",
        ["balanced", "growth", "quality", "conservative"],
        index=0,
        help="Balanced: Growth+Risk, Growth: Max returns, Quality: High Sharpe, Conservative: Low risk"
    )
    
    # Optimization button with session state handling
    if st.button("🚀 AI-Optimize Strategy", use_container_width=True, type="primary"):
        st.session_state.trigger_optimization = True
        st.session_state.optimization_in_progress = True
    
    # Show optimization status
    if st.session_state.get("optimization_in_progress", False):
        st.info("🔄 Optimization in progress... Please wait.")
    
    if st.session_state.get("optimization_result") and st.session_state.optimization_result.get("improvement_found"):
        st.success("✅ Optimization completed! Check the 'AI Optimization Results' tab for details.")
        if st.button("🗑️ Clear Results", help="Clear optimization results"):
            st.session_state.optimization_result = None
            st.session_state.optimization_in_progress = False
            st.session_state.trigger_optimization = False
    
    # Initialize session state for optimization
    if "trigger_optimization" not in st.session_state:
        st.session_state.trigger_optimization = False
    if "optimization_in_progress" not in st.session_state:
        st.session_state.optimization_in_progress = False

# ===================== Data + signals =====================

# Convert dates to strings for API
from_date_str = from_date.strftime("%Y-%m-%d")
to_date_str = to_date.strftime("%Y-%m-%d")

# Show loading message
with st.spinner(f"Fetching real market data for {symbol}..."):
    data = fetch_real_data(symbol, from_date_str, to_date_str)

if data is None:
    st.stop()

# Display data info
st.info(f"📊 Loaded {len(data)} trading days of data for **{symbol}** from {from_date_str} to {to_date_str}")

# Performance tip for users
if len(data) > 300:
    st.warning(f"📈 **Performance Tip**: You're using {len(data)} days of data. For faster strategy testing, consider using 3-6 months (~65-125 days). You can always extend the period for final validation.")
elif len(data) < 60:
    st.warning(f"⚠️ **Data Notice**: Only {len(data)} days of data available. Consider using at least 3 months (~65 days) for more reliable backtesting results.")

# ===================== AI OPTIMIZATION TRIGGER =====================

# Check if optimization was triggered
if st.session_state.get("trigger_optimization", False) and st.session_state.get("optimization_in_progress", False):
    # Reset trigger
    st.session_state.trigger_optimization = False
    
    # Get current strategy parameters
    current_params = {
        "strategy_type": strategy_type,
        "max_pos_pct": max_pos_pct,
        "stop_loss": stop_loss,
        "take_profit": take_profit,
    }
    
    if strategy_type == "Momentum":
        current_params.update({
            "momentum_period": momentum_period,
            "momentum_threshold": momentum_threshold,
            "vol_mult": vol_mult,
        })
    elif strategy_type == "Mean Reversion":
        current_params.update({
            "rsi_period": rsi_period,
            "rsi_lo": rsi_lo,
            "rsi_hi": rsi_hi,
            "bb_period": bb_period,
            "bb_std": bb_std,
        })
    elif strategy_type == "Breakout":
        current_params.update({
            "brk_period": brk_period,
            "vol_mult": vol_mult,
        })
    
    # Strategy config for optimization
    strategy_config = {"type": strategy_type}
    
    # Run the optimization
    with st.spinner(f"🤖 Running AI optimization in {optimization_mode} mode..."):
        try:
            ai_analyzer = st.session_state.get("ai_analyzer")
            if ai_analyzer:
                # Get current performance metrics as baseline
                current_perf = run_actual_backtest(current_params, data)
                
                # Run AI optimization
                optimization_result = universal_iterate_optimization(
                    ai_analyzer, strategy_config, current_params, data, 
                    optimization_mode=optimization_mode
                )
                
                # Store results
                st.session_state.optimization_result = optimization_result
                st.session_state.optimization_in_progress = False
                
                if optimization_result.get("improvement_found"):
                    st.success("🎉 AI optimization completed! Found improvements.")
                    st.info("📊 Check the 'AI Optimization Results' tab for detailed analysis.")
                else:
                    st.info("✅ Optimization completed. Current parameters are already optimal!")
                    
            else:
                st.error("❌ AI Analyzer not available. Please check your Groq API key.")
                st.session_state.optimization_in_progress = False
                
        except Exception as e:
            st.error(f"❌ Optimization failed: {str(e)}")
            st.session_state.optimization_in_progress = False

# ===================== Enhanced Signal Generation with AI Strategy Support =====================

# Check if an AI-generated strategy is active
if "generated_strategy" in st.session_state and st.session_state["generated_strategy"]:
    # Use AI-generated strategy
    ai_strategy = st.session_state["generated_strategy"]
    st.success(f"🤖 **Using AI-Generated Strategy**: {ai_strategy['name']}")
    
    with st.expander("📄 **AI Strategy Details**", expanded=False):
        st.markdown(f"**Description:** {ai_strategy.get('description', 'AI-generated strategy')}")
        st.markdown("**Generated Code:**")
        st.code(ai_strategy["code"], language="python")
    
    # Execute AI-generated strategy with enhanced validation
    try:
        # Get AI analyzer for validation if available
        ai_analyzer = st.session_state.get("ai_analyzer", None)
        
        # Use safe execution with GroqAI validation
        execution_result = safe_execute_ai_strategy(
            ai_strategy, 
            data, 
            {},  # Empty params for now, will be filled from AI strategy defaults
            ai_analyzer
        )
        
        if execution_result["ok"]:
            signals = execution_result["signals"]
            
            # Show validation details
            if execution_result["validation_applied"]:
                st.info("✅ **GroqAI Validation Applied**: Code has been validated and optimized for your data structure")
                
                # Show code differences if validation made changes
                if execution_result["fixed_code"] != execution_result["original_code"]:
                    with st.expander("🔧 **Code Validation Changes**", expanded=False):
                        st.markdown("**Original AI Code:**")
                        st.code(execution_result["original_code"], language="python")
                        st.markdown("**Validated & Fixed Code:**")
                        st.code(execution_result["fixed_code"], language="python")
                        st.success("✅ Code has been automatically fixed for compatibility")
            
            st.success(f"✅ **AI Strategy Executed Successfully!** Generated {(signals != 'HOLD').sum()} trading signals")
            
            # Show parameter usage
            params_used = execution_result["params_used"]
            if params_used:
                with st.expander("⚙️ **Parameters Used**", expanded=False):
                    for param, value in params_used.items():
                        st.metric(param.replace("_", " ").title(), f"{value}")
            
            # Override strategy_type for consistent handling
            strategy_type = "AI Generated"
            
        else:
            st.error(f"❌ **AI Strategy Execution Failed**: {execution_result['error']}")
            st.warning("🔄 **Fallback**: Using safe HOLD signals. Check the debug information below.")
            
            signals = execution_result["signals"]  # This will be safe HOLD signals
            
            # Show detailed debug information
            with st.expander("🐛 **Debug Information**", expanded=True):
                st.markdown("**Error Details:**")
                st.error(execution_result["error"])
                
                st.markdown("**Available Data Columns:**")
                st.write(list(data.columns))
                
                st.markdown("**Sample Data:**")
                st.dataframe(data.head(3))
                
                st.markdown("**Original AI Code:**")
                st.code(execution_result["original_code"], language="python")
                
                if ai_analyzer:
                    st.markdown("**Suggestion:** Try regenerating the strategy or check for column name mismatches.")
                else:
                    st.markdown("**Note:** GroqAI validation not available. Ensure your API key is configured.")
            
    except Exception as e:
        st.error(f"❌ **Critical Error in AI Strategy Execution**: {str(e)}")
        st.warning("🔄 **Emergency Fallback**: Using safe HOLD signals")
        signals = pd.Series("HOLD", index=data.index)
        
        # Show emergency debug
        with st.expander("� **Emergency Debug Information**"):
            st.error(f"Exception: {str(e)}")
            st.markdown("**Data Structure:**")
            st.write(f"Columns: {list(data.columns)}")
            st.write(f"Index type: {type(data.index)}")
            st.write(f"Data length: {len(data)}")
            
else:
    # Use manual strategy selection (original logic)
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
    stop_loss_pct=stop_loss / 100.0,
    take_profit_pct=take_profit / 100.0,
    max_pos_frac=max_pos_pct / 100.0,
)

# ===================== STRATEGY RESULTS & OPTIMIZATION DISPLAY =====================

# Create main content tabs for better organization
main_tabs = st.tabs(["📊 Backtest Results", "🤖 AI Optimization Results", "📈 Strategy Analysis"])

with main_tabs[0]:
    # ===================== Backtest Results Tab =====================
    st.subheader("📈 Backtest Performance")
    
    # Performance metrics in a clean layout
    metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
    
    with metric_col1:
        st.metric(
            label="💰 Total Return",
            value=f"{metrics['total_return']:.2f}%",
            delta=f"vs benchmark" if 'benchmark_return' in metrics else None
        )
    
    with metric_col2:
        st.metric(
            label="📈 Sortino Ratio",
            value=f"{metrics['sortino_ratio']:.3f}",
            delta=f"Risk-adjusted return" if metrics['sortino_ratio'] > 1 else None
        )
    
    with metric_col3:
        st.metric(
            label="🎯 Win Rate",
            value=f"{metrics['win_rate']:.1f}%",
            delta=f"{metrics['total_trades']} trades"
        )
    
    with metric_col4:
        st.metric(
            label="� Max Drawdown",
            value=f"{metrics['max_drawdown']:.2f}%",
            delta=f"Risk level" if metrics['max_drawdown'] < 20 else None,
            delta_color="inverse"
        )

    # Chart and data in columns
    chart_col, data_col = st.columns([3, 1])
    
    with chart_col:
        # Optimize plotting for large datasets
        plot_data = bt_df
        if len(bt_df) > 500:  # If more than 500 data points, sample for plotting
            st.info(f"📊 Optimizing plot display for {len(bt_df)} data points...")
            # Sample every nth point to keep plot responsive while maintaining shape
            step = max(1, len(bt_df) // 400)  # Aim for ~400 points max
            plot_data = bt_df.iloc[::step].copy()
            # Always include the last point for accuracy
            if not plot_data.index[-1] == bt_df.index[-1]:
                plot_data = pd.concat([plot_data, bt_df.iloc[[-1]]])
        
        fig = go.Figure()
        # Portfolio
        fig.add_trace(
            go.Scatter(
                x=plot_data["Date"],
                y=plot_data["Portfolio_Value"],
                name="Portfolio Value",
                mode="lines",
                line=dict(width=3, color="#1f77b4"),
            )
        )
        # Price (secondary)
        fig.add_trace(
            go.Scatter(
                x=plot_data["Date"],
                y=plot_data["Close"],
                name="Stock Price",
                mode="lines",
                yaxis="y2",
                line=dict(width=2, color="#ff7f0e", dash="dot"),
            )
        )
        
        # Signal markers - only show significant signals to avoid clutter
        buys = bt_df[bt_df["Signal"] == "BUY"]
        sells = bt_df[bt_df["Signal"] == "SELL"]
        
        # Limit signal markers if too many
        if len(buys) > 50:
            buys = buys.iloc[::max(1, len(buys)//40)]  # Show max 40 buy signals
        if len(sells) > 50:
            sells = sells.iloc[::max(1, len(sells)//40)]  # Show max 40 sell signals
        
        fig.add_trace(
            go.Scatter(
                x=buys["Date"],
                y=buys["Close"],
                mode="markers",
                name="BUY Signal",
                yaxis="y2",
                marker=dict(color="green", size=10, symbol="triangle-up"),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=sells["Date"],
                y=sells["Close"],
                mode="markers",
                name="SELL Signal",
                yaxis="y2",
                marker=dict(color="red", size=10, symbol="triangle-down"),
            )
        )

        fig.update_layout(
            title=f"{symbol} Strategy Performance",
            xaxis=dict(domain=[0.0, 1.0], title="Date"),
            yaxis=dict(title="Portfolio Value (₹)", side="left"),
            yaxis2=dict(title="Stock Price (₹)", overlaying="y", side="right"),
            height=500,
            margin=dict(l=20, r=20, t=40, b=20),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
            hovermode="x unified"
        )
        st.plotly_chart(fig, use_container_width=True)

    with data_col:
        st.markdown("**📊 Strategy Performance**")
        
        performance_data = {
            "Metric": ["Total Return", "CAGR", "Volatility", "Sharpe Ratio", "Sortino Ratio", "Max Drawdown", "Win Rate", "Total Trades"],
            "Value": [
                f"{metrics['total_return']:.2f}%",
                f"{metrics.get('cagr', 0):.2f}%",
                f"{metrics.get('volatility', 0):.2f}%",
                f"{metrics['sharpe_ratio']:.3f}",
                f"{metrics['sortino_ratio']:.3f}",
                f"{metrics['max_drawdown']:.2f}%",
                f"{metrics['win_rate']:.1f}%",
                f"{metrics['total_trades']}"
            ]
        }
        
        st.dataframe(
            pd.DataFrame(performance_data),
            use_container_width=True,
            hide_index=True
        )
        
        # Strategy details
        st.markdown("**⚙️ Strategy Settings**")
        strategy_details = {
            "Strategy": strategy_type,
            "Symbol": symbol,
            "Data Period": f"{from_date} to {to_date}",
            "Position Size": f"{max_pos_pct}%",
            "Stop Loss": f"{stop_loss}%",
            "Take Profit": f"{take_profit}%"
        }
        
        for key, value in strategy_details.items():
            st.text(f"{key}: {value}")

    # Expandable data section
    with st.expander("📊 View Raw Backtest Data", expanded=False):
        st.dataframe(bt_df.tail(100), use_container_width=True)

with main_tabs[1]:
    # ===================== AI Optimization Results Tab =====================
    st.subheader("🤖 AI Strategy Optimization Results")
    
    if st.session_state.optimization_result and st.session_state.optimization_result.get("improvement_found"):
        st.success("🎉 **AI Optimization Successfully Found Improvements!**")
        st.info("📊 Check the Performance Comparison below for detailed improvements.")
        
        # Get optimization data
        champion_params = st.session_state.optimization_result.get("champion_params", {})
        final_perf = st.session_state.optimization_result.get("champion_perf", {})
        baseline_perf = st.session_state.optimization_result.get("baseline_perf", {})
        baseline_params = st.session_state.optimization_result.get("baseline_params", {})
        
        # Parameter Comparison Section
        st.markdown("### ⚙️ Parameter Comparison")
        
        # Create comparison table using stored baseline parameters
        param_comparison_data = []
        for param_key, optimized_value in champion_params.items():
            if param_key in ["strategy_type"]:  # Skip strategy type
                continue
                
            # Get the original value from stored baseline parameters
            original_value = baseline_params.get(param_key)
            
            if original_value is not None:
                # Calculate change
                change = optimized_value - original_value
                change_pct = (change / original_value * 100) if original_value != 0 else 0
                change_str = f"{change:+.2f} ({change_pct:+.1f}%)"
                
                param_comparison_data.append({
                    "Parameter": param_key.replace("_", " ").title(),
                    "Original": f"{original_value}",
                    "Optimized": f"{optimized_value}",
                    "Change": change_str
                })
        
        if param_comparison_data:
            comparison_df = pd.DataFrame(param_comparison_data)
            st.dataframe(comparison_df, use_container_width=True, hide_index=True)
        
        # Performance comparison
        st.markdown("### 📊 Performance Comparison")
        
        # Create side-by-side performance comparison table
        performance_comparison = [
            {
                "Metric": "📈 Sortino Ratio",
                "Original": f"{baseline_perf.get('sortino_ratio', 0):.3f}",
                "Optimized": f"{final_perf.get('sortino_ratio', 0):.3f}",
                "Improvement": "✅ Better" if final_perf.get('sortino_ratio', 0) > baseline_perf.get('sortino_ratio', 0) else "🔄 Same" if final_perf.get('sortino_ratio', 0) == baseline_perf.get('sortino_ratio', 0) else "❌ Worse"
            },
            {
                "Metric": "� Total Return",
                "Original": f"{baseline_perf.get('total_return', 0):.2f}%",
                "Optimized": f"{final_perf.get('total_return', 0):.2f}%",
                "Improvement": "✅ Better" if final_perf.get('total_return', 0) > baseline_perf.get('total_return', 0) else "🔄 Same" if final_perf.get('total_return', 0) == baseline_perf.get('total_return', 0) else "❌ Worse"
            },
            {
                "Metric": "� CAGR",
                "Original": f"{baseline_perf.get('cagr', 0):.2f}%",
                "Optimized": f"{final_perf.get('cagr', 0):.2f}%",
                "Improvement": "✅ Better" if final_perf.get('cagr', 0) > baseline_perf.get('cagr', 0) else "🔄 Same" if final_perf.get('cagr', 0) == baseline_perf.get('cagr', 0) else "❌ Worse"
            },
            {
                "Metric": "📉 Max Drawdown",
                "Original": f"{baseline_perf.get('max_drawdown', 0):.2f}%",
                "Optimized": f"{final_perf.get('max_drawdown', 0):.2f}%",
                "Improvement": "✅ Better" if final_perf.get('max_drawdown', 0) < baseline_perf.get('max_drawdown', 0) else "🔄 Same" if final_perf.get('max_drawdown', 0) == baseline_perf.get('max_drawdown', 0) else "❌ Worse"
            },
            {
                "Metric": "🎯 Win Rate",
                "Original": f"{baseline_perf.get('win_rate', 0):.1f}%",
                "Optimized": f"{final_perf.get('win_rate', 0):.1f}%",
                "Improvement": "✅ Better" if final_perf.get('win_rate', 0) > baseline_perf.get('win_rate', 0) else "🔄 Same" if final_perf.get('win_rate', 0) == baseline_perf.get('win_rate', 0) else "❌ Worse"
            },
            {
                "Metric": "⚖️ Sharpe Ratio",
                "Original": f"{baseline_perf.get('sharpe_ratio', 0):.3f}",
                "Optimized": f"{final_perf.get('sharpe_ratio', 0):.3f}",
                "Improvement": "✅ Better" if final_perf.get('sharpe_ratio', 0) > baseline_perf.get('sharpe_ratio', 0) else "🔄 Same" if final_perf.get('sharpe_ratio', 0) == baseline_perf.get('sharpe_ratio', 0) else "❌ Worse"
            }
        ]
        
        performance_df = pd.DataFrame(performance_comparison)
        # st.dataframe(performance_df, use_container_width=True, hide_index=True)
        
        # Display as metrics instead of table
        comp_col1, comp_col2, comp_col3, comp_col4 = st.columns(4)
        
        with comp_col1:
            sortino_improvement = ((final_perf.get("sortino_ratio", 0) - baseline_perf.get("sortino_ratio", 0)) / baseline_perf.get("sortino_ratio", 1) * 100) if baseline_perf.get("sortino_ratio", 0) > 0 else 0
            st.metric("📈 Sortino Ratio", f"{final_perf.get('sortino_ratio', 0):.3f}", f"{sortino_improvement:+.1f}%")
            
        with comp_col2:
            return_improvement = final_perf.get('total_return', 0) - baseline_perf.get('total_return', 0)
            st.metric("💰 Total Return", f"{final_perf.get('total_return', 0):.2f}%", f"{return_improvement:+.2f}pp")
            
        with comp_col3:
            dd_change = final_perf.get('max_drawdown', 0) - baseline_perf.get('max_drawdown', 0)
            st.metric("📉 Max Drawdown", f"{final_perf.get('max_drawdown', 0):.2f}%", f"{dd_change:+.2f}pp", delta_color="inverse")
            
        with comp_col4:
            wr_change = final_perf.get('win_rate', 0) - baseline_perf.get('win_rate', 0)
            st.metric("🎯 Win Rate", f"{final_perf.get('win_rate', 0):.1f}%", f"{wr_change:+.1f}pp")

        # Additional metrics row
        comp_col5, comp_col6, comp_col7, comp_col8 = st.columns(4)
        
        with comp_col5:
            cagr_improvement = final_perf.get('cagr', 0) - baseline_perf.get('cagr', 0)
            st.metric("📊 CAGR", f"{final_perf.get('cagr', 0):.2f}%", f"{cagr_improvement:+.2f}pp")
            
        with comp_col6:
            sharpe_improvement = final_perf.get('sharpe_ratio', 0) - baseline_perf.get('sharpe_ratio', 0)
            st.metric("⚖️ Sharpe Ratio", f"{final_perf.get('sharpe_ratio', 0):.3f}", f"{sharpe_improvement:+.3f}")
            
        with comp_col7:
            pf_improvement = final_perf.get('profit_factor', 0) - baseline_perf.get('profit_factor', 0)
            st.metric("💹 Profit Factor", f"{final_perf.get('profit_factor', 0):.2f}", f"{pf_improvement:+.2f}")
            
        with comp_col8:
            trades_change = final_perf.get('total_trades', 0) - baseline_perf.get('total_trades', 0)
            st.metric("🔄 Total Trades", f"{final_perf.get('total_trades', 0)}", f"{trades_change:+.0f}")

        st.success("💡 **How to Apply**: Adjust the sliders in the sidebar to match these optimized values!")
        
        # Show optimization journey
        iterations = st.session_state.optimization_result.get("iterations", [])
        if iterations:
            st.markdown("### 📈 Optimization Journey")
            
            # Enhanced journey table with parameter details
            journey_data = []
            for i, iter_data in enumerate(iterations):
                # Get parameter changes summary
                iter_params = iter_data.get("parameters", {})
                param_summary = []
                for key, value in iter_params.items():
                    if key not in ["strategy_type"]:
                        param_summary.append(f"{key}={value}")
                param_str = ", ".join(param_summary[:3])  # Show first 3 params
                if len(param_summary) > 3:
                    param_str += "..."
                
                journey_data.append({
                    "Iteration": iter_data.get("iteration", i + 1),
                    "Score": f"{iter_data.get('composite_score', 0):.1f}",
                    "CAGR": f"{iter_data.get('cagr', 0):.2f}%",
                    "Sortino": f"{iter_data.get('sortino', 0):.3f}",
                    "Max DD": f"{iter_data.get('max_drawdown', 0):.1f}%",
                    "Status": "✅ Accepted" if iter_data.get("accepted", False) else "❌ Rejected",
                    "Key Changes": param_str
                })
            
            journey_df = pd.DataFrame(journey_data)
            st.dataframe(journey_df, use_container_width=True, hide_index=True)
            
            # Add detailed parameter view
            with st.expander("🔍 View Detailed Parameter Changes", expanded=False):
                for i, iter_data in enumerate(iterations):
                    status_icon = "✅" if iter_data.get("accepted", False) else "❌"
                    st.markdown(f"**{status_icon} Iteration {iter_data.get('iteration', i + 1)}**")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown("**Parameters Tested:**")
                        iter_params = iter_data.get("parameters", {})
                        for key, value in iter_params.items():
                            if key != "strategy_type":
                                st.text(f"• {key.replace('_', ' ').title()}: {value}")
                    
                    with col2:
                        st.markdown("**Results:**")
                        st.text(f"• Score: {iter_data.get('composite_score', 0):.2f}")
                        st.text(f"• CAGR: {iter_data.get('cagr', 0):.2f}%")
                        st.text(f"• Sortino: {iter_data.get('sortino', 0):.3f}")
                        st.text(f"• Max DD: {iter_data.get('max_drawdown', 0):.2f}%")
                    
                    if iter_data.get("ai_reasoning"):
                        st.markdown("**AI Reasoning:**")
                        st.text(iter_data.get("ai_reasoning", ""))
                    
                    st.markdown("---")
        
        if st.button("🗑️ Clear Results", use_container_width=True):
            st.session_state.optimization_result = None
            st.rerun()
    
    elif st.session_state.optimization_result:
        st.warning("🔍 **AI Analysis Complete - No Significant Improvements Found**")
        st.info("Your current strategy parameters appear to be well-tuned for the current market conditions.")
        
        # Show what was attempted even if no improvement
        iterations = st.session_state.optimization_result.get("iterations", [])
        if iterations:
            st.markdown("### 🔬 Analysis Attempts")
            attempt_data = []
            for i, iter_data in enumerate(iterations):
                attempt_data.append({
                    "Attempt": iter_data.get("iteration", i + 1),
                    "Score": f"{iter_data.get('composite_score', 0):.1f}",
                    "CAGR": f"{iter_data.get('cagr', 0):.2f}%",
                    "Sortino": f"{iter_data.get('sortino', 0):.3f}",
                    "Result": iter_data.get("rejection_reason", "No improvement")[:50] + "..." if len(iter_data.get("rejection_reason", "")) > 50 else iter_data.get("rejection_reason", "No improvement")
                })
            
            attempt_df = pd.DataFrame(attempt_data)
            st.dataframe(attempt_df, use_container_width=True, hide_index=True)
        
        if st.button("🗑️ Clear Results", key="clear_no_improvement", use_container_width=True):
            st.session_state.optimization_result = None
            st.rerun()
    
    else:
        st.info("🤖 **No optimization results yet.** Run an optimization to see detailed analysis here.")

with main_tabs[2]:
    # ===================== Strategy Analysis Tab =====================  
    st.subheader("📈 Detailed Strategy Analysis")
    
    analysis_col1, analysis_col2 = st.columns([1, 1])
    
    with analysis_col1:
        st.markdown("**📊 Performance Metrics**")
        m = metrics
        st.metric("Total Return", f"{m['total_return']:.2f}%")
        st.metric("CAGR", f"{m['cagr']:.2f}%")
        st.metric("Max Drawdown", f"{m['max_drawdown']:.2f}%")
        st.metric("Sharpe Ratio", f"{m['sharpe_ratio']:.2f}")
        st.metric("Sortino Ratio", f"{m['sortino_ratio']:.2f}")
        st.metric("Profit Factor", f"{m['profit_factor']:.2f}")
        st.metric("Win Rate", f"{m['win_rate']:.1f}%")
        st.metric("Turnover (×/yr)", f"{m['turnover']:.2f}")

    with analysis_col2:
        st.markdown("**📊 Signal Distribution**")
        sig_counts = bt_df["Signal"].value_counts()
        
        # Create a simple bar chart for signal distribution
        fig_signals = go.Figure(data=[
            go.Bar(x=sig_counts.index, y=sig_counts.values, 
                  marker_color=['green' if x == 'BUY' else 'red' if x == 'SELL' else 'gray' for x in sig_counts.index])
        ])
        fig_signals.update_layout(
            title="Trading Signals Distribution",
            xaxis_title="Signal Type",
            yaxis_title="Count",
            height=300
        )
        st.plotly_chart(fig_signals, use_container_width=True)
        
        st.markdown("**⚙️ Strategy Configuration**")
        st.text(f"Strategy: {strategy_type}")
        st.text(f"Symbol: {symbol}")
        st.text(f"Period: {from_date} to {to_date}")
        st.text(f"Position Size: {max_pos_pct}%")
        st.text(f"Stop Loss: {stop_loss}%")
        st.text(f"Take Profit: {take_profit}%")

# ===================== REMOVE OLD LAYOUT SECTION =====================
# The old column layout is removed as it's now organized in tabs
    fig_pie = px.pie(
        values=sig_counts.values, names=sig_counts.index, title="Signal Distribution"
    )
    fig_pie.update_layout(height=300, margin=dict(l=10, r=10, t=40, b=10))
    st.plotly_chart(fig_pie, use_container_width=True)

    # Actions
    st.subheader("🎯 Actions")
    
    # Check if AI strategy is active for manifest
    if "generated_strategy" in st.session_state and st.session_state["generated_strategy"]:
        ai_strategy = st.session_state["generated_strategy"]
        # Create manifest for AI-generated strategy
        manifest = {
            "name": ai_strategy['name'],
            "type": "AI Generated",
            "family": ai_strategy.get("family", "hybrid"),
            "timeframe": timeframe,
            "description": ai_strategy.get("description", "AI-generated strategy"),
            "ai_generated": True,
            "code": ai_strategy["code"],
            "knobs": {},
            "invariants": [
                "no_lookahead",
                "apply_costs_externally", 
                "close_to_close_returns",
                "ai_generated_strategy"
            ],
            "performance": m,
        }
        
        # Add AI strategy parameter bounds
        if "params" in ai_strategy:
            for param, details in ai_strategy["params"].items():
                if isinstance(details, dict) and "bounds" in details:
                    manifest["knobs"][param] = details["bounds"]
    else:
        # export a "manifest" for manual strategy
        manifest = {
            "name": strategy_name,
            "type": strategy_type,
            "timeframe": timeframe,
            "knobs": {},
            "invariants": [
                "no_lookahead",
                "apply_costs_externally",
                "close_to_close_returns",
            ],
            "performance": m,
        }
        
        # Add strategy-specific knobs based on strategy type
        if strategy_type == "Momentum":
            manifest["knobs"] = {
                "momentum_period": [
                    max(5, int(momentum_period * 0.7)),
                    int(momentum_period * 1.3),
                ],
                "momentum_threshold": [
                    round(momentum_threshold * 0.7, 3),
                    round(momentum_threshold * 1.3, 3),
                ],
                "vol_mult": [
                    max(1.0, round(vol_mult * 0.8, 2)),
                    round(vol_mult * 1.2, 2),
                ],
            }
        elif strategy_type == "Mean Reversion":
            manifest["knobs"] = {
                "rsi_period": [
                    max(5, int(rsi_period * 0.7)),
                    int(rsi_period * 1.3),
                ],
                "rsi_lo": [
                    max(5, int(rsi_lo * 0.8)),
                    int(rsi_lo * 1.1),
                ],
                "rsi_hi": [
                    int(rsi_hi * 0.9),
                    min(95, int(rsi_hi * 1.05)),
                ],
                "bb_period": [max(10, int(bb_period * 0.8)), int(bb_period * 1.2)],
                "bb_std": [
                    round(max(1.0, bb_std * 0.8), 2),
                    round(bb_std * 1.2, 2),
                ],
            }
        elif strategy_type == "Breakout":
            manifest["knobs"] = {
                "brk_period": [
                    max(10, int(brk_period * 0.8)),
                    int(brk_period * 1.2),
                ],
                "vol_mult": [
                    max(1.0, round(vol_mult * 0.8, 2)),
                    round(vol_mult * 1.2, 2),
                ],
            }
        else:  # Custom strategy
            manifest["knobs"] = {
                "note": "Custom strategy - no predefined parameters"
            }

    # Use proper JSON encoding instead of pd.Series().to_json()
    import json

    st.download_button(
        "📦 Export Manifest JSON",
        data=json.dumps(manifest, separators=(",", ":")),
        file_name="strategy_manifest.json",
        mime="application/json",
        use_container_width=True,
    )

    # ===================== END OF STRATEGY BUILDER =====================
    # All UI is now organized in tabs above
