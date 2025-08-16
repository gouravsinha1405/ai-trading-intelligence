import json
import os
import re
from copy import deepcopy
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from groq import Groq

# Constants for production use
IST = ZoneInfo("Asia/Kolkata")
MAX_CHANGES = 3
MAX_NEWS = 10


class GroqAnalyzer:
    """Enhanced AI-powered market analysis using structured data signals for Groq API"""

    def __init__(self, api_key: str, model: str = "llama-3.3-70b-versatile"):
        """Initialize the enhanced Groq analyzer"""
        self.client = Groq(api_key=api_key)
        self.model = model
        self.temperature = 0.1  # Lower for more deterministic analysis

    def update_model(self, model: str):
        """Update the AI model"""
        self.model = model

    def update_temperature(self, temperature: float):
        """Update the response temperature"""
        self.temperature = temperature

    def _compact(self, obj: dict) -> str:
        """Compact JSON without pretty printing to save tokens"""
        return json.dumps(obj, separators=(",", ":"), ensure_ascii=False)

    def _preflight(self, data: dict) -> Optional[Dict]:
        """Guard against empty inputs to avoid wasting tokens"""
        missing = []
        if not data["market_stats"].get("index"):
            missing.append("market_stats.index")
        if not data["strategy"].get("knobs"):
            missing.append("strategy.knobs")
        if not data["performance"].get("headline"):
            missing.append("performance.headline")

        if missing:
            return {
                "ok": False,
                "issues": [f"Missing required data: {', '.join(missing)}"],
                "changes": [],
                "risks": [],
                "test_plan": [],
            }
        return None

    def _extract_json(self, s: str) -> str:
        """Extract JSON from response using stack parser for robustness"""
        s = s.strip()

        # Strip code fences quickly
        if s.startswith("```"):
            s = re.sub(r"^```(?:json)?|```$", "", s, flags=re.M).strip()

        # Use stack parser to find balanced braces, but look for JSON patterns
        candidates = []
        depth = 0
        start = None

        for i, ch in enumerate(s):
            if ch == "{":
                if depth == 0:
                    # Check if this looks like start of JSON (next char should be " or
                    # })
                    if i + 1 < len(s) and s[i + 1] in '"}\n \t':
                        start = i
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0 and start is not None:
                    candidates.append(s[start : i + 1])
                    start = None

        # Return the longest candidate (most likely to be complete JSON)
        if candidates:
            return max(candidates, key=len)

        # Fallback to original string if no candidates found
        return s

    def _now_ist(self) -> str:
        """Get current time in IST timezone"""
        return datetime.now(tz=IST).isoformat()

    def _clamp_changes(self, suggestion: dict, knobs: dict) -> dict:
        """Enforce ≤3 changes and validate ranges within knob bounds"""
        changes = suggestion.get("changes", [])
        changes = changes[:MAX_CHANGES]  # Enforce max changes
        clamped = []

        for ch in changes:
            param = ch.get("param")
            rng = ch.get("new_range", [])

            if not (param in knobs and isinstance(rng, list) and len(rng) == 2):
                continue

            # Get allowed range from knobs
            low, high = knobs[param]

            # Clamp the suggested range within allowed bounds
            a = max(low, min(high, float(rng[0])))
            b = max(low, min(high, float(rng[1])))

            if a > b:
                a, b = b, a  # Ensure proper order

            ch["new_range"] = [round(a, 6), round(b, 6)]
            clamped.append(ch)

        suggestion["changes"] = clamped
        return suggestion

    def _postvalidate(
        self, result: dict, knobs: dict, constraints: dict, invariants: List[str]
    ) -> dict:
        """Enforce global constraints and validate areas/params post-LLM"""
        allowed_areas = {
            "gate",
            "entry",
            "exit",
            "sizing",
            "timing",
            "portfolio",
            "universe",
            "risk",
        }
        validated_changes = []
        max_risk_per_trade = constraints.get("risk_per_trade", 0.01)

        for ch in result.get("changes", []):
            # Validate area
            if ch.get("area") not in allowed_areas:
                continue

            # Validate param exists in knobs
            param = ch.get("param")
            if param not in knobs:
                continue

            # Validate range format
            new_range = ch.get("new_range", [None, None])
            if len(new_range) != 2 or any(x is None for x in new_range):
                continue

            lo, hi = new_range
            if np.isnan([lo, hi]).any():
                continue

            # Enforce risk per trade cap
            if param == "risk_per_trade":
                lo = min(lo, max_risk_per_trade)
                hi = min(hi, max_risk_per_trade)
                ch["new_range"] = [lo, hi]

            # Avoid degenerate ranges
            if abs(hi - lo) < 1e-9:
                continue

            validated_changes.append(ch)

        # Enforce MAX_CHANGES limit
        result["changes"] = validated_changes[:MAX_CHANGES]
        return result

    def _make_request(
        self, prompt: str, system_prompt: str = None, json_mode: bool = False
    ) -> str:
        """Make a request to Groq API with retry logic and JSON mode support"""

        messages = []

        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        messages.append({"role": "user", "content": prompt})

        kwargs = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": 2048,
        }

        # Enable JSON mode if supported
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}

        # Basic retry logic with telemetry
        for attempt in range(3):
            try:
                t0 = datetime.now(tz=IST)
                response = self.client.chat.completions.create(**kwargs)
                latency_ms = (datetime.now(tz=IST) - t0).total_seconds() * 1000

                # Return content with telemetry metadata
                content = response.choices[0].message.content
                return content

            except Exception as e:
                if attempt == 2:  # Last attempt
                    return f"__LLM_ERROR__:{str(e)}"

        return "__LLM_ERROR__:Maximum retries exceeded"

    def _get_telemetry_meta(self) -> Dict:
        """Get telemetry metadata for debugging"""
        return {
            "model": self.model,
            "temperature": self.temperature,
            "ts": self._now_ist(),
            "prompt_version": "v1.1",
        }

    def _compute_market_stats(self, price_data: pd.DataFrame) -> Dict:
        """Compress market data into statistical signals"""
        if price_data.empty:
            return {}

        # Calculate returns
        returns = price_data["Close"].pct_change().dropna()

        # Basic return statistics
        ret_stats = {
            "ret_mean": round(returns.mean(), 6),
            "ret_std": round(returns.std(), 4),
            "skew": round(returns.skew(), 2),
            "kurt": round(returns.kurtosis(), 2),
            "q": [
                round(q, 4)
                for q in returns.quantile([0.05, 0.25, 0.5, 0.75, 0.95]).tolist()
            ],
        }

        # ATR calculations using Wilder's smoothing
        if all(col in price_data.columns for col in ["High", "Low", "Close"]):
            atr_pct = self._atr_pct_wilder(price_data)
            if not atr_pct.empty:
                ret_stats.update(
                    {
                        "atrp_mean": round(atr_pct.mean(), 4),
                        "atrp_p95": round(atr_pct.quantile(0.95), 4),
                    }
                )

        # Autocorrelation and trend measures
        if len(returns) > 20:
            ret_stats.update(
                {
                    "lag1_autocorr": round(returns.autocorr(lag=1), 3),
                    "hurst": round(self._calculate_hurst(price_data["Close"]), 3),
                    "r2_80": round(
                        self._calculate_trend_r2(price_data["Close"], 80), 3
                    ),
                }
            )

        # Breadth (safer calculation with adequate samples)
        if len(price_data) >= 220:
            sma_200 = price_data["Close"].rolling(200, min_periods=200).mean()
            mask = sma_200.notna()
            if mask.sum() >= 20:
                # Use last 60 observations for breadth calculation
                recent_close = price_data["Close"][mask].tail(60)
                recent_sma = sma_200[mask].tail(60)
                breadth = (recent_close > recent_sma).mean()
                ret_stats["breadth_above_200dma"] = round(float(breadth), 3)

        return ret_stats

    def _atr_pct_wilder(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate ATR percentage using Wilder's smoothing (EMA)"""
        if not all(c in df.columns for c in ["High", "Low", "Close"]):
            return pd.Series(dtype=float)

        high_low = df["High"] - df["Low"]
        high_close = (df["High"] - df["Close"].shift(1)).abs()
        low_close = (df["Low"] - df["Close"].shift(1)).abs()

        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = tr.ewm(alpha=1 / period, adjust=False).mean()

        return (atr / df["Close"]).replace([np.inf, -np.inf], np.nan)

    def _calculate_hurst(self, prices: pd.Series, max_lag: int = 100) -> float:
        """Calculate Hurst exponent for trend persistence using log prices"""
        try:
            # Use log prices to avoid scale effects
            log_prices = np.log(prices.replace(0, np.nan)).dropna()
            lags = range(2, min(max_lag, len(log_prices) // 4))

            if len(log_prices) < 10 or len(list(lags)) < 3:
                return 0.5

            tau = [
                np.sqrt(np.std(np.subtract(log_prices[lag:], log_prices[:-lag])))
                for lag in lags
            ]

            # Filter out zero or invalid tau values
            valid_pairs = [
                (lag, t) for lag, t in zip(lags, tau) if t > 0 and np.isfinite(t)
            ]

            if len(valid_pairs) < 3:
                return 0.5

            lags_valid, tau_valid = zip(*valid_pairs)
            poly = np.polyfit(np.log(lags_valid), np.log(tau_valid), 1)
            return float(np.clip(poly[0], 0.0, 1.0))
        except BaseException:
            return 0.5

    def _calculate_trend_r2(self, prices: pd.Series, window: int) -> float:
        """Calculate rolling R-squared for trendiness using log prices"""
        try:
            # Use log prices for better trend analysis
            log_prices = np.log(prices.replace(0, np.nan)).dropna()
            x = np.arange(window)
            r2_values = []

            for i in range(window, len(log_prices)):
                y = log_prices.iloc[i - window : i].values
                if len(y) == window and np.isfinite(y).all():
                    slope, intercept = np.polyfit(x, y, 1)
                    y_pred = slope * x + intercept
                    ss_res = np.sum((y - y_pred) ** 2)
                    ss_tot = np.sum((y - y.mean()) ** 2)
                    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
                    r2_values.append(max(0.0, r2))

            return float(np.median(r2_values)) if r2_values else 0.0
        except BaseException:
            return 0.0

    def _extract_regime_diagnostics(
        self, data: pd.DataFrame, regime_data: pd.DataFrame = None
    ) -> Dict:
        """Extract regime analysis into compact signals"""

        # Basic gate variables (can be enhanced with actual regime detector)
        diagnostics = {
            "gate_vars": {},
            "thresholds": {},
            "duty_cycle": {},
            "perf_by_regime": {},
        }

        if not data.empty:
            # Calculate basic technical indicators for gates
            returns = data["Close"].pct_change()

            # Volatility proxy (not ADX - renamed for clarity)
            if len(data) > 20:
                vol_proxy = returns.rolling(14).std() * 100
                diagnostics["gate_vars"]["vol_proxy14_med"] = round(
                    vol_proxy.median(), 3
                )
                diagnostics["thresholds"]["vol_on"] = round(vol_proxy.quantile(0.6), 3)
                diagnostics["thresholds"]["vol_off"] = round(vol_proxy.quantile(0.4), 3)

            # ATR percentage using Wilder's smoothing
            if all(col in data.columns for col in ["High", "Low"]):
                atr_pct = self._atr_pct_wilder(data)
                if not atr_pct.empty:
                    diagnostics["gate_vars"]["atrp_med"] = round(atr_pct.median(), 4)
                    diagnostics["thresholds"]["atrp_lo"] = round(
                        atr_pct.quantile(0.25), 4
                    )
                    diagnostics["thresholds"]["atrp_hi"] = round(
                        atr_pct.quantile(0.75), 4
                    )

            # R-squared trendiness
            r2_80 = self._calculate_trend_r2(data["Close"], 80)
            diagnostics["gate_vars"]["r2_80_med"] = round(r2_80, 3)

        # Only use real regime data if available - don't fabricate
        if regime_data is not None and not regime_data.empty:
            if "market_regime" in regime_data.columns:
                regime_counts = regime_data["market_regime"].value_counts(
                    normalize=True
                )
                diagnostics["duty_cycle"] = {
                    regime: round(count, 3) for regime, count in regime_counts.items()
                }

                # Calculate real performance by regime if available
                if "returns" in regime_data.columns:
                    perf_by_regime = {}
                    for regime in regime_data["market_regime"].unique():
                        regime_returns = regime_data[
                            regime_data["market_regime"] == regime
                        ]["returns"]
                        if len(regime_returns) > 5:  # Minimum observations
                            sharpe = (
                                regime_returns.mean()
                                / regime_returns.std()
                                * np.sqrt(252)
                                if regime_returns.std() > 0
                                else 0
                            )
                            perf_by_regime[regime] = {"sharpe": round(sharpe, 2)}
                    diagnostics["perf_by_regime"] = perf_by_regime
                else:
                    diagnostics["perf_by_regime"] = {}
            else:
                diagnostics["duty_cycle"] = {}
                diagnostics["perf_by_regime"] = {}
        else:
            # Leave empty if no real regime data - let LLM handle insufficient data
            diagnostics["duty_cycle"] = {}
            diagnostics["perf_by_regime"] = {}

        return diagnostics

    def _extract_news_signals(self, news_data: List[Dict]) -> List[Dict]:
        """Extract and structure news into AI-friendly signals with IST timestamps"""
        signals = []

        for item in (news_data or [])[:MAX_NEWS]:
            # Truncate text to save tokens
            title = (item.get("title") or "")[:160]
            summary = (item.get("summary") or "")[:280]
            source = item.get("source") or "MARKET"

            # Handle timezone conversion to IST - assume UTC if tz-naive
            published = item.get("published")
            try:
                # Safer default: assume UTC if tz-naive, then convert to IST
                dt = pd.to_datetime(published, utc=True)  # assume UTC if naive
                ts = (dt if dt.tzinfo else dt.tz_localize("UTC")).tz_convert(IST)
            except Exception:
                ts = datetime.now(tz=IST)

            # Combined text for analysis
            text = f"{title}. {summary}".lower()

            # Event type classification
            event_type = "general"
            if any(k in text for k in ["earnings", "result", "profit", "revenue"]):
                event_type = "earnings"
            elif any(k in text for k in ["rbi", "policy", "rate", "inflation"]):
                event_type = "policy"
            elif any(k in text for k in ["merger", "acquisition", "deal"]):
                event_type = "m_and_a"
            elif any(k in text for k in ["regulation", "sebi", "compliance"]):
                event_type = "regulatory"

            # Basic sentiment scoring
            positive_words = ["gain", "rise", "growth", "beat", "upgrade", "strong"]
            negative_words = [
                "fall",
                "loss",
                "decline",
                "miss",
                "downgrade",
                "concern",
                "weak",
            ]

            pos_count = sum(1 for w in positive_words if w in text)
            neg_count = sum(1 for w in negative_words if w in text)

            sentiment = 0.2 * pos_count - 0.2 * neg_count
            sentiment = float(max(-0.8, min(0.8, sentiment)))

            signal = {
                "ts": ts.isoformat(),
                "type": event_type,
                "entities": item.get(
                    "entities", [source]
                ),  # Allow caller to pass tickers
                "sentiment": round(sentiment, 2),
                "surprise": item.get("surprise", 0.0),
                "horizon": "days",
                "confidence": float(item.get("confidence", 0.7)),
            }

            signals.append(signal)

        return signals

    def optimize_strategy_structured(
        self,
        market_data: pd.DataFrame,
        strategy_config: Dict,
        performance_metrics: Dict,
        regime_data: pd.DataFrame = None,
        news_data: List[Dict] = None,
    ) -> Dict:
        """
        Enhanced strategy optimization using structured signals

        Args:
            market_data: OHLCV price data
            strategy_config: Strategy configuration with knobs and constraints
            performance_metrics: Current performance metrics
            regime_data: Regime detection results (optional)
            news_data: Recent news data (optional)

        Returns:
            Structured optimization recommendations
        """

        # Build compressed data bundle
        optimization_data = {
            "meta": {
                "universe": strategy_config.get("universe", "NIFTY50"),
                "timeframe": strategy_config.get("timeframe", "15m"),
                "currency": "INR",
                "cost_model": "india_equities_v1",
                "objective": strategy_config.get("objective", "maximize_sortino"),
                "constraints": strategy_config.get(
                    "constraints",
                    {"max_dd": 0.2, "turnover_pa": 2.0, "risk_per_trade": 0.0075},
                ),
                "train_range": strategy_config.get(
                    "train_range", "2019-01-01/2023-12-31"
                ),
                "test_range": strategy_config.get(
                    "test_range", "2024-01-01/2025-08-15"
                ),
            },
            "market_stats": {"index": self._compute_market_stats(market_data)},
            "regime": self._extract_regime_diagnostics(market_data, regime_data),
            # Compact news early to keep prompts minimal
            "news_events": self._extract_news_signals(news_data)[:5]
            if news_data
            else [],
            "strategy": {
                "name": strategy_config.get("name", "unknown_strategy"),
                "pseudocode": strategy_config.get(
                    "description", "Strategy logic not provided"
                ),
                "knobs": strategy_config.get("knobs", {}),
                "invariants": strategy_config.get("invariants", []),
            },
            "performance": {
                "headline": {
                    "cagr": performance_metrics.get("total_return", 0) / 100,
                    "sharpe": performance_metrics.get("sharpe_ratio", 0),
                    "sortino": performance_metrics.get(
                        "sortino_ratio",
                        performance_metrics.get("sharpe_ratio", 0) * 1.2,
                    ),
                    "max_dd": abs(performance_metrics.get("max_drawdown", 0)) / 100,
                    "pf": performance_metrics.get("profit_factor", 1.0),
                    "hit": performance_metrics.get("win_rate", 50) / 100,
                    "avg_win": performance_metrics.get("avg_win", 0.01),
                    "avg_loss": performance_metrics.get("avg_loss", -0.01),
                    "exposure": performance_metrics.get("exposure", 0.6),
                    "turnover": performance_metrics.get("turnover", 1.5),
                    "trades": performance_metrics.get("total_trades", 100),
                },
                "by_slice": performance_metrics.get("regime_performance", {}),
                "fail_modes": performance_metrics.get(
                    "failure_modes", ["Insufficient data for failure mode analysis"]
                ),
            },
        }

        # Preflight check - don't waste tokens if inputs are empty
        preflight_result = self._preflight(optimization_data)
        if preflight_result:
            return {"meta": self._get_telemetry_meta(), **preflight_result}

        # Create structured prompt with compact JSON
        system_prompt = """You are a quantitative strategy optimization expert specializing in Indian equity markets.

CRITICAL INSTRUCTIONS:
- Return ONLY valid JSON with the exact structure requested
- Limit to ≤3 changes per iteration
- Provide parameter RANGES, not single values
- Include specific test plans for validation
- Consider transaction costs and market microstructure
- Account for regime-dependent performance
- No personal investment advice. Use only provided numbers; do not infer current market conditions.

Focus on:
- Risk-adjusted returns optimization
- Drawdown reduction techniques
- Market regime adaptation
- Transaction cost mitigation
- Overfitting prevention

If any of market_stats, regime, strategy.knobs, or performance.headline is empty → set ok=false."""

        schema_hint = """{
  "ok": boolean,
  "issues": string[],
  "changes": [{
    "area": "gate|entry|exit|sizing|timing|portfolio|universe|risk",
    "param": string,
    "new_range": [number,number],
    "why": string,
    "expected_effect": string
  }],
  "risks": string[],
  "test_plan": [{
    "name": string,
    "desc": string,
    "metric": string
  }]
}"""

        # Use compact JSON to save tokens
        ctx_json = self._compact(optimization_data)

        prompt = f"""Context JSON (compact, IST timestamps):
{ctx_json}

Schema (informal): {schema_hint}

Return ONLY JSON. No markdown, no prose."""

        # Make request with JSON mode and validation
        raw = self._make_request(prompt, system_prompt, json_mode=True)

        if raw.startswith("__LLM_ERROR__"):
            return {"meta": self._get_telemetry_meta(), "ok": False, "error": raw}

        # Extract and parse JSON
        resp_text = self._extract_json(raw)
        try:
            result = json.loads(resp_text)
        except Exception as e:
            return {
                "meta": self._get_telemetry_meta(),
                "ok": False,
                "error": f"JSON parse error: {e}",
                "raw_response": raw,
            }

        # Validate response structure
        required_keys = {"ok", "issues", "changes", "risks", "test_plan"}
        if not required_keys.issubset(result.keys()):
            return {
                "meta": self._get_telemetry_meta(),
                "ok": False,
                "error": "Invalid response structure",
                "raw_response": result,
            }

        # Clamp and enforce change budget
        result = self._clamp_changes(
            result, optimization_data["strategy"].get("knobs", {})
        )

        # Post-validate constraints and areas
        result = self._postvalidate(
            result,
            optimization_data["strategy"].get("knobs", {}),
            optimization_data["meta"].get("constraints", {}),
            optimization_data["strategy"].get("invariants", []),
        )

        # Return with telemetry metadata
        return {"meta": self._get_telemetry_meta(), **result}

    def analyze_query(self, query: str) -> str:
        """Analyze a general trading query"""

        system_prompt = """You are an expert algorithmic trading advisor with deep knowledge of:
        - Market microstructure and regime detection
        - Quantitative trading strategies
        - Risk management and portfolio optimization
        - Technical and fundamental analysis
        - Indian stock markets (NSE/BSE)

        Provide practical, actionable insights for algorithmic trading. Focus on:
        - Data-driven approaches
        - Risk-adjusted returns
        - Market regime considerations
        - Implementation details

        Keep responses concise but comprehensive."""

        return self._make_request(query, system_prompt)

    def analyze_market_regime(self) -> str:
        """Analyze current market regime"""

        prompt = """Based on recent market conditions in Indian equity markets:

        1. Analyze the current market regime (trending, ranging, volatile)
        2. Identify key technical and fundamental drivers
        3. Suggest optimal trading strategies for this regime
        4. Highlight risk factors to monitor
        5. Recommend position sizing and risk management approaches

        Focus on actionable insights for the current market environment."""

        system_prompt = """You are a market regime analysis expert. Analyze market conditions using:
        - Price action patterns
        - Volatility clusters
        - Volume analysis
        - Sector rotation
        - Macroeconomic factors

        Provide specific strategy recommendations."""

        return self._make_request(prompt, system_prompt)

    def optimize_strategies(self) -> str:
        """Provide strategy optimization suggestions"""

        prompt = """Analyze and optimize algorithmic trading strategies considering:

        1. Current market regime and volatility
        2. Strategy performance metrics and drawdowns
        3. Risk-adjusted returns optimization
        4. Position sizing and capital allocation
        5. Entry/exit timing improvements
        6. Multi-timeframe analysis integration

        Suggest specific improvements and parameter adjustments."""

        system_prompt = """You are a quantitative strategy optimization expert. Focus on:
        - Sharpe ratio and risk-adjusted metrics
        - Maximum drawdown minimization
        - Strategy diversification
        - Parameter optimization techniques
        - Overfitting prevention

        Provide actionable optimization recommendations."""

        return self._make_request(prompt, system_prompt)

    def assess_risk(self) -> str:
        """Assess portfolio and strategy risks"""

        prompt = """Conduct a comprehensive risk assessment covering:

        1. Portfolio concentration and diversification
        2. Strategy correlation and redundancy
        3. Market risk exposure (beta, sector, style)
        4. Liquidity and operational risks
        5. Tail risk and black swan scenarios
        6. Risk mitigation strategies

        Provide specific risk management recommendations."""

        system_prompt = """You are a quantitative risk management expert. Analyze:
        - Value at Risk (VaR) and Expected Shortfall
        - Stress testing scenarios
        - Correlation analysis
        - Risk budgeting
        - Dynamic hedging strategies

        Focus on practical risk management solutions."""

        return self._make_request(prompt, system_prompt)

    def analyze_news_sentiment(self, news_data: List[Dict]) -> str:
        """Analyze news sentiment and market impact"""

        news_text = "\n".join(
            [
                f"- {item.get('title', '')}: {item.get('summary', '')}"
                for item in news_data[:10]
            ]
        )

        prompt = f"""Analyze the following recent market news for trading insights:

        {news_text}

        Provide:
        1. Overall market sentiment (bullish/bearish/neutral)
        2. Sector-specific impacts
        3. Short-term vs long-term implications
        4. Trading opportunities and risks
        5. Key events to monitor"""

        system_prompt = """You are a news sentiment analysis expert for trading. Focus on:
        - Market moving events
        - Sector rotation implications
        - Risk-on vs risk-off sentiment
        - Policy and regulatory impacts

        Provide actionable trading insights."""

        return self._make_request(prompt, system_prompt)

    def generate_strategy_ideas(self, market_data: Dict) -> str:
        """Generate new strategy ideas based on market data"""

        prompt = f"""Based on current market conditions, generate innovative trading strategy ideas:

        Market Context:
        - Volatility: {market_data.get('volatility', 'N/A')}
        - Trend: {market_data.get('trend', 'N/A')}
        - Volume: {market_data.get('volume', 'N/A')}

        Generate 3-5 strategy concepts with:
        1. Strategy logic and signals
        2. Target timeframe and markets
        3. Risk management approach
        4. Expected performance characteristics
        5. Implementation considerations"""

        system_prompt = """You are a quantitative strategy developer. Create innovative strategies using:
        - Multi-factor models
        - Alternative data sources
        - Machine learning techniques
        - Cross-asset correlations

        Focus on novel, implementable ideas."""

        return self._make_request(prompt, system_prompt)

    def build_strategy_manifest(self, strategy_type: str, params: Dict) -> Dict:
        """Build a strategy manifest for AI optimization"""

        manifests = {
            "SMA Crossover": {
                "name": "sma_crossover",
                "description": "Long when fast SMA > slow SMA; exit when fast SMA < slow SMA or stop loss triggered",
                "knobs": {
                    "sma_fast": [5, 30],
                    "sma_slow": [20, 100],
                    "stop_loss_pct": [0.02, 0.05],
                    "risk_per_trade": [0.005, 0.02],
                },
                "invariants": [
                    "sma_fast < sma_slow",
                    "risk_per_trade <= 2%",
                    "no_trades_first_last_5min",
                ],
            },
            "Mean Reversion": {
                "name": "mean_reversion_bb_rsi",
                "description": "Long when price < BB_lower AND RSI < oversold; exit when price > BB_upper OR RSI > overbought",
                "knobs": {
                    "bb_period": [15, 30],
                    "bb_std": [1.5, 2.5],
                    "rsi_period": [10, 20],
                    "rsi_oversold": [20, 35],
                    "rsi_overbought": [65, 80],
                    "risk_per_trade": [0.005, 0.015],
                },
                "invariants": [
                    "rsi_oversold < 50 < rsi_overbought",
                    "bb_std >= 1.0",
                    "risk_per_trade <= 1.5%",
                ],
            },
            "Regime-Aware Strategy": {
                "name": "regime_adaptive",
                "description": "Adapts strategy based on market regime: trend-following in bull markets, mean reversion in sideways, defensive in bear/high-vol",
                "knobs": {
                    "trend_sma_fast": [8, 20],
                    "trend_sma_slow": [30, 60],
                    "mean_bb_period": [15, 25],
                    "mean_rsi_period": [10, 18],
                    "regime_lookback": [30, 90],
                    "vol_threshold": [0.015, 0.035],
                    "risk_per_trade": [0.003, 0.012],
                },
                "invariants": [
                    "regime_lookback >= 20",
                    "trend_sma_fast < trend_sma_slow",
                    "max_position_size <= 20%",
                ],
            },
        }

        manifest = manifests.get(
            strategy_type,
            {
                "name": "custom_strategy",
                "description": "Custom strategy - please provide description",
                "knobs": params,
                "invariants": [],
            },
        )

        # Override with actual parameters if provided
        if params:
            current_knobs = {}
            for param, value in params.items():
                if isinstance(value, (int, float)):
                    # Create range around current value
                    range_pct = 0.3  # 30% range around current value
                    min_val = value * (1 - range_pct)
                    max_val = value * (1 + range_pct)
                    current_knobs[param] = [round(min_val, 4), round(max_val, 4)]
                else:
                    current_knobs[param] = value

            manifest["knobs"].update(current_knobs)

        return manifest

    def analyze_performance_slices(self, backtest_results: pd.DataFrame) -> Dict:
        """Analyze performance by different slices for AI optimization"""

        if backtest_results.empty:
            return {
                "headline": {},
                "by_slice": {},
                "fail_modes": ["Insufficient data for analysis"],
            }

        # Create copy to avoid mutating input DataFrame
        results_copy = backtest_results.copy()
        returns = results_copy["Portfolio_Value"].pct_change().dropna()

        # Calculate headline metrics
        total_return = (
            results_copy["Portfolio_Value"].iloc[-1]
            / results_copy["Portfolio_Value"].iloc[0]
            - 1
        )
        trading_days = len(results_copy)
        years = trading_days / 252

        headline = {
            "cagr": round((1 + total_return) ** (1 / years) - 1 if years > 0 else 0, 3),
            "sharpe": round(
                returns.mean() / returns.std() * np.sqrt(252)
                if returns.std() > 0
                else 0,
                2,
            ),
            "max_dd": round(
                self._calculate_max_drawdown(results_copy["Portfolio_Value"]), 3
            ),
            "hit": round((returns > 0).mean(), 2),
            "avg_win": round(
                returns[returns > 0].mean() if len(returns[returns > 0]) > 0 else 0, 4
            ),
            "avg_loss": round(
                returns[returns < 0].mean() if len(returns[returns < 0]) > 0 else 0, 4
            ),
            "trades": len(results_copy[results_copy["Signal"].isin(["BUY", "SELL"])]),
        }

        # Performance by weekday
        by_slice = {}
        if "Date" in results_copy.columns or results_copy.index.name == "Date":
            if "Date" in results_copy.columns:
                dates = pd.to_datetime(results_copy["Date"])
            else:
                dates = results_copy.index

            results_copy = results_copy.assign(Weekday=dates.dt.day_name())
            weekday_performance = {}

            for day in results_copy["Weekday"].unique():
                day_data = results_copy[results_copy["Weekday"] == day]
                day_returns = day_data["Portfolio_Value"].pct_change().dropna()
                if len(day_returns) > 0:
                    pos_sum = day_returns[day_returns > 0].sum()
                    neg_sum = day_returns[day_returns < 0].sum()

                    # Guard against division by zero
                    # NOTE: This PF uses returns as proxy, not actual trade P&L
                    # For true PF, use gross_profit / gross_loss from trade ledger
                    if abs(neg_sum) < 1e-10:
                        profit_factor = 1.0 if pos_sum < 1e-10 else float("inf")
                    else:
                        profit_factor = abs(pos_sum / neg_sum)

                    weekday_performance[day] = {
                        "pf": round(min(profit_factor, 999.9), 2)
                    }

            by_slice["weekday"] = weekday_performance

        # Volatility tercile analysis
        vol_terciles = {}
        if len(returns) > 30:
            vol_rolling = returns.rolling(20).std()
            tercile_cutoffs = vol_rolling.quantile([0.33, 0.67])

            low_vol_returns = returns[vol_rolling <= tercile_cutoffs.iloc[0]]
            high_vol_returns = returns[vol_rolling >= tercile_cutoffs.iloc[1]]

            if len(low_vol_returns) > 0 and low_vol_returns.std() > 0:
                vol_terciles["low"] = {
                    "sharpe": round(
                        low_vol_returns.mean() / low_vol_returns.std() * np.sqrt(252), 2
                    )
                }
            if len(high_vol_returns) > 0 and high_vol_returns.std() > 0:
                vol_terciles["high"] = {
                    "sharpe": round(
                        high_vol_returns.mean() / high_vol_returns.std() * np.sqrt(252),
                        2,
                    )
                }

            by_slice["vol_tercile"] = vol_terciles

        # Failure mode analysis
        fail_modes = []

        # Find worst drawdown periods
        dd_series = self._calculate_drawdown_series(results_copy["Portfolio_Value"])
        worst_dd_idx = dd_series.idxmin()
        if worst_dd_idx is not None:
            fail_modes.append(
                f"Worst drawdown: {round(dd_series.min()*100, 1)}% around period {worst_dd_idx}"
            )

        # Consecutive losses
        consecutive_losses = self._find_max_consecutive_losses(returns)
        if consecutive_losses > 3:
            fail_modes.append(f"Max consecutive losses: {consecutive_losses}")

        if not fail_modes:
            fail_modes = ["No significant failure patterns detected"]

        return {"headline": headline, "by_slice": by_slice, "fail_modes": fail_modes}

    def _calculate_max_drawdown(self, portfolio_values: pd.Series) -> float:
        """Calculate maximum drawdown"""
        peak = portfolio_values.expanding().max()
        drawdown = (portfolio_values - peak) / peak
        return abs(drawdown.min())

    def _calculate_drawdown_series(self, portfolio_values: pd.Series) -> pd.Series:
        """Calculate drawdown series"""
        peak = portfolio_values.expanding().max()
        return (portfolio_values - peak) / peak

    def _find_max_consecutive_losses(self, returns: pd.Series) -> int:
        """Find maximum consecutive losing periods"""
        losses = returns < 0
        consecutive = 0
        max_consecutive = 0

        for loss in losses:
            if loss:
                consecutive += 1
                max_consecutive = max(max_consecutive, consecutive)
            else:
                consecutive = 0

        return max_consecutive

    def _apply_llm_changes(self, current_knobs: dict, changes: list) -> dict:
        """Move each changed knob toward the midpoint of the suggested range."""
        new_knobs = deepcopy(current_knobs)
        for ch in changes:
            p = ch.get("param")
            lo, hi = ch.get("new_range", [None, None])
            if (
                p in new_knobs
                and isinstance(new_knobs[p], (list, tuple))
                and lo is not None
                and hi is not None
            ):
                # If knob stores an allowed range, keep it. If it stores a point value,
                # set to midpoint.
                if isinstance(new_knobs[p], (int, float)):
                    new_knobs[p] = round((lo + hi) / 2, 6)
                else:
                    # Keep allowed range but bias a suggested "current" value you store
                    # elsewhere if needed.
                    pass
            elif p in new_knobs and isinstance(new_knobs[p], (int, float)):
                new_knobs[p] = round((lo + hi) / 2, 6)
        return new_knobs

    def iterate_improvement(
        self,
        run_backtest_fn,
        # callable: (strategy_config) -> (equity_df, perf_metrics_dict)
        strategy_config: dict,
        market_data: pd.DataFrame,
        regime_data: Optional[pd.DataFrame] = None,
        news_data: Optional[list] = None,
        max_iters: int = 6,
        min_oos_gain_pct: float = 10.0,
        # promotion threshold on objective (e.g., Sortino)
        drawdown_tolerance_pp: float = 2.0,
    ) -> dict:
        """
        Closed-loop iterative strategy optimization with AI-guided parameter tuning.

        Returns a history dict with each iteration's config, metrics, LLM suggestion, and decision.
        Assumes run_backtest_fn applies costs and returns (equity_df, perf_metrics).

        Args:
            run_backtest_fn: Function that takes strategy_config and returns (equity_df, perf_metrics_dict)
            strategy_config: Initial strategy configuration with knobs
            market_data: OHLCV price data for analysis
            regime_data: Optional regime detection results
            news_data: Optional recent news data
            max_iters: Maximum optimization iterations
            min_oos_gain_pct: Minimum % gain on objective to promote challenger
            drawdown_tolerance_pp: Maximum additional drawdown in percentage points

        Returns:
            dict: {
                "champion_config": final best configuration,
                "champion_perf": final best performance metrics,
                "champion_slices": final performance analysis,
                "iterations": list of iteration history
            }
        """
        history = []
        champion_cfg = deepcopy(strategy_config)
        champion_eq, champion_perf = run_backtest_fn(champion_cfg)
        champion_slices = self.analyze_performance_slices(champion_eq)
        champion_obj = champion_perf.get(
            "sortino_ratio", champion_perf.get("sharpe_ratio", 0.0)
        )
        champion_dd = abs(champion_perf.get("max_drawdown", 0.0))

        current_cfg = deepcopy(champion_cfg)

        for it in range(1, max_iters + 1):
            # 1) Backtest current configuration
            eq, perf = run_backtest_fn(current_cfg)
            slices = self.analyze_performance_slices(eq)

            # 2) Ask LLM for bounded changes using structured optimization
            suggestion = self.optimize_strategy_structured(
                market_data=market_data,
                strategy_config=current_cfg,
                performance_metrics=perf,
                regime_data=regime_data,
                news_data=news_data,
            )

            # If LLM couldn't produce a valid plan, stop optimization
            if not suggestion.get("ok", False) or not suggestion.get("changes"):
                history.append(
                    {
                        "iter": it,
                        "config": current_cfg,
                        "perf": perf,
                        "slices": slices,
                        "suggestion": suggestion,
                        "decision": "stop_no_valid_changes",
                    }
                )
                break

            # 3) Apply LLM suggested changes to create challenger
            new_cfg = deepcopy(current_cfg)
            new_cfg["knobs"] = self._apply_llm_changes(
                current_cfg.get("knobs", {}), suggestion["changes"]
            )

            # 4) Evaluate challenger configuration
            ch_eq, ch_perf = run_backtest_fn(new_cfg)
            ch_slices = self.analyze_performance_slices(ch_eq)
            ch_obj = ch_perf.get("sortino_ratio", ch_perf.get("sharpe_ratio", 0.0))
            ch_dd = abs(ch_perf.get("max_drawdown", 0.0))

            # 5) Promotion rule with risk controls
            gain_pct = 100.0 * (ch_obj - champion_obj) / max(1e-9, abs(champion_obj))
            dd_ok = (ch_dd * 100.0) <= ((champion_dd * 100.0) + drawdown_tolerance_pp)
            promote = (gain_pct >= min_oos_gain_pct) and dd_ok

            decision = "keep_champion"
            if promote:
                champion_cfg, champion_eq, champion_perf, champion_slices = (
                    new_cfg,
                    ch_eq,
                    ch_perf,
                    ch_slices,
                )
                champion_obj, champion_dd = ch_obj, ch_dd
                current_cfg = deepcopy(new_cfg)
                decision = "promoted_to_champion"
            else:
                # Continue iteration from challenger to explore local parameter space
                current_cfg = deepcopy(new_cfg)

            history.append(
                {
                    "iter": it,
                    "config_before": current_cfg,
                    "perf_before": perf,
                    "slices_before": slices,
                    "suggestion": suggestion,
                    "config_after": new_cfg,
                    "perf_after": ch_perf,
                    "slices_after": ch_slices,
                    "decision": decision,
                    "gain_pct_on_objective": round(gain_pct, 2),
                    "drawdown_pp": round(ch_dd * 100.0, 2),
                }
            )

            # Optional early stop: two consecutive non-promotions indicate local optimum
            if (
                it >= 2
                and history[-1]["decision"] != "promoted_to_champion"
                and history[-2]["decision"] != "promoted_to_champion"
            ):
                break

        return {
            "champion_config": champion_cfg,
            "champion_perf": champion_perf,
            "champion_slices": champion_slices,
            "iterations": history,
            "total_iterations": len(history),
            "final_objective": champion_obj,
            "final_drawdown_pct": round(champion_dd * 100.0, 2),
        }
