"""
Investment Bodyguard - Risk Detection Engine

LEARNING OBJECTIVES:
1. Time Series Analysis - Understanding financial data patterns
2. Volatility Modeling - Measuring and predicting risk
3. Anomaly Detection - Identifying unusual market behavior
4. Feature Engineering - Creating predictive indicators
5. Model Evaluation - Backtesting and validation

CONCEPTS YOU'LL MASTER:
- Rolling Statistics (Moving averages, volatility)
- Z-Score Analysis (Statistical outlier detection)
- GARCH Models (Volatility clustering)
- Technical Indicators (RSI, Bollinger Bands, MACD)
- Risk Metrics (VaR, Maximum Drawdown, Sharpe Ratio)
"""

import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from typing import Dict, List, Tuple, Optional
import warnings
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class RiskAlert:
    """Structure for risk alerts"""
    timestamp: datetime
    symbol: str
    risk_level: str  # 'LOW', 'MEDIUM', 'HIGH', 'CRITICAL'
    risk_score: float  # 0-100
    alert_type: str  # 'VOLATILITY', 'DRAWDOWN', 'ANOMALY', 'CORRELATION'
    message: str
    recommended_action: str

class RiskDetectionEngine:
    """
    Core AI engine for detecting portfolio risks
    
    LEARNING FOCUS:
    - How to process financial time series data
    - Mathematical concepts behind risk measurement
    - Machine learning for anomaly detection
    - Feature engineering for financial data
    """
    
    def __init__(self, lookback_days: int = 252):
        """
        Initialize the risk detection engine
        
        CONCEPT: Lookback Period
        - 252 = typical trading days in a year
        - This gives us 1 year of historical context
        - Critical for calculating annual volatility and risk metrics
        """
        self.lookback_days = lookback_days
        self.scaler = StandardScaler()
        self.anomaly_detector = IsolationForest(
            contamination=0.1,  # Expect 10% of data to be anomalies
            random_state=42
        )
        
        # Risk thresholds (you can customize these)
        self.risk_thresholds = {
            'volatility': {'low': 15, 'medium': 25, 'high': 40},  # Annualized %
            'drawdown': {'low': 5, 'medium': 10, 'high': 20},    # %
            'correlation': {'low': 0.3, 'medium': 0.6, 'high': 0.8}
        }
        
        logger.info("Risk Detection Engine initialized")
    
    def fetch_market_data(self, symbol: str, period: str = "1y") -> pd.DataFrame:
        """
        Fetch and prepare market data for analysis
        
        LEARNING: Data Preparation
        - Always use adjusted close prices (accounts for splits/dividends)
        - Calculate returns (percentage change in price)
        - Handle missing data appropriately
        """
        try:
            # Download data from Yahoo Finance
            ticker = yf.Ticker(symbol)
            data = ticker.history(period=period)
            
            if data.empty:
                raise ValueError(f"No data found for symbol {symbol}")
            
            # Calculate returns (daily percentage change)
            # CONCEPT: Returns are more stationary than prices
            data['Returns'] = data['Close'].pct_change()
            
            # Calculate log returns (for mathematical convenience)
            # CONCEPT: Log returns are additive and normally distributed
            data['Log_Returns'] = np.log(data['Close'] / data['Close'].shift(1))
            
            # Remove first row (NaN values from pct_change)
            data = data.dropna()
            
            logger.info(f"Fetched {len(data)} days of data for {symbol}")
            return data
            
        except Exception as e:
            logger.error(f"Error fetching data for {symbol}: {e}")
            return pd.DataFrame()
    
    def calculate_volatility_metrics(self, data: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate various volatility measures
        
        LEARNING: Volatility is the Key Risk Metric
        - Historical Volatility: Standard deviation of returns
        - Annualized Volatility: Scaled to yearly basis
        - Rolling Volatility: How volatility changes over time
        - GARCH: Advanced volatility modeling (we'll add this later)
        """
        returns = data['Returns'].dropna()
        
        # Daily volatility (standard deviation of returns)
        daily_vol = returns.std()
        
        # Annualized volatility (multiply by sqrt of trading days)
        # CONCEPT: Volatility scales with square root of time
        annual_vol = daily_vol * np.sqrt(252) * 100  # Convert to percentage
        
        # Rolling 30-day volatility
        rolling_vol = returns.rolling(window=30).std() * np.sqrt(252) * 100
        current_vol = rolling_vol.iloc[-1] if not rolling_vol.empty else 0
        
        # Volatility percentile (where current vol stands historically)
        vol_percentile = (rolling_vol < current_vol).mean() * 100
        
        return {
            'daily_volatility': daily_vol,
            'annual_volatility': annual_vol,
            'current_volatility': current_vol,
            'volatility_percentile': vol_percentile
        }
    
    def calculate_drawdown_metrics(self, data: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate drawdown metrics
        
        LEARNING: Drawdown Analysis
        - Drawdown = Peak-to-trough decline
        - Maximum Drawdown = Worst loss from any peak
        - Current Drawdown = How far we are from recent peak
        - Recovery Time = How long to get back to breakeven
        """
        prices = data['Close']
        
        # Calculate running maximum (peak prices)
        running_max = prices.expanding().max()
        
        # Calculate drawdown (negative values)
        drawdown = (prices - running_max) / running_max * 100
        
        # Maximum drawdown (most negative value)
        max_drawdown = drawdown.min()
        
        # Current drawdown
        current_drawdown = drawdown.iloc[-1]
        
        # Days since peak
        peak_idx = prices.idxmax()
        days_since_peak = (prices.index[-1] - peak_idx).days if isinstance(peak_idx, pd.Timestamp) else len(prices) - peak_idx
        
        return {
            'max_drawdown': max_drawdown,
            'current_drawdown': current_drawdown,
            'days_since_peak': days_since_peak
        }
    
    def calculate_technical_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate technical indicators for feature engineering
        
        LEARNING: Technical Analysis
        - RSI: Relative Strength Index (momentum)
        - Bollinger Bands: Volatility bands
        - MACD: Moving Average Convergence Divergence
        - These become features for our ML models
        """
        df = data.copy()
        
        # RSI (Relative Strength Index)
        # CONCEPT: Measures overbought/oversold conditions
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # Bollinger Bands
        # CONCEPT: Price tends to revert to mean, bands show volatility
        window = 20
        df['SMA_20'] = df['Close'].rolling(window=window).mean()
        std = df['Close'].rolling(window=window).std()
        df['BB_Upper'] = df['SMA_20'] + (std * 2)
        df['BB_Lower'] = df['SMA_20'] - (std * 2)
        df['BB_Position'] = (df['Close'] - df['BB_Lower']) / (df['BB_Upper'] - df['BB_Lower'])
        
        # MACD
        # CONCEPT: Shows relationship between two moving averages
        exp1 = df['Close'].ewm(span=12).mean()
        exp2 = df['Close'].ewm(span=26).mean()
        df['MACD'] = exp1 - exp2
        df['MACD_Signal'] = df['MACD'].ewm(span=9).mean()
        df['MACD_Histogram'] = df['MACD'] - df['MACD_Signal']
        
        # Volume indicators (if available)
        if 'Volume' in df.columns:
            df['Volume_SMA'] = df['Volume'].rolling(window=20).mean()
            df['Volume_Ratio'] = df['Volume'] / df['Volume_SMA']
        
        return df
    
    def engineer_risk_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Create advanced features for risk prediction
        
        LEARNING: Feature Engineering
        - This is where art meets science in ML
        - Good features often matter more than complex models
        - Financial domain knowledge is crucial here
        """
        df = data.copy()
        
        # Volatility features
        returns = df['Returns']
        
        # Rolling volatility at different windows
        for window in [5, 10, 20, 60]:
            df[f'Vol_{window}D'] = returns.rolling(window).std() * np.sqrt(252) * 100
        
        # Volatility rank (percentile over lookback period)
        current_vol = df['Vol_20D']
        df['Vol_Rank'] = current_vol.rolling(self.lookback_days).rank(pct=True) * 100
        
        # Return features
        for window in [1, 5, 10, 20]:
            df[f'Return_{window}D'] = df['Close'].pct_change(periods=window) * 100
        
        # Momentum features
        df['Momentum_1M'] = (df['Close'] / df['Close'].shift(20) - 1) * 100
        df['Momentum_3M'] = (df['Close'] / df['Close'].shift(60) - 1) * 100
        
        # Risk-adjusted returns (Sharpe-like measures)
        df['Risk_Adj_Return'] = df['Return_20D'] / df['Vol_20D']
        
        # Correlation with market (we'll use SPY as proxy)
        try:
            spy_data = yf.Ticker('SPY').history(period='1y')['Close']
            spy_returns = spy_data.pct_change()
            
            # Align dates
            aligned_returns = df['Returns'].reindex(spy_returns.index)
            correlation = aligned_returns.rolling(60).corr(spy_returns)
            df['Market_Correlation'] = correlation.reindex(df.index)
            
        except Exception as e:
            logger.warning(f"Could not calculate market correlation: {e}")
            df['Market_Correlation'] = 0.5  # Default moderate correlation
        
        return df
    
    def detect_anomalies(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Use machine learning to detect unusual market behavior
        
        LEARNING: Anomaly Detection
        - Isolation Forest: Finds data points that are "easy to isolate"
        - Unsupervised learning: No need for labeled training data
        - Perfect for finding unusual market conditions
        """
        df = data.copy()
        
        # Select features for anomaly detection
        feature_columns = [
            'Returns', 'Vol_20D', 'RSI', 'BB_Position', 
            'MACD', 'Return_5D', 'Risk_Adj_Return'
        ]
        
        # Only use rows where all features are available
        feature_data = df[feature_columns].dropna()
        
        if len(feature_data) < 50:  # Need minimum data for training
            logger.warning("Insufficient data for anomaly detection")
            df['Anomaly_Score'] = 0
            df['Is_Anomaly'] = False
            return df
        
        # Standardize features (important for ML algorithms)
        scaled_features = self.scaler.fit_transform(feature_data)
        
        # Train anomaly detector
        self.anomaly_detector.fit(scaled_features)
        
        # Get anomaly scores
        anomaly_scores = self.anomaly_detector.decision_function(scaled_features)
        anomaly_predictions = self.anomaly_detector.predict(scaled_features)
        
        # Add results back to dataframe
        df.loc[feature_data.index, 'Anomaly_Score'] = anomaly_scores
        df.loc[feature_data.index, 'Is_Anomaly'] = anomaly_predictions == -1
        
        # Fill missing values
        df['Anomaly_Score'] = df['Anomaly_Score'].fillna(0)
        df['Is_Anomaly'] = df['Is_Anomaly'].fillna(False)
        
        return df
    
    def generate_risk_score(self, data: pd.DataFrame) -> float:
        """
        Combine multiple risk factors into single score
        
        LEARNING: Risk Aggregation
        - Multiple risk factors need to be combined intelligently
        - Different weights for different risk types
        - Final score should be interpretable (0-100)
        """
        if data.empty:
            return 0
        
        latest = data.iloc[-1]
        
        # Volatility component (0-40 points)
        vol_score = min(latest.get('Vol_20D', 20) / 40 * 40, 40)
        
        # Drawdown component (0-30 points)
        drawdown_metrics = self.calculate_drawdown_metrics(data)
        drawdown_score = min(abs(drawdown_metrics['current_drawdown']) / 20 * 30, 30)
        
        # Technical indicator component (0-20 points)
        rsi = latest.get('RSI', 50)
        rsi_extreme = max(0, abs(rsi - 50) - 20) / 30 * 10  # Penalty for extreme RSI
        
        bb_position = latest.get('BB_Position', 0.5)
        bb_extreme = max(0, abs(bb_position - 0.5) - 0.3) / 0.2 * 10
        
        technical_score = rsi_extreme + bb_extreme
        
        # Anomaly component (0-10 points)
        anomaly_score = 10 if latest.get('Is_Anomaly', False) else 0
        
        # Combine scores
        total_score = vol_score + drawdown_score + technical_score + anomaly_score
        
        return min(total_score, 100)  # Cap at 100
    
    def analyze_symbol(self, symbol: str) -> Tuple[Dict, List[RiskAlert]]:
        """
        Complete risk analysis for a single symbol
        
        This is our main analysis pipeline that combines everything
        """
        logger.info(f"Starting risk analysis for {symbol}")
        
        # Fetch data
        data = self.fetch_market_data(symbol)
        if data.empty:
            return {}, []
        
        # Calculate technical indicators
        data = self.calculate_technical_indicators(data)
        
        # Engineer risk features
        data = self.engineer_risk_features(data)
        
        # Detect anomalies
        data = self.detect_anomalies(data)
        
        # Calculate metrics
        vol_metrics = self.calculate_volatility_metrics(data)
        drawdown_metrics = self.calculate_drawdown_metrics(data)
        risk_score = self.generate_risk_score(data)
        
        # Compile results
        analysis = {
            'symbol': symbol,
            'timestamp': datetime.now(),
            'risk_score': risk_score,
            'volatility_metrics': vol_metrics,
            'drawdown_metrics': drawdown_metrics,
            'current_price': data['Close'].iloc[-1],
            'latest_return': data['Returns'].iloc[-1] * 100,
            'rsi': data['RSI'].iloc[-1],
            'is_anomaly': data['Is_Anomaly'].iloc[-1]
        }
        
        # Generate alerts
        alerts = self._generate_alerts(symbol, analysis, data)
        
        logger.info(f"Risk analysis complete for {symbol}. Risk score: {risk_score:.1f}")
        return analysis, alerts
    
    def _generate_alerts(self, symbol: str, analysis: Dict, data: pd.DataFrame) -> List[RiskAlert]:
        """Generate risk alerts based on analysis"""
        alerts = []
        risk_score = analysis['risk_score']
        
        # High volatility alert
        if analysis['volatility_metrics']['annual_volatility'] > self.risk_thresholds['volatility']['high']:
            alerts.append(RiskAlert(
                timestamp=datetime.now(),
                symbol=symbol,
                risk_level='HIGH',
                risk_score=risk_score,
                alert_type='VOLATILITY',
                message=f"Extremely high volatility detected: {analysis['volatility_metrics']['annual_volatility']:.1f}%",
                recommended_action="Consider reducing position size or setting tighter stop-losses"
            ))
        
        # Drawdown alert
        if analysis['drawdown_metrics']['current_drawdown'] < -self.risk_thresholds['drawdown']['medium']:
            alerts.append(RiskAlert(
                timestamp=datetime.now(),
                symbol=symbol,
                risk_level='MEDIUM',
                risk_score=risk_score,
                alert_type='DRAWDOWN',
                message=f"Significant drawdown: {analysis['drawdown_metrics']['current_drawdown']:.1f}%",
                recommended_action="Monitor closely, consider exit if drawdown worsens"
            ))
        
        # Anomaly alert
        if analysis['is_anomaly']:
            alerts.append(RiskAlert(
                timestamp=datetime.now(),
                symbol=symbol,
                risk_level='HIGH',
                risk_score=risk_score,
                alert_type='ANOMALY',
                message="Unusual market behavior detected by AI model",
                recommended_action="Exercise extra caution, review position sizing"
            ))
        
        # Critical risk score alert
        if risk_score > 80:
            alerts.append(RiskAlert(
                timestamp=datetime.now(),
                symbol=symbol,
                risk_level='CRITICAL',
                risk_score=risk_score,
                alert_type='OVERALL_RISK',
                message=f"Critical risk level: {risk_score:.1f}/100",
                recommended_action="Strong recommendation to reduce exposure immediately"
            ))
        
        return alerts

# Example usage and testing
if __name__ == "__main__":
    print("🎓 LEARNING SESSION: Building Risk Detection Engine")
    print("="*60)
    
    # Initialize the engine
    engine = RiskDetectionEngine()
    
    # Test with a popular stock
    symbol = "AAPL"
    print(f"📊 Analyzing {symbol} to demonstrate concepts...")
    
    try:
        analysis, alerts = engine.analyze_symbol(symbol)
        
        print(f"\n📈 Analysis Results for {symbol}:")
        print(f"Risk Score: {analysis['risk_score']:.1f}/100")
        print(f"Current Price: ${analysis['current_price']:.2f}")
        print(f"Annual Volatility: {analysis['volatility_metrics']['annual_volatility']:.1f}%")
        print(f"Max Drawdown: {analysis['drawdown_metrics']['max_drawdown']:.1f}%")
        print(f"RSI: {analysis['rsi']:.1f}")
        print(f"Anomaly Detected: {analysis['is_anomaly']}")
        
        if alerts:
            print(f"\n🚨 {len(alerts)} Risk Alerts Generated:")
            for alert in alerts:
                print(f"- {alert.alert_type}: {alert.message}")
                print(f"  Action: {alert.recommended_action}")
        else:
            print("\n✅ No significant risk alerts")
            
    except Exception as e:
        print(f"❌ Error during analysis: {e}")
    
    print("\n" + "="*60)
    print("🎯 CONCEPTS LEARNED:")
    print("✅ Time series data processing")
    print("✅ Volatility calculation and interpretation") 
    print("✅ Drawdown analysis")
    print("✅ Technical indicators (RSI, Bollinger Bands, MACD)")
    print("✅ Feature engineering for financial data")
    print("✅ Anomaly detection with machine learning")
    print("✅ Risk score aggregation")
    print("✅ Alert generation logic")
    print("\n🚀 Next: We'll build the portfolio monitoring system!")
