import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.mixture import GaussianMixture

class RegimeDetector:
    """Detect market regimes using various statistical and ML methods"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.scaler = StandardScaler()
    
    def detect_volatility_regimes(self, data: pd.DataFrame, window: int = 30, 
                                n_regimes: int = 3, use_walkforward: bool = True,
                                train_window: int = 750, test_window: int = 60) -> pd.DataFrame:
        """
        Detect volatility regimes using rolling volatility with walk-forward approach
        
        Args:
            data: DataFrame with price data
            window: Rolling window for volatility calculation
            n_regimes: Number of volatility regimes
            use_walkforward: If True, use walk-forward clustering to prevent look-ahead bias
            train_window: Training window size for walk-forward
            test_window: Test window size for walk-forward
            
        Returns:
            DataFrame with volatility regime labels
        """
        try:
            # Calculate rolling volatility
            returns = data['Close'].pct_change()
            volatility = returns.rolling(window=window).std() * np.sqrt(252)  # Annualized
            
            # Remove NaN values
            vol_data = volatility.dropna()
            
            if len(vol_data) < window:
                self.logger.warning("Insufficient data for regime detection")
                return data
            
            if use_walkforward and len(vol_data) >= train_window + test_window:
                # Walk-forward clustering to prevent look-ahead bias
                regime_labels = self._kmeans_vol_regimes_walkforward(
                    vol_data, n_regimes, train_window, test_window
                )
            else:
                # Fallback to rolling quantile method (safer for small datasets)
                regime_labels = self._rolling_quantile_regimes(vol_data, n_regimes)
            
            # Add to original data
            result_df = data.copy()
            result_df['Volatility'] = volatility
            result_df['Volatility_Regime'] = regime_labels
            
            # Forward fill regime labels
            result_df['Volatility_Regime'] = result_df['Volatility_Regime'].ffill()
            
            self.logger.info(f"Detected {n_regimes} volatility regimes using {'walk-forward' if use_walkforward else 'quantile'} method")
            return result_df
            
        except Exception as e:
            self.logger.error(f"Error detecting volatility regimes: {e}")
            return data
    
    def detect_trend_regimes(self, data: pd.DataFrame, ema_period: int = 200, 
                           lookback: int = 20, r2_window: int = 80,
                           r2_threshold: float = 0.30) -> pd.DataFrame:
        """
        Detect trend regimes using EMA slope and R-squared linearity test
        
        Args:
            data: DataFrame with price data
            ema_period: EMA period for trend detection
            lookback: Lookback period for slope calculation
            r2_window: Window for R-squared calculation
            r2_threshold: Minimum R-squared for trend confirmation
            
        Returns:
            DataFrame with trend regime labels
        """
        try:
            # Calculate EMA and its slope
            ema = data['Close'].ewm(span=ema_period, adjust=False).mean()
            ema_slope = (ema - ema.shift(lookback)) / lookback  # price/day slope
            
            # Calculate R-squared for trend strength
            r2_strength = self._rolling_r2_logprice(data['Close'], r2_window)
            
            # Price position relative to EMA
            price_above_ema = data['Close'] > ema
            
            # Define regime conditions with hysteresis
            slope_positive = ema_slope > 0
            trend_strong = r2_strength > r2_threshold
            
            # Create trend regime labels
            conditions = [
                price_above_ema & slope_positive & trend_strong,  # Strong Uptrend
                price_above_ema & slope_positive & ~trend_strong,  # Weak Uptrend
                price_above_ema & ~slope_positive,                # Uptrend Weakening
                ~price_above_ema & ~slope_positive & trend_strong, # Strong Downtrend
                ~price_above_ema & ~slope_positive & ~trend_strong, # Weak Downtrend
                ~price_above_ema & slope_positive                  # Downtrend Weakening
            ]
            
            choices = [
                'Strong_Uptrend', 'Weak_Uptrend', 'Uptrend_Weakening',
                'Strong_Downtrend', 'Weak_Downtrend', 'Downtrend_Weakening'
            ]
            
            trend_regime = np.select(conditions, choices, default='Sideways')
            
            # Add to data
            result_df = data.copy()
            result_df['EMA'] = ema
            result_df['EMA_Slope'] = ema_slope
            result_df['R2_Strength'] = r2_strength
            result_df['Trend_Regime'] = trend_regime
            
            self.logger.info("Trend regimes detected successfully with EMA slope method")
            return result_df
            
        except Exception as e:
            self.logger.error(f"Error detecting trend regimes: {e}")
            return data
    
    def detect_market_state_regimes(self, data: pd.DataFrame, window: int = 30,
                                  use_gmm: bool = True, train_window: int = 750,
                                  test_window: int = 60) -> pd.DataFrame:
        """
        Detect comprehensive market state regimes using multiple factors with walk-forward approach
        
        Args:
            data: DataFrame with OHLCV data
            window: Window for calculations
            use_gmm: If True, use Gaussian Mixture Model for probabilistic clustering
            train_window: Training window size for walk-forward
            test_window: Test window size for walk-forward
            
        Returns:
            DataFrame with market state regime labels
        """
        try:
            # Calculate various market indicators with safer computations
            returns = data['Close'].pct_change()
            volatility = returns.rolling(window=window).std()
            
            # Price momentum
            momentum = data['Close'].pct_change(window)
            
            # Volume ratio with safety guards
            eps = 1e-6
            avg_volume = data['Volume'].rolling(window=window).mean().clip(lower=eps)
            volume_ratio = (data['Volume'] / avg_volume).clip(0, 10)  # Clip outliers
            
            # Corrected RSI using Wilder's smoothing
            rsi = self._calculate_rsi_wilder(data, period=14)
            
            # Combine features for regime detection
            features_df = pd.DataFrame({
                'returns': returns,
                'volatility': volatility,
                'momentum': momentum,
                'volume_ratio': volume_ratio,
                'rsi': rsi
            }).dropna()
            
            if len(features_df) < max(window * 2, 100):  # Minimum data requirement
                self.logger.warning("Insufficient data for market state regime detection")
                return data
            
            # Walk-forward regime detection
            if len(features_df) >= train_window + test_window:
                if use_gmm:
                    regime_labels, regime_probs = self._gmm_market_regimes_walkforward(
                        features_df, n_components=4, train_window=train_window, test_window=test_window
                    )
                else:
                    regime_labels = self._kmeans_market_regimes_walkforward(
                        features_df, n_clusters=4, train_window=train_window, test_window=test_window
                    )
                    regime_probs = None
            else:
                # Fallback for small datasets
                regime_labels = self._simple_regime_classification(features_df)
                regime_probs = None
            
            # Add to original data
            result_df = data.copy()
            
            # Initialize with NaN and align indices properly
            result_df['Market_State_Regime'] = pd.Series(dtype=object, index=data.index)
            
            # Add regime labels where available
            common_index = result_df.index.intersection(regime_labels.index)
            result_df.loc[common_index, 'Market_State_Regime'] = regime_labels.loc[common_index]
            
            # Forward fill regime labels
            result_df['Market_State_Regime'] = result_df['Market_State_Regime'].ffill()
            
            # Add probabilities if available
            if regime_probs is not None and len(regime_probs) > 0:
                prob_index = regime_labels.dropna().index
                for i, regime_name in enumerate(['Bull_Market', 'Bear_Market', 'High_Volatility', 'Sideways_Market']):
                    if i < regime_probs.shape[1]:
                        prob_series = pd.Series(dtype=float, index=data.index)
                        if len(prob_index) <= len(regime_probs):
                            prob_series.loc[prob_index] = regime_probs[:len(prob_index), i]
                        result_df[f'{regime_name}_Prob'] = prob_series
            
            self.logger.info("Market state regimes detected successfully with walk-forward approach")
            return result_df
            
        except Exception as e:
            self.logger.error(f"Error detecting market state regimes: {e}")
            return data
    
    def _calculate_rsi_wilder(self, data: pd.DataFrame, period: int = 14) -> pd.Series:
        """
        Calculate RSI using Wilder's smoothing (correct method)
        
        Args:
            data: DataFrame with price data
            period: RSI period
            
        Returns:
            RSI series
        """
        try:
            delta = data['Close'].diff()
            up = delta.clip(lower=0)
            down = -delta.clip(upper=0)
            
            # Wilder's EMA (alpha = 1/period)
            gain = up.ewm(alpha=1/period, adjust=False).mean()
            loss = down.ewm(alpha=1/period, adjust=False).mean()
            
            # Avoid division by zero
            rs = gain / loss.replace(0, np.nan)
            rsi = 100 - (100 / (1 + rs))
            
            # Fill NaN values with neutral RSI (50)
            return rsi.fillna(50)
            
        except Exception as e:
            self.logger.error(f"Error calculating RSI: {e}")
            return pd.Series(50, index=data.index)  # Return neutral RSI
    
    def _calculate_rsi(self, data: pd.DataFrame, window: int = 14) -> pd.Series:
        """Legacy RSI calculation - kept for backward compatibility"""
        return self._calculate_rsi_wilder(data, window)
    
    def detect_regime_changes(self, regime_series: pd.Series) -> List[Dict]:
        """
        Detect regime change points
        
        Args:
            regime_series: Series with regime labels
            
        Returns:
            List of regime change events
        """
        try:
            changes = []
            current_regime = None
            
            for date, regime in regime_series.items():
                if regime != current_regime and current_regime is not None:
                    change_event = {
                        'date': date,
                        'from_regime': current_regime,
                        'to_regime': regime,
                        'change_type': self._classify_regime_change(current_regime, regime)
                    }
                    changes.append(change_event)
                
                current_regime = regime
            
            self.logger.info(f"Detected {len(changes)} regime changes")
            return changes
            
        except Exception as e:
            self.logger.error(f"Error detecting regime changes: {e}")
            return []
    
    def _classify_regime_change(self, from_regime: str, to_regime: str) -> str:
        """Classify the type of regime change"""
        # Define regime change classifications
        if 'Bull' in from_regime and 'Bear' in to_regime:
            return 'Bull_to_Bear'
        elif 'Bear' in from_regime and 'Bull' in to_regime:
            return 'Bear_to_Bull'
        elif 'Low_Vol' in from_regime and 'High_Vol' in to_regime:
            return 'Vol_Spike'
        elif 'High_Vol' in from_regime and 'Low_Vol' in to_regime:
            return 'Vol_Calm'
        else:
            return 'Other'
    
    def get_regime_statistics(self, data: pd.DataFrame, regime_column: str) -> Dict:
        """
        Calculate statistics for each regime
        
        Args:
            data: DataFrame with regime labels
            regime_column: Name of the regime column
            
        Returns:
            Dictionary with regime statistics
        """
        try:
            stats = {}
            
            for regime in data[regime_column].unique():
                if pd.isna(regime):
                    continue
                
                regime_data = data[data[regime_column] == regime]
                returns = regime_data['Close'].pct_change().dropna()
                
                regime_stats = {
                    'duration_days': len(regime_data),
                    'avg_return': returns.mean(),
                    'volatility': returns.std(),
                    'max_drawdown': self._calculate_max_drawdown(regime_data['Close']),
                    'sharpe_ratio': returns.mean() / returns.std() if returns.std() > 0 else 0,
                    'frequency': len(regime_data) / len(data)
                }
                
                stats[regime] = regime_stats
            
            return stats
            
        except Exception as e:
            self.logger.error(f"Error calculating regime statistics: {e}")
            return {}
    
    def _calculate_max_drawdown(self, price_series: pd.Series) -> float:
        """Calculate maximum drawdown for a price series"""
        try:
            peak = price_series.expanding().max()
            drawdown = (price_series - peak) / peak
            return drawdown.min()
        except:
            return 0.0
    
    def predict_regime_probability(self, current_data: pd.DataFrame, 
                                 historical_regimes: pd.DataFrame,
                                 regime_column: str,
                                 use_full_features: bool = True) -> Dict:
        """
        Predict probability of different regimes based on current market conditions
        
        Args:
            current_data: Recent market data
            historical_regimes: Historical data with regime labels
            regime_column: Name of the regime column
            use_full_features: If True, use all features used in regime detection
            
        Returns:
            Dictionary with regime probabilities
        """
        try:
            if use_full_features:
                # Use the same features as in regime detection
                current_features = self._extract_features(current_data)
                historical_features = self._extract_features(historical_regimes)
                
                # Calculate regime probabilities using feature similarity
                probabilities = self._calculate_feature_based_probabilities(
                    current_features, historical_features, historical_regimes[regime_column]
                )
            else:
                # Simplified approach using only returns and volatility
                recent_returns = current_data['Close'].pct_change().tail(10).mean()
                recent_volatility = current_data['Close'].pct_change().tail(30).std()
                
                # Calculate historical regime characteristics
                regime_chars = {}
                for regime in historical_regimes[regime_column].unique():
                    if pd.isna(regime):
                        continue
                    
                    regime_data = historical_regimes[historical_regimes[regime_column] == regime]
                    returns = regime_data['Close'].pct_change()
                    
                    regime_chars[regime] = {
                        'avg_returns': returns.mean(),
                        'avg_volatility': returns.std()
                    }
                
                # Calculate probabilities based on similarity to current conditions
                probabilities = {}
                total_similarity = 0
                
                for regime, chars in regime_chars.items():
                    # Euclidean distance-based similarity
                    return_diff = abs(recent_returns - chars['avg_returns'])
                    vol_diff = abs(recent_volatility - chars['avg_volatility'])
                    
                    # Inverse distance as similarity (add small constant to avoid division by zero)
                    distance = np.sqrt(return_diff**2 + vol_diff**2)
                    similarity = 1 / (1 + distance)
                    probabilities[regime] = similarity
                    total_similarity += similarity
                
                # Normalize to probabilities
                if total_similarity > 0:
                    for regime in probabilities:
                        probabilities[regime] = probabilities[regime] / total_similarity
            
            return probabilities
            
        except Exception as e:
            self.logger.error(f"Error predicting regime probability: {e}")
            return {}
    
    # Helper methods for walk-forward clustering and improved calculations
    
    def _kmeans_vol_regimes_walkforward(self, vol_series: pd.Series, n_regimes: int = 3,
                                      train_window: int = 750, test_window: int = 60) -> pd.Series:
        """Walk-forward K-means clustering for volatility regimes"""
        try:
            pipe = Pipeline([
                ("scaler", StandardScaler()),
                ("kmeans", KMeans(n_clusters=n_regimes, random_state=42, n_init="auto"))
            ])
            
            labels = pd.Series(index=vol_series.index, dtype=object)
            idx = vol_series.dropna().index
            
            for start in range(0, len(idx) - (train_window + test_window) + 1, test_window):
                train_idx = idx[start:start+train_window]
                test_idx = idx[start+train_window:start+train_window+test_window]
                
                # Fit on training data
                train_data = vol_series.loc[train_idx].values.reshape(-1, 1)
                pipe.fit(train_data)
                
                # Predict on test data
                test_data = vol_series.loc[test_idx].values.reshape(-1, 1)
                test_labels = pipe.predict(test_data)
                
                # Map cluster labels to regime names based on cluster centers
                cluster_centers = pipe.named_steps['kmeans'].cluster_centers_.flatten()
                regime_order = np.argsort(cluster_centers)
                
                if n_regimes == 3:
                    regime_names = ['Low_Vol', 'Medium_Vol', 'High_Vol']
                elif n_regimes == 2:
                    regime_names = ['Low_Vol', 'High_Vol']
                else:
                    regime_names = [f'Regime_{i}' for i in range(n_regimes)]
                
                regime_mapping = {}
                for i, regime_idx in enumerate(regime_order):
                    regime_mapping[regime_idx] = regime_names[i]
                
                # Apply mapping
                mapped_labels = [regime_mapping[label] for label in test_labels]
                labels.loc[test_idx] = mapped_labels
            
            return labels
            
        except Exception as e:
            self.logger.error(f"Error in walk-forward K-means: {e}")
            return pd.Series(index=vol_series.index)
    
    def _rolling_quantile_regimes(self, vol_series: pd.Series, n_regimes: int = 3,
                                window: int = 252) -> pd.Series:
        """Rolling quantile-based regime detection (safer alternative)"""
        try:
            if n_regimes == 2:
                q_high = vol_series.rolling(window).quantile(0.5)
                regime = pd.Series("Low_Vol", index=vol_series.index)
                regime[vol_series >= q_high] = "High_Vol"
            elif n_regimes == 3:
                q_low = vol_series.rolling(window).quantile(0.33)
                q_high = vol_series.rolling(window).quantile(0.67)
                regime = pd.Series("Medium_Vol", index=vol_series.index)
                regime[vol_series <= q_low] = "Low_Vol"
                regime[vol_series >= q_high] = "High_Vol"
            else:
                # Default to 3 regimes
                return self._rolling_quantile_regimes(vol_series, 3, window)
            
            return regime
            
        except Exception as e:
            self.logger.error(f"Error in rolling quantile regimes: {e}")
            return pd.Series("Medium_Vol", index=vol_series.index)
    
    def _rolling_r2_logprice(self, price_series: pd.Series, window: int = 80) -> pd.Series:
        """Calculate rolling R-squared for log-price trend strength"""
        try:
            log_price = np.log(price_series)
            r2_series = pd.Series(index=price_series.index, dtype=float)
            
            for i in range(window, len(price_series)):
                y = log_price.iloc[i-window:i].values
                x = np.arange(window)
                
                # Calculate R-squared manually
                x_mean = np.mean(x)
                y_mean = np.mean(y)
                
                ss_tot = np.sum((y - y_mean) ** 2)
                ss_reg = np.sum((np.polyval(np.polyfit(x, y, 1), x) - y_mean) ** 2)
                
                r2_series.iloc[i] = ss_reg / ss_tot if ss_tot > 0 else 0
            
            return r2_series.fillna(0)
            
        except Exception as e:
            self.logger.error(f"Error calculating R-squared: {e}")
            return pd.Series(0, index=price_series.index)
    
    def _gmm_market_regimes_walkforward(self, features_df: pd.DataFrame, n_components: int = 4,
                                      train_window: int = 750, test_window: int = 60) -> Tuple[pd.Series, np.ndarray]:
        """Walk-forward GMM clustering for market regimes"""
        try:
            labels = pd.Series(index=features_df.index, dtype=object)
            all_probs = []
            idx = features_df.index
            
            for start in range(0, len(idx) - (train_window + test_window) + 1, test_window):
                train_idx = idx[start:start+train_window]
                test_idx = idx[start+train_window:start+train_window+test_window]
                
                # Prepare data
                scaler = StandardScaler()
                train_features = scaler.fit_transform(features_df.loc[train_idx])
                test_features = scaler.transform(features_df.loc[test_idx])
                
                # Fit GMM
                gmm = GaussianMixture(n_components=n_components, random_state=42)
                gmm.fit(train_features)
                
                # Predict
                test_labels = gmm.predict(test_features)
                test_probs = gmm.predict_proba(test_features)
                
                # Assign regime names based on cluster characteristics
                regime_names = self._assign_regime_names(
                    features_df.loc[train_idx], gmm.predict(train_features)
                )
                
                mapped_labels = [regime_names.get(label, f'Regime_{label}') for label in test_labels]
                labels.loc[test_idx] = mapped_labels
                all_probs.extend(test_probs)
            
            probs_array = np.array(all_probs)
            return labels, probs_array
            
        except Exception as e:
            self.logger.error(f"Error in walk-forward GMM: {e}")
            return pd.Series(index=features_df.index), np.array([])
    
    def _kmeans_market_regimes_walkforward(self, features_df: pd.DataFrame, n_clusters: int = 4,
                                         train_window: int = 750, test_window: int = 60) -> pd.Series:
        """Walk-forward K-means clustering for market regimes"""
        try:
            pipe = Pipeline([
                ("scaler", StandardScaler()),
                ("kmeans", KMeans(n_clusters=n_clusters, random_state=42, n_init="auto"))
            ])
            
            labels = pd.Series(index=features_df.index, dtype=object)
            idx = features_df.index
            
            for start in range(0, len(idx) - (train_window + test_window) + 1, test_window):
                train_idx = idx[start:start+train_window]
                test_idx = idx[start+train_window:start+train_window+test_window]
                
                # Fit on training data
                pipe.fit(features_df.loc[train_idx])
                
                # Predict on test data
                test_labels = pipe.predict(features_df.loc[test_idx])
                
                # Assign regime names
                regime_names = self._assign_regime_names(
                    features_df.loc[train_idx], pipe.predict(features_df.loc[train_idx])
                )
                
                mapped_labels = [regime_names.get(label, f'Regime_{label}') for label in test_labels]
                labels.loc[test_idx] = mapped_labels
            
            return labels
            
        except Exception as e:
            self.logger.error(f"Error in walk-forward K-means market regimes: {e}")
            return pd.Series(index=features_df.index)
    
    def _assign_regime_names(self, features_df: pd.DataFrame, cluster_labels: np.ndarray) -> Dict:
        """Assign meaningful names to clusters based on their characteristics"""
        try:
            regime_names = {}
            n_clusters = len(np.unique(cluster_labels))
            
            for i in range(n_clusters):
                mask = cluster_labels == i
                cluster_data = features_df[mask]
                
                avg_returns = cluster_data['returns'].mean()
                avg_volatility = cluster_data['volatility'].mean()
                avg_momentum = cluster_data['momentum'].mean()
                
                # Adaptive thresholds based on data distribution
                vol_threshold = features_df['volatility'].quantile(0.75)
                return_pos_threshold = features_df['returns'].quantile(0.6)
                return_neg_threshold = features_df['returns'].quantile(0.4)
                
                if avg_returns > return_pos_threshold and avg_volatility < vol_threshold:
                    regime_names[i] = 'Bull_Market'
                elif avg_returns < return_neg_threshold and avg_volatility < vol_threshold:
                    regime_names[i] = 'Bear_Market'
                elif avg_volatility > vol_threshold:
                    regime_names[i] = 'High_Volatility'
                else:
                    regime_names[i] = 'Sideways_Market'
            
            return regime_names
            
        except Exception as e:
            self.logger.error(f"Error assigning regime names: {e}")
            return {i: f'Regime_{i}' for i in range(len(np.unique(cluster_labels)))}
    
    def _simple_regime_classification(self, features_df: pd.DataFrame) -> pd.Series:
        """Simple rule-based regime classification for small datasets"""
        try:
            # Calculate adaptive thresholds
            vol_high = features_df['volatility'].quantile(0.75)
            return_pos = features_df['returns'].quantile(0.6)
            return_neg = features_df['returns'].quantile(0.4)
            
            # Apply rules
            regime = pd.Series("Sideways_Market", index=features_df.index)
            
            high_vol_mask = features_df['volatility'] > vol_high
            bull_mask = (features_df['returns'] > return_pos) & ~high_vol_mask
            bear_mask = (features_df['returns'] < return_neg) & ~high_vol_mask
            
            regime[high_vol_mask] = "High_Volatility"
            regime[bull_mask] = "Bull_Market"
            regime[bear_mask] = "Bear_Market"
            
            return regime
            
        except Exception as e:
            self.logger.error(f"Error in simple regime classification: {e}")
            return pd.Series("Sideways_Market", index=features_df.index)
    
    def _extract_features(self, data: pd.DataFrame, window: int = 30) -> pd.DataFrame:
        """Extract features for regime detection"""
        try:
            returns = data['Close'].pct_change()
            volatility = returns.rolling(window=window).std()
            momentum = data['Close'].pct_change(window)
            
            # Safe volume ratio calculation
            eps = 1e-6
            avg_volume = data['Volume'].rolling(window=window).mean().clip(lower=eps)
            volume_ratio = (data['Volume'] / avg_volume).clip(0, 10)
            
            rsi = self._calculate_rsi_wilder(data, period=14)
            
            features = pd.DataFrame({
                'returns': returns,
                'volatility': volatility,
                'momentum': momentum,
                'volume_ratio': volume_ratio,
                'rsi': rsi
            }).dropna()
            
            return features
            
        except Exception as e:
            self.logger.error(f"Error extracting features: {e}")
            return pd.DataFrame()
    
    def _calculate_feature_based_probabilities(self, current_features: pd.DataFrame,
                                             historical_features: pd.DataFrame,
                                             regime_labels: pd.Series) -> Dict:
        """Calculate regime probabilities using full feature vectors"""
        try:
            if len(current_features) == 0 or len(historical_features) == 0:
                return {}
            
            # Use the most recent feature vector
            current_vector = current_features.iloc[-1].values
            
            # Calculate probabilities for each regime
            regime_probs = {}
            total_prob = 0
            
            for regime in regime_labels.unique():
                if pd.isna(regime):
                    continue
                
                # Get historical vectors for this regime
                regime_mask = regime_labels == regime
                regime_features = historical_features[regime_mask]
                
                if len(regime_features) == 0:
                    continue
                
                # Calculate average distance to regime centroid
                regime_centroid = regime_features.mean().values
                distance = np.linalg.norm(current_vector - regime_centroid)
                
                # Convert distance to probability (inverse relationship)
                prob = 1 / (1 + distance)
                regime_probs[regime] = prob
                total_prob += prob
            
            # Normalize probabilities
            if total_prob > 0:
                for regime in regime_probs:
                    regime_probs[regime] /= total_prob
            
            return regime_probs
            
        except Exception as e:
            self.logger.error(f"Error calculating feature-based probabilities: {e}")
            return {}
