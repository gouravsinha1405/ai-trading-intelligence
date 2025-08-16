import pandas as pd
import numpy as np
from typing import Optional, Dict, List
import logging
from statsmodels.robust import mad

class DataCleaner:
    """Clean and preprocess market data"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def clean_ohlcv_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean OHLCV data with safer practices - don't fabricate prices
        
        Args:
            df: DataFrame with OHLCV columns
            
        Returns:
            Cleaned DataFrame
        """
        try:
            x = df.copy()
            req = ['Open', 'High', 'Low', 'Close']
            
            # Check required columns
            if not set(req).issubset(x.columns):
                self.logger.error(f"Missing required columns: {set(req) - set(x.columns)}")
                return df
            
            # Handle datetime index and duplicates
            if not isinstance(x.index, pd.DatetimeIndex) and 'Date' in x.columns:
                x['Date'] = pd.to_datetime(x['Date'], errors='coerce')
                x = x.set_index('Date')
            
            x = x.sort_index()
            
            # Drop duplicate timestamps (keep last)
            x = x[~x.index.duplicated(keep='last')].copy()
            
            # Basic hygiene - remove NaN and non-positive prices
            x = x.dropna(subset=req)
            for col in req:
                x = x[x[col] > 0]
            
            # OHLC consistency with epsilon fix-or-drop (don't fabricate prices)
            eps = 1e-6  # Tolerance for minor rounding errors
            hi_needed = x[['Open', 'Close']].max(axis=1)
            lo_needed = x[['Open', 'Close']].min(axis=1)
            
            hi_gap = hi_needed - x['High']  # How much High is below needed
            lo_gap = x['Low'] - lo_needed   # How much Low is above needed
            
            # Only fix minor gaps (≈ one tick), drop severely broken bars
            tick_tolerance = 0.05  # Adjust based on instrument tick size
            minor_hi = hi_gap.between(0, tick_tolerance)
            minor_lo = lo_gap.between(0, tick_tolerance)
            
            # Fix minor inconsistencies
            x.loc[minor_hi, 'High'] = hi_needed[minor_hi]
            x.loc[minor_lo, 'Low'] = lo_needed[minor_lo]
            
            # Drop severely broken bars
            x = x[~((hi_gap > tick_tolerance) | (lo_gap > tick_tolerance))]
            
            # Volume handling - clip negatives but keep NaN (don't fabricate 0s)
            if 'Volume' in x.columns:
                x['Volume'] = x['Volume'].clip(lower=0)
                # Don't fillna(0) - missing volume should stay missing
            
            # Apply timezone if not already set
            if isinstance(x.index, pd.DatetimeIndex) and x.index.tz is None:
                try:
                    # Ensure no duplicate index before timezone operations
                    if x.index.duplicated().any():
                        self.logger.warning("Found remaining duplicates, cleaning again")
                        x = x[~x.index.duplicated(keep='last')].copy()
                    
                    x = x.tz_localize('Asia/Kolkata', nonexistent='shift_forward', ambiguous='NaT')
                except Exception as tz_e:
                    self.logger.warning(f"Timezone localization failed: {tz_e}")
                    # Try to continue without timezone
                    pass
            
            self.logger.info(f"Data cleaned safely: {len(df)} -> {len(x)} records")
            return x
            
        except Exception as e:
            self.logger.error(f"Error in safe OHLCV cleaning: {e}")
            return df
    
    def detect_outliers_robust(self, df: pd.DataFrame, window: int = 100, threshold: float = 8.0) -> pd.DataFrame:
        """
        Detect outliers using rolling robust statistics instead of global 3-sigma
        
        Args:
            df: DataFrame with price data
            window: Rolling window for robust statistics
            threshold: MAD-based threshold (8 is conservative)
            
        Returns:
            DataFrame with outliers removed
        """
        try:
            cleaned_df = df.copy()
            price_cols = ['Open', 'High', 'Low', 'Close']
            available_cols = [col for col in price_cols if col in cleaned_df.columns]
            
            for col in available_cols:
                # Rolling median and MAD (Median Absolute Deviation)
                roll_med = cleaned_df[col].rolling(window, min_periods=window//2).median()
                roll_mad = cleaned_df[col].rolling(window, min_periods=window//2).apply(
                    lambda x: mad(x.dropna(), c=1) if len(x.dropna()) > 5 else np.nan, raw=False
                )
                
                # Robust z-score
                z_score = (cleaned_df[col] - roll_med) / roll_mad.replace(0, np.nan)
                
                # Flag extreme outliers
                outlier_mask = z_score.abs() >= threshold
                outliers_found = outlier_mask.sum()
                
                if outliers_found > 0:
                    self.logger.warning(f"Found {outliers_found} outliers in {col} (review manually before dropping)")
                    # Log some examples for manual review
                    outlier_samples = cleaned_df[outlier_mask][col].head(3)
                    self.logger.info(f"Example outliers in {col}: {outlier_samples.tolist()}")
                
                # Remove extreme outliers (conservative threshold)
                cleaned_df = cleaned_df[~outlier_mask]
            
            self.logger.info(f"Robust outlier detection: {len(df)} -> {len(cleaned_df)} records")
            return cleaned_df
            
        except Exception as e:
            self.logger.error(f"Error in robust outlier detection: {e}")
            return df
    def handle_missing_data(self, df: pd.DataFrame, method: str = 'keep_nan') -> pd.DataFrame:
        """
        Handle missing data in time series (avoid interpolating OHLC)
        
        Args:
            df: DataFrame with time series data
            method: Method for handling missing data 
                   ('keep_nan', 'forward_fill', 'drop') - never interpolate OHLC
            
        Returns:
            DataFrame with missing data handled appropriately
        """
        try:
            price_cols = ['Open', 'High', 'Low', 'Close']
            
            if method == 'keep_nan':
                # Keep NaN for OHLC (don't fabricate prices), only fill non-price columns
                non_price_cols = [col for col in df.columns if col not in price_cols]
                result_df = df.copy()
                if non_price_cols:
                    result_df[non_price_cols] = result_df[non_price_cols].ffill()
                return result_df
                
            elif method == 'forward_fill':
                self.logger.warning("Forward filling OHLC prices - use with caution")
                return df.ffill()
                
            elif method == 'drop':
                return df.dropna()
                
            else:
                self.logger.warning(f"Unknown method {method}, keeping NaN values")
                return df
                
        except Exception as e:
            self.logger.error(f"Error handling missing data: {e}")
            return df
    
    def detect_data_gaps(self, df: pd.DataFrame, sessions: Optional[pd.DatetimeIndex] = None) -> List[Dict]:
        """
        Detect gaps in time series data using trading calendar (excludes weekends/holidays)
        
        Args:
            df: DataFrame with datetime index
            sessions: Optional trading sessions index (excludes weekends/holidays)
                     If None, uses business days as fallback
            
        Returns:
            List of dictionaries describing gaps
        """
        gaps = []
        
        try:
            if not isinstance(df.index, pd.DatetimeIndex):
                self.logger.warning("DataFrame does not have datetime index")
                return gaps
            
            # If no trading sessions provided, use business days (removes weekends)
            if sessions is None:
                # Fallback: business days only - better to pass NSE sessions including holidays
                sessions = pd.bdate_range(
                    df.index.min().normalize(), 
                    df.index.max().normalize(), 
                    tz=df.index.tz
                )
                self.logger.info("Using business days as trading sessions (consider providing NSE calendar)")
            
            # Find missing trading sessions
            actual_sessions = df.index.normalize().unique()
            missing_sessions = sessions.difference(actual_sessions)
            
            if missing_sessions.empty:
                return gaps
            
            # Group consecutive missing sessions
            missing_df = pd.DataFrame({'date': missing_sessions}).sort_values('date')
            
            # Create groups for consecutive dates
            missing_df['group'] = (missing_df['date'].diff() > pd.Timedelta('1D')).cumsum()
            
            for group_id, group in missing_df.groupby('group'):
                gap = {
                    'start_date': group['date'].iloc[0],
                    'end_date': group['date'].iloc[-1],
                    'duration': len(group),
                    'gap_type': 'missing_session'
                }
                gaps.append(gap)
            
            self.logger.info(f"Found {len(gaps)} trading session gaps (excluding weekends/holidays)")
            return gaps
            
        except Exception as e:
            self.logger.error(f"Error detecting data gaps: {e}")
            return gaps
    
    def validate_data_quality(self, df: pd.DataFrame) -> Dict:
        """
        Validate data quality and return quality metrics
        
        Args:
            df: DataFrame to validate
            
        Returns:
            Dictionary with quality metrics
        """
        quality_metrics = {
            'total_records': len(df),
            'missing_data_pct': 0,
            'duplicate_records': 0,
            'outlier_records': 0,
            'data_quality_score': 0
        }
        
        try:
            # Missing data percentage
            total_cells = df.shape[0] * df.shape[1]
            missing_cells = df.isnull().sum().sum()
            quality_metrics['missing_data_pct'] = (missing_cells / total_cells) * 100
            
            # Duplicate records
            quality_metrics['duplicate_records'] = df.duplicated().sum()
            
            # Outliers (for numerical columns)
            numerical_cols = df.select_dtypes(include=[np.number]).columns
            outlier_count = 0
            
            for col in numerical_cols:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
                outlier_count += len(outliers)
            
            quality_metrics['outlier_records'] = outlier_count
            
            # Calculate overall quality score (0-100)
            missing_score = max(0, 100 - quality_metrics['missing_data_pct'])
            duplicate_score = max(0, 100 - (quality_metrics['duplicate_records'] / len(df)) * 100)
            outlier_score = max(0, 100 - (quality_metrics['outlier_records'] / len(df)) * 100)
            
            quality_metrics['data_quality_score'] = (missing_score + duplicate_score + outlier_score) / 3
            
            self.logger.info(f"Data quality score: {quality_metrics['data_quality_score']:.2f}")
            return quality_metrics
            
        except Exception as e:
            self.logger.error(f"Error validating data quality: {e}")
            return quality_metrics
    
    def standardize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Standardize column names and data types
        
        Args:
            df: DataFrame to standardize
            
        Returns:
            Standardized DataFrame
        """
        try:
            standardized_df = df.copy()
            
            # Column name mapping
            column_mapping = {
                # Common variations
                'open': 'Open',
                'high': 'High',
                'low': 'Low',
                'close': 'Close',
                'volume': 'Volume',
                'adj close': 'Adj_Close',
                'adjusted close': 'Adj_Close',
                'date': 'Date',
                'datetime': 'Date',
                'timestamp': 'Date'
            }
            
            # Apply column mapping
            for old_name, new_name in column_mapping.items():
                if old_name in standardized_df.columns:
                    standardized_df.rename(columns={old_name: new_name}, inplace=True)
            
            # Ensure proper data types
            price_columns = ['Open', 'High', 'Low', 'Close', 'Adj_Close']
            for col in price_columns:
                if col in standardized_df.columns:
                    standardized_df[col] = pd.to_numeric(standardized_df[col], errors='coerce')
            
            if 'Volume' in standardized_df.columns:
                standardized_df['Volume'] = pd.to_numeric(standardized_df['Volume'], errors='coerce')
            
            # Handle date column
            if 'Date' in standardized_df.columns:
                standardized_df['Date'] = pd.to_datetime(standardized_df['Date'], errors='coerce')
                if not isinstance(standardized_df.index, pd.DatetimeIndex):
                    standardized_df.set_index('Date', inplace=True)
            
            self.logger.info("Columns standardized successfully")
            return standardized_df
            
        except Exception as e:
            self.logger.error(f"Error standardizing columns: {e}")
            return df
    
    def resample_data(self, df: pd.DataFrame, target_freq: str = '5T') -> pd.DataFrame:
        """
        Resample time series data with proper session boundaries and timezone handling
        
        Args:
            df: DataFrame with datetime index
            target_freq: Target frequency ('5T', '15T', 'H', 'D', etc.)
            
        Returns:
            Resampled DataFrame with session boundary awareness
        """
        try:
            if not isinstance(df.index, pd.DatetimeIndex):
                self.logger.error("DataFrame must have datetime index for resampling")
                return df
            
            # Ensure timezone awareness (Indian market)
            if df.index.tz is None:
                df = df.tz_localize('Asia/Kolkata', nonexistent='shift_forward', ambiguous='NaT')
            else:
                df = df.tz_convert('Asia/Kolkata')
            
            # Filter to market hours only (9:15 AM - 3:30 PM IST)
            df = df.between_time('09:15', '15:30')
            
            # OHLCV resampling rules
            agg_rules = {
                'Open': 'first',
                'High': 'max', 
                'Low': 'min',
                'Close': 'last'
            }
            
            # Add Volume if present
            if 'Volume' in df.columns:
                agg_rules['Volume'] = 'sum'
            
            # Apply only to columns that exist
            available_rules = {col: rule for col, rule in agg_rules.items() if col in df.columns}
            
            if not available_rules:
                self.logger.warning("No OHLCV columns found for resampling")
                return df.resample(target_freq, label='right', closed='right').last()
            
            # Resample with proper labeling
            resampled_df = df.resample(
                target_freq, 
                label='right', 
                closed='right'
            ).agg(available_rules)
            
            # Handle other columns (non-OHLCV) - take last value
            other_cols = [col for col in df.columns if col not in available_rules.keys()]
            if other_cols:
                other_data = df[other_cols].resample(
                    target_freq, 
                    label='right', 
                    closed='right'
                ).last()
                resampled_df = pd.concat([resampled_df, other_data], axis=1)
            
            # Drop incomplete last bar if OHLC has NaN (market still open)
            if len(resampled_df) > 0:
                ohlc_cols = ['Open', 'High', 'Low', 'Close']
                available_ohlc = [col for col in ohlc_cols if col in resampled_df.columns]
                
                if available_ohlc:
                    last_bar_complete = resampled_df[available_ohlc].iloc[-1].notna().all()
                    if not last_bar_complete:
                        resampled_df = resampled_df.iloc[:-1]
                        self.logger.info("Dropped incomplete last bar")
            
            self.logger.info(f"Data resampled to {target_freq}: {len(df)} -> {len(resampled_df)} bars")
            return resampled_df
            
        except Exception as e:
            self.logger.error(f"Error resampling data: {e}")
            return df
    
    def get_nse_trading_calendar(self, start_date: pd.Timestamp, end_date: pd.Timestamp) -> pd.DatetimeIndex:
        """
        Generate NSE trading calendar (business days minus known holidays)
        
        Args:
            start_date: Start date
            end_date: End date
            
        Returns:
            DatetimeIndex of trading sessions
        """
        try:
            # Start with business days
            business_days = pd.bdate_range(start_date, end_date, tz='Asia/Kolkata')
            
            # Known NSE holidays (add more as needed)
            nse_holidays_2024_2025 = [
                '2024-01-26',  # Republic Day
                '2024-03-08',  # Holi
                '2024-03-29',  # Good Friday
                '2024-04-17',  # Ram Navami
                '2024-05-01',  # Maharashtra Day
                '2024-08-15',  # Independence Day
                '2024-10-02',  # Gandhi Jayanti
                '2024-10-31',  # Diwali Laxmi Puja
                '2024-11-01',  # Diwali (Balipratipada)
                '2024-11-15',  # Guru Nanak Jayanti
                '2025-01-26',  # Republic Day
                '2025-03-14',  # Holi
                '2025-04-18',  # Good Friday
                '2025-05-01',  # Maharashtra Day
                '2025-08-15',  # Independence Day
                '2025-10-02',  # Gandhi Jayanti
            ]
            
            # Convert to datetime and localize
            holiday_dates = pd.to_datetime(nse_holidays_2024_2025).tz_localize('Asia/Kolkata')
            
            # Remove holidays from business days
            trading_sessions = business_days.difference(holiday_dates)
            
            self.logger.info(f"Generated NSE trading calendar: {len(trading_sessions)} sessions")
            return trading_sessions
            
        except Exception as e:
            self.logger.error(f"Error generating trading calendar: {e}")
            return pd.bdate_range(start_date, end_date, tz='Asia/Kolkata')
    
    def adjust_for_corporate_actions(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """
        Adjust OHLCV data for corporate actions (splits, bonuses, dividends)
        
        Args:
            df: DataFrame with raw OHLCV data
            symbol: Stock symbol
            
        Returns:
            DataFrame with adjusted prices
        """
        try:
            # This is a placeholder - in production, fetch corporate actions from data provider
            # For now, just add the adjusted close column structure
            
            adjusted_df = df.copy()
            
            # Placeholder: assume no corporate actions (adjustment factor = 1)
            # In production, calculate adjustment factors from corporate action events
            adjustment_factor = pd.Series(1.0, index=df.index)
            
            # Create adjusted columns
            if 'Close' in adjusted_df.columns:
                adjusted_df['Adj_Close'] = adjusted_df['Close'] * adjustment_factor
                
            # Optionally adjust OHLC (uncomment if needed)
            # adjusted_df['Adj_Open'] = adjusted_df['Open'] * adjustment_factor
            # adjusted_df['Adj_High'] = adjusted_df['High'] * adjustment_factor  
            # adjusted_df['Adj_Low'] = adjusted_df['Low'] * adjustment_factor
            
            # Volume should be inversely adjusted for splits
            if 'Volume' in adjusted_df.columns:
                # Volume adjustment would be inverse of price adjustment
                adjusted_df['Adj_Volume'] = adjusted_df['Volume'] / adjustment_factor
            
            self.logger.info(f"Corporate actions adjustment applied for {symbol}")
            self.logger.warning("Using placeholder corporate actions - implement real data source")
            
            return adjusted_df
            
        except Exception as e:
            self.logger.error(f"Error adjusting for corporate actions: {e}")
            return df
    
    def validate_ohlc_integrity(self, df: pd.DataFrame) -> Dict:
        """
        Validate OHLC data integrity with detailed checks
        
        Args:
            df: DataFrame with OHLC data
            
        Returns:
            Dictionary with validation results
        """
        validation_results = {
            'total_records': len(df),
            'ohlc_consistency_violations': 0,
            'negative_prices': 0,
            'zero_volume_days': 0,
            'suspicious_gaps': 0,
            'integrity_score': 0
        }
        
        try:
            if len(df) == 0:
                return validation_results
            
            # OHLC consistency checks
            ohlc_cols = ['Open', 'High', 'Low', 'Close']
            available_ohlc = [col for col in ohlc_cols if col in df.columns]
            
            if len(available_ohlc) >= 4:
                try:
                    # High should be >= max(Open, Close)
                    high_max = df[['Open', 'Close']].max(axis=1)
                    high_mask = (df['High'] < high_max).fillna(False)
                    high_violations = sum(high_mask)  # Simple sum() function
                    
                    # Low should be <= min(Open, Close)  
                    low_min = df[['Open', 'Close']].min(axis=1)
                    low_mask = (df['Low'] > low_min).fillna(False)
                    low_violations = sum(low_mask)  # Simple sum() function
                    
                    validation_results['ohlc_consistency_violations'] = high_violations + low_violations
                except Exception as ohlc_e:
                    self.logger.warning(f"OHLC consistency check failed: {ohlc_e}")
                    validation_results['ohlc_consistency_violations'] = 0
            
            # Negative price checks
            for col in available_ohlc:
                try:
                    negative_mask = (df[col] <= 0).fillna(False)
                    negative_count = sum(negative_mask)  # Simple sum() function
                    validation_results['negative_prices'] += negative_count
                except Exception as neg_e:
                    self.logger.warning(f"Negative price check failed for {col}: {neg_e}")
            
            # Volume checks
            if 'Volume' in df.columns:
                try:
                    volume_mask = (df['Volume'] == 0).fillna(False)
                    validation_results['zero_volume_days'] = sum(volume_mask)  # Simple sum() function
                except Exception as vol_e:
                    self.logger.warning(f"Volume check failed: {vol_e}")
                    validation_results['zero_volume_days'] = 0
            
            # Price gap analysis (returns > 20% in a day)
            if 'Close' in df.columns and len(df) > 1:
                try:
                    daily_returns = df['Close'].pct_change().abs()
                    gap_mask = (daily_returns > 0.20).fillna(False)
                    validation_results['suspicious_gaps'] = sum(gap_mask)  # Simple sum() function
                except Exception as gap_e:
                    self.logger.warning(f"Price gap analysis failed: {gap_e}")
                    validation_results['suspicious_gaps'] = 0
            
            # Calculate integrity score (0-100)
            total_issues = (
                validation_results['ohlc_consistency_violations'] +
                validation_results['negative_prices'] + 
                validation_results['suspicious_gaps']
            )
            
            if len(df) > 0:
                integrity_score = max(0, 100 - (total_issues / len(df)) * 100)
                validation_results['integrity_score'] = round(integrity_score, 2)
            
            self.logger.info(f"OHLC integrity score: {validation_results['integrity_score']}")
            return validation_results
            
        except Exception as e:
            self.logger.error(f"Error validating OHLC integrity: {e}")
            return validation_results
