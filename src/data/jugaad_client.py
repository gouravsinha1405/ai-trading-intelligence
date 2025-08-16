import logging
import random
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import pandas as pd
import pytz

# Graceful fallback for jugaad-data import
JUGAAD_AVAILABLE = False
try:
    from jugaad_data.nse import NSELive, index_df, stock_df
    JUGAAD_AVAILABLE = True
    print("✅ jugaad-data loaded successfully")
except ImportError as e:
    print(f"⚠️ jugaad-data not available: {e}")
    print("📊 Using yfinance as primary data source")

# Fallback to yfinance
try:
    import yfinance as yf
    YF_AVAILABLE = True
    print("✅ yfinance loaded successfully")
except ImportError:
    YF_AVAILABLE = False
    print("❌ yfinance not available")

# Timezone for Indian markets
IST = pytz.timezone('Asia/Kolkata')


class JugaadDataClient:
    """Production-ready client for fetching live market data using jugaad-data"""

    def __init__(self, rate_limit: float = 1.0):
        """
        Initialize JugaadDataClient

        Args:
            rate_limit: Rate limit between API calls in seconds
        """
        self.rate_limit = rate_limit
        self.logger = logging.getLogger(__name__)

        # Initialize DataCleaner for production-grade cleaning
        try:
            from src.data.data_cleaner import DataCleaner
            self.data_cleaner = DataCleaner()
        except ImportError:
            self.logger.warning("DataCleaner not available, using basic cleaning")
            self.data_cleaner = None

        # NSE holidays (basic set - should be updated annually)
        self.nse_holidays = {
            # 2024 holidays (example)
            datetime(2024, 1, 26).date(),  # Republic Day
            datetime(2024, 3, 8).date(),   # Holi
            datetime(2024, 3, 29).date(),  # Good Friday
            datetime(2024, 8, 15).date(),  # Independence Day
            datetime(2024, 10, 2).date(),  # Gandhi Jayanti
            datetime(2024, 11, 1).date(),  # Diwali
            datetime(2024, 11, 15).date(),  # Guru Nanak Jayanti
        }

        # Initialize NSE Live client if jugaad-data is available
        if JUGAAD_AVAILABLE:
            try:
                self.nse_live = NSELive()
                self.logger.info("NSE Live client initialized successfully")
            except Exception as e:
                self.nse_live = None
                self.logger.warning(f"NSE Live client initialization failed: {e}")
        else:
            self.nse_live = None
            self.logger.info(
                "NSE Live client not available (jugaad-data not installed)")

    def _to_float(self, x) -> float:
        """
        Safely convert NSE numeric strings to float

        Args:
            x: Input value (may be string with commas, ₹ symbol, etc.)

        Returns:
            Float value or NaN if conversion fails
        """
        if x is None:
            return float('nan')
        if isinstance(x, (int, float)):
            return float(x)

        # Clean string representation
        s = str(x).replace(',', '').replace('₹', '').replace('Rs.', '').strip()
        try:
            return float(s)
        except (ValueError, TypeError):
            return float('nan')

    def get_stock_data(self, symbol: str, from_date, to_date,
                       series: str = "EQ") -> Optional[pd.DataFrame]:
        """
        Get historical stock data with fallback support

        Args:
            symbol: Stock symbol (e.g., 'RELIANCE', 'TCS')
            from_date: Start date (datetime or string YYYY-MM-DD)
            to_date: End date (datetime or string YYYY-MM-DD)
            series: Series type (EQ for equity)

        Returns:
            DataFrame with OHLCV data
        """
        # Try jugaad-data first if available
        if JUGAAD_AVAILABLE:
            try:
                # Convert string dates to datetime if needed
                if isinstance(from_date, str):
                    from_date = datetime.strptime(from_date, '%Y-%m-%d')
                if isinstance(to_date, str):
                    to_date = datetime.strptime(to_date, '%Y-%m-%d')

                df = stock_df(
                    symbol=symbol,
                    from_date=from_date,
                    to_date=to_date,
                    series=series)

                if df is not None and not df.empty:
                    df = self._clean_data(df)
                    self.logger.info(
                        f"Retrieved {len(df)} records for {symbol} via jugaad-data")
                    return df

            except Exception as e:
                self.logger.warning(
                    f"jugaad-data failed for {symbol}: {e}, trying yfinance")

        # Fallback to yfinance
        if YF_AVAILABLE:
            try:
                import yfinance as yf

                # Convert NSE symbol to Yahoo Finance format
                yf_symbol = symbol + ".NS"  # Add .NS for NSE stocks

                ticker = yf.Ticker(yf_symbol)
                df = ticker.history(start=from_date, end=to_date)

                if not df.empty:
                    # Rename columns to match jugaad-data format
                    df.rename(columns={
                        'Open': 'OPEN',
                        'High': 'HIGH',
                        'Low': 'LOW',
                        'Close': 'CLOSE',
                        'Volume': 'VOLUME'
                    }, inplace=True)

                    df = self._clean_data(df)
                    self.logger.info(
                        f"Retrieved {len(df)} records for {symbol} via yfinance")
                    return df

            except Exception as e:
                self.logger.error(f"yfinance also failed for {symbol}: {e}")

        self.logger.error(f"All data sources failed for {symbol}")
        return None

    def get_index_data(self, index: str, from_date: datetime,
                       to_date: datetime) -> Optional[pd.DataFrame]:
        """
        Get historical index data

        Args:
            index: Index name (e.g., 'NIFTY 50', 'NIFTY BANK')
            from_date: Start date
            to_date: End date

        Returns:
            DataFrame with index OHLC data
        """
        try:
            df = index_df(symbol=index, from_date=from_date, to_date=to_date)

            if df is not None and not df.empty:
                df = self._clean_data_robust(df)
                self.logger.info(f"Retrieved {len(df)} records for {index}")
                return df
            else:
                self.logger.warning(
                    f"No data found for {index} from {from_date} to {to_date}")
                return None

        except Exception as e:
            self.logger.error(f"Error fetching index data for {index}: {e}")
            return None

    def get_live_price(self, symbol: str) -> Optional[Dict]:
        """
        Get live price for a symbol with robust parsing

        Args:
            symbol: Stock symbol

        Returns:
            Dictionary with live price data (properly parsed numbers)
        """
        if not self.nse_live:
            self.logger.error("NSE Live client not initialized")
            return None

        try:
            data = self.nse_live.stock_quote(symbol)

            if not data:
                self.logger.warning(f"No live data for {symbol}")
                return None

            # Handle nested price info structure (common in NSE responses)
            price_info = data.get('priceInfo', {})

            # Extract data with preference for nested structure
            last_price = price_info.get('lastPrice', data.get('lastPrice'))
            change = price_info.get('change', data.get('change'))
            p_change = price_info.get('pChange', data.get('pChange'))

            # Handle intraday high/low nested structure
            intraday_hl = price_info.get('intraDayHighLow', {})
            day_high = intraday_hl.get('max', data.get('dayHigh'))
            day_low = intraday_hl.get('min', data.get('dayLow'))

            open_price = price_info.get('open', data.get('open'))
            volume = data.get('totalTradedVolume', 0)

            return {
                'symbol': symbol,
                'price': self._to_float(last_price),
                'change': self._to_float(change),
                'pChange': self._to_float(p_change),
                'open': self._to_float(open_price),
                'high': self._to_float(day_high),
                'low': self._to_float(day_low),
                'volume': int(self._to_float(volume) or 0),
                'timestamp': pd.Timestamp.now(tz=IST)
            }

        except Exception as e:
            self.logger.error(f"Error fetching live price for {symbol}: {e}")
            return None

    def get_multiple_live_prices(self, symbols: List[str]) -> Dict[str, Dict]:
        """
        Get live prices for multiple symbols

        Args:
            symbols: List of stock symbols

        Returns:
            Dictionary with symbol as key and price data as value
        """
        results = {}

        for symbol in symbols:
            price_data = self.get_live_price(symbol)
            if price_data:
                results[symbol] = price_data

            # Add small delay to avoid rate limiting
            time.sleep(0.1)

        return results

    def get_market_status(self) -> Dict:
        """
        Get current market status with timezone awareness and holiday checking

        Returns:
            Dictionary with market status information
        """
        if not self.nse_live:
            return {"status": "unknown", "message": "NSE Live client not available"}

        try:
            # Get current time in IST
            now = pd.Timestamp.now(tz=IST)

            # Check if today is a trading holiday
            today_date = now.date()
            if today_date in self.nse_holidays:
                return {"status": "closed", "message": f"Market closed - NSE Holiday"}

            # Indian market hours: 9:15 AM to 3:30 PM IST
            market_open = now.replace(hour=9, minute=15, second=0, microsecond=0)
            market_close = now.replace(hour=15, minute=30, second=0, microsecond=0)

            # Check if it's a weekday
            if now.weekday() >= 5:  # Saturday = 5, Sunday = 6
                return {"status": "closed", "message": "Market closed - Weekend"}

            if market_open <= now <= market_close:
                return {"status": "open", "message": "Market is open"}
            elif now < market_open:
                return {"status": "pre-market", "message": "Pre-market hours"}
            else:
                return {"status": "closed", "message": "Market closed"}

        except Exception as e:
            self.logger.error(f"Error checking market status: {e}")
            return {"status": "error", "message": str(e)}

    def get_stocks_data(self, symbols: List[str], start_date: str, end_date: str,
                        use_cache: bool = True) -> Dict[str, pd.DataFrame]:
        """
        Get historical data for multiple stocks with enhanced rate limiting

        Args:
            symbols: List of stock symbols
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            use_cache: Whether to use cached data

        Returns:
            Dictionary mapping symbols to DataFrames
        """
        results = {}
        failed_symbols = []

        for i, symbol in enumerate(symbols):
            try:
                # Enhanced rate limiting with jitter
                if i > 0:
                    # Add jitter for rate limiting
                    delay = self.rate_limit + random.uniform(0, 0.5)  # nosec B311
                    time.sleep(delay)

                df = self.get_stock_data(symbol, start_date, end_date)

                if df is not None and not df.empty:
                    results[symbol] = df
                    self.logger.info(
                        f"Successfully fetched data for {symbol}: {len(df)} records")
                else:
                    failed_symbols.append(symbol)
                    self.logger.warning(f"No data retrieved for {symbol}")

            except Exception as e:
                failed_symbols.append(symbol)
                self.logger.error(f"Error fetching data for {symbol}: {e}")
                # Exponential backoff on repeated failures
                time.sleep(min(self.rate_limit * 2, 5.0))

        if failed_symbols:
            self.logger.warning(f"Failed to fetch data for: {failed_symbols}")

        return results

    def _clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and standardize the data using production-ready methods

        Args:
            df: Raw DataFrame from jugaad-data

        Returns:
            Cleaned and standardized DataFrame
        """
        try:
            if df is None or df.empty:
                return df

            cleaned_df = df.copy()

            # Ensure datetime index
            date_cols = ['DATE', 'CH_TIMESTAMP']
            date_col = None
            for col in date_cols:
                if col in cleaned_df.columns:
                    date_col = col
                    break

            if date_col:
                cleaned_df[date_col] = pd.to_datetime(cleaned_df[date_col])
                cleaned_df.set_index(date_col, inplace=True)

            # Standardize column names - handle different jugaad-data formats
            column_mapping = {
                # Standard format
                'OPEN': 'Open',
                'HIGH': 'High',
                'LOW': 'Low',
                'CLOSE': 'Close',
                'VOLUME': 'Volume',
                'TURNOVER': 'Turnover',
                # Alternative format with CH_ prefix
                'CH_OPENING_PRICE': 'Open',
                'CH_TRADE_HIGH_PRICE': 'High',
                'CH_TRADE_LOW_PRICE': 'Low',
                'CH_CLOSING_PRICE': 'Close',
                'CH_TOT_TRADED_QTY': 'Volume',
                'CH_TOT_TRADED_VAL': 'Turnover',
                # Additional mappings
                'LTP': 'Close',  # Last Traded Price as Close if Close not available
                'VALUE': 'Turnover'
            }

            cleaned_df.rename(columns=column_mapping, inplace=True)

            # Use DataCleaner for production-grade cleaning
            if hasattr(self, 'data_cleaner') and self.data_cleaner:
                # Check if it has the expected method name
                if hasattr(self.data_cleaner, 'clean_data'):
                    cleaned_df = self.data_cleaner.clean_data(cleaned_df)
                elif hasattr(self.data_cleaner, 'clean_ohlcv_data'):
                    cleaned_df = self.data_cleaner.clean_ohlcv_data(cleaned_df)
                else:
                    self.logger.warning("DataCleaner doesn't have expected methods")
            else:
                # Fallback basic cleaning
                essential_cols = ['Open', 'High', 'Low', 'Close']
                available_cols = [
                    col for col in essential_cols if col in cleaned_df.columns]

                if available_cols:
                    cleaned_df.dropna(subset=available_cols, inplace=True)

            # Sort by date
            cleaned_df.sort_index(inplace=True)

            self.logger.debug(f"Data cleaned: {len(cleaned_df)} records")
            return cleaned_df

        except Exception as e:
            self.logger.error(f"Error cleaning data: {e}")
            return df  # Return original if cleaning fails

    def adjust_for_corporate_actions(
            self,
            df: pd.DataFrame,
            symbol: str) -> pd.DataFrame:
        """
        Adjust historical data for corporate actions (splits, bonuses, dividends)

        Args:
            df: Historical price DataFrame
            symbol: Stock symbol

        Returns:
            Adjusted DataFrame
        """
        try:
            # This is a placeholder for corporate action adjustments
            # In a production system, you would fetch corporate action data
            # and apply appropriate adjustment factors

            # For now, we'll return the original data
            # Future enhancement: Implement actual corporate action handling
            self.logger.debug(f"Corporate action adjustment placeholder for {symbol}")

            return df

        except Exception as e:
            self.logger.error(f"Error adjusting for corporate actions: {e}")
            return df

    def get_top_stocks(self) -> List[str]:
        """
        Get list of top stocks for analysis

        Returns:
            List of stock symbols
        """
        return [
            'RELIANCE', 'TCS', 'HDFCBANK', 'INFY', 'HINDUNILVR',
            'ICICIBANK', 'KOTAKBANK', 'SBIN', 'BHARTIARTL', 'ITC',
            'ASIANPAINT', 'LT', 'AXISBANK', 'MARUTI', 'SUNPHARMA',
            'ULTRACEMCO', 'TITAN', 'WIPRO', 'NESTLEIND', 'POWERGRID'
        ]

    def get_major_indices(self) -> List[str]:
        """
        Get list of major indices

        Returns:
            List of index names
        """
        return [
            'NIFTY 50',
            'NIFTY BANK',
            'NIFTY IT',
            'NIFTY PHARMA',
            'NIFTY AUTO',
            'NIFTY FMCG'
        ]
