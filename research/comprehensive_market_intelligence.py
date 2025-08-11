"""
Comprehensive Market Intelligence Engine
==========================================

VISION: Build India's most comprehensive market analysis system that correlates
stock movements with EVERY possible factor - news, politics, disasters, tech trends,
social sentiment, and global events.

GOAL: Create atomic-level explanations for market movements with 10+ years of data
to build predictive models that understand WHY markets move, not just WHEN.

DATA SOURCES TO INTEGRATE:
1. Market Data (10 years): Prices, volumes, fundamentals, sector performance
2. News Sentiment: Economic Times, Business Standard, Mint, etc.
3. Political Events: Elections, budget announcements, policy changes
4. Global Events: Wars, natural disasters, pandemics, tech disruptions  
5. Google Trends: Search patterns, public sentiment indicators
6. Economic Data: GDP, inflation, interest rates, FII/DII flows
7. Sector Intelligence: Tech adoption, regulatory changes, earnings patterns

ORIGINAL INNOVATION BY: Gourav Sinha
DATE: August 11, 2025
PURPOSE: Build explainable AI for market movements with multi-factor analysis
NOVELTY: First comprehensive event-correlation engine for Indian markets
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import requests
import json
import warnings
from typing import Dict, List, Tuple, Optional
import time

# For news and sentiment analysis
try:
    from newsapi import NewsApiClient
except ImportError:
    print("Install newsapi-python: pip install newsapi-python")

# For Google Trends
try:
    from pytrends.request import TrendReq
except ImportError:
    print("Install pytrends: pip install pytrends")

# For advanced NLP
try:
    from textblob import TextBlob
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
except ImportError:
    print("Install sentiment libraries: pip install textblob vaderSentiment")

warnings.filterwarnings('ignore')

class ComprehensiveMarketIntelligence:
    """
    AI-Powered Market Intelligence with Multi-Factor Analysis
    
    LEARNING OBJECTIVES:
    1. Correlate market movements with news events
    2. Understand political impact on sectors
    3. Track global event effects on Indian markets  
    4. Build sentiment-based trading signals
    5. Create explainable AI for market predictions
    
    BUSINESS VALUE:
    - Predict market movements before they happen
    - Explain WHY stocks moved (not just that they moved)
    - Generate alpha through information asymmetry
    - Build institutional-grade research capability
    """
    
    def __init__(self):
        self.start_date = "2014-01-01"  # 10+ years of data
        self.end_date = datetime.now().strftime("%Y-%m-%d")
        
        # Initialize sentiment analyzer
        try:
            self.sentiment_analyzer = SentimentIntensityAnalyzer()
        except:
            self.sentiment_analyzer = None
            
        # Google Trends setup
        try:
            self.pytrends = TrendReq(hl='en-US', tz=360)
        except:
            self.pytrends = None
            
        print("🎯 Comprehensive Market Intelligence Engine Initialized")
        print(f"📅 Analysis Period: {self.start_date} to {self.end_date}")
    
    def get_comprehensive_market_data(self, symbols: List[str]) -> Dict:
        """
        Get 10+ years of comprehensive market data
        
        LEARNING: Historical Analysis
        - Price movements over different time periods
        - Volume patterns during events
        - Sector rotation patterns
        - Market cap changes over time
        """
        print("📊 Fetching 10+ years of market data...")
        
        market_data = {}
        
        for symbol in symbols:
            try:
                print(f"  Fetching {symbol}...")
                ticker = yf.Ticker(symbol)
                
                # Get historical data
                hist_data = ticker.history(start=self.start_date, end=self.end_date)
                
                # Get company info
                info = ticker.info
                
                # Calculate additional metrics
                hist_data['Daily_Return'] = hist_data['Close'].pct_change()
                hist_data['Volatility_20D'] = hist_data['Daily_Return'].rolling(20).std() * np.sqrt(252)
                hist_data['SMA_50'] = hist_data['Close'].rolling(50).mean()
                hist_data['SMA_200'] = hist_data['Close'].rolling(200).mean()
                
                # Volume analysis
                hist_data['Volume_SMA'] = hist_data['Volume'].rolling(20).mean()
                hist_data['Volume_Spike'] = hist_data['Volume'] / hist_data['Volume_SMA']
                
                market_data[symbol] = {
                    'historical_data': hist_data,
                    'company_info': info,
                    'sector': info.get('sector', 'Unknown'),
                    'industry': info.get('industry', 'Unknown')
                }
                
                time.sleep(0.1)  # Rate limiting
                
            except Exception as e:
                print(f"  Error fetching {symbol}: {e}")
                continue
        
        print(f"✅ Retrieved data for {len(market_data)} symbols")
        return market_data
    
    def get_major_events_timeline(self) -> List[Dict]:
        """
        Create timeline of major events affecting Indian markets
        
        LEARNING: Event Impact Analysis
        - How different types of events affect markets
        - Sector-specific impacts of various events
        - Recovery patterns after major shocks
        - Correlation between event magnitude and market reaction
        """
        print("🌍 Building major events timeline...")
        
        # Major events affecting Indian markets (2014-2025)
        major_events = [
            # Political Events
            {"date": "2014-05-16", "event": "Modi Government Election Victory", "type": "political", "impact": "positive", "sectors": ["banking", "infrastructure", "defense"]},
            {"date": "2016-11-08", "event": "Demonetization Announcement", "type": "policy", "impact": "negative", "sectors": ["banking", "consumer", "real_estate"]},
            {"date": "2017-07-01", "event": "GST Implementation", "type": "policy", "impact": "mixed", "sectors": ["fmcg", "logistics", "technology"]},
            {"date": "2019-05-23", "event": "Modi Re-election", "type": "political", "impact": "positive", "sectors": ["infrastructure", "banking", "defense"]},
            
            # Global Events
            {"date": "2020-03-24", "event": "COVID-19 Lockdown", "type": "pandemic", "impact": "negative", "sectors": ["aviation", "hospitality", "auto"]},
            {"date": "2020-03-23", "event": "Circuit Breaker Market Crash", "type": "market_crash", "impact": "negative", "sectors": ["all"]},
            {"date": "2022-02-24", "event": "Russia-Ukraine War", "type": "war", "impact": "negative", "sectors": ["oil", "defense", "commodities"]},
            
            # Economic Events
            {"date": "2018-09-21", "event": "IL&FS Crisis", "type": "financial_crisis", "impact": "negative", "sectors": ["banking", "nbfc", "real_estate"]},
            {"date": "2019-08-30", "event": "Corporate Tax Cut", "type": "policy", "impact": "positive", "sectors": ["banking", "auto", "fmcg"]},
            
            # Tech & Regulatory
            {"date": "2016-09-01", "event": "Jio Launch", "type": "tech_disruption", "impact": "mixed", "sectors": ["telecom", "technology"]},
            {"date": "2021-01-01", "event": "Digital Payment Boom", "type": "tech_adoption", "impact": "positive", "sectors": ["fintech", "technology"]},
            
            # Natural Disasters
            {"date": "2018-08-15", "event": "Kerala Floods", "type": "natural_disaster", "impact": "negative", "sectors": ["agriculture", "insurance"]},
            
            # Global Tech Events
            {"date": "2023-01-01", "event": "AI/ChatGPT Boom", "type": "tech_revolution", "impact": "positive", "sectors": ["technology", "software"]},
            {"date": "2021-01-01", "event": "Crypto Adoption", "type": "fintech", "impact": "mixed", "sectors": ["fintech", "banking"]},
        ]
        
        # Sort by date
        major_events.sort(key=lambda x: x['date'])
        
        print(f"✅ Created timeline with {len(major_events)} major events")
        return major_events
    
    def analyze_google_trends(self, keywords: List[str], timeframe: str = "2014-01-01 2025-08-11") -> Dict:
        """
        Analyze Google Trends data for market-relevant keywords
        
        LEARNING: Sentiment & Interest Analysis
        - Public interest patterns before market moves
        - Correlation between search trends and stock performance
        - Early warning signals from search behavior
        """
        if not self.pytrends:
            print("⚠️ Google Trends not available (install pytrends)")
            return {}
            
        print("🔍 Analyzing Google Trends data...")
        
        trends_data = {}
        
        try:
            for keyword in keywords:
                print(f"  Analyzing trends for: {keyword}")
                
                # Build payload
                self.pytrends.build_payload([keyword], timeframe=timeframe, geo='IN')
                
                # Get interest over time
                interest_data = self.pytrends.interest_over_time()
                
                if not interest_data.empty:
                    trends_data[keyword] = interest_data[keyword]
                
                time.sleep(1)  # Rate limiting
                
        except Exception as e:
            print(f"  Error fetching trends: {e}")
        
        print(f"✅ Retrieved trends for {len(trends_data)} keywords")
        return trends_data
    
    def calculate_event_impact(self, market_data: Dict, events: List[Dict], window_days: int = 10) -> List[Dict]:
        """
        Calculate market impact of major events
        
        LEARNING: Event-Driven Analysis
        - Immediate vs. delayed market reactions
        - Sector-specific event sensitivities
        - Recovery patterns after different event types
        - Predictive patterns for similar future events
        """
        print("📈 Calculating event impacts on market...")
        
        event_impacts = []
        
        for event in events:
            event_date = pd.to_datetime(event['date'])
            
            # Calculate impact for each symbol
            for symbol, data in market_data.items():
                hist_data = data['historical_data']
                sector = data['sector']
                
                # Handle timezone compatibility
                if hist_data.index.tz is not None:
                    if event_date.tz is None:
                        event_date = event_date.tz_localize(hist_data.index.tz)
                    else:
                        event_date = event_date.tz_convert(hist_data.index.tz)
                
                # Find the closest trading day
                try:
                    closest_date = hist_data.index[hist_data.index >= event_date][0] if len(hist_data.index[hist_data.index >= event_date]) > 0 else None
                except:
                    # If timezone issues persist, convert to naive datetime
                    event_date_naive = event_date.tz_localize(None) if event_date.tz is not None else event_date
                    hist_index_naive = hist_data.index.tz_localize(None) if hist_data.index.tz is not None else hist_data.index
                    closest_date = hist_index_naive[hist_index_naive >= event_date_naive][0] if len(hist_index_naive[hist_index_naive >= event_date_naive]) > 0 else None
                
                if closest_date is None:
                    continue
                
                # Calculate returns around the event
                pre_event_idx = hist_data.index.get_loc(closest_date)
                
                if pre_event_idx < window_days or pre_event_idx + window_days >= len(hist_data):
                    continue
                
                # Pre-event return (10 days before)
                pre_start = pre_event_idx - window_days
                pre_end = pre_event_idx - 1
                pre_return = (hist_data.iloc[pre_end]['Close'] / hist_data.iloc[pre_start]['Close'] - 1) * 100
                
                # Post-event return (10 days after)
                post_start = pre_event_idx
                post_end = pre_event_idx + window_days
                post_return = (hist_data.iloc[post_end]['Close'] / hist_data.iloc[post_start]['Close'] - 1) * 100
                
                # Event day return
                event_day_return = hist_data.iloc[pre_event_idx]['Daily_Return'] * 100
                
                # Volume spike analysis
                event_volume = hist_data.iloc[pre_event_idx]['Volume']
                avg_volume = hist_data.iloc[pre_start:pre_end]['Volume'].mean()
                volume_spike = (event_volume / avg_volume) if avg_volume > 0 else 1
                
                event_impacts.append({
                    'event': event['event'],
                    'date': event['date'],
                    'type': event['type'],
                    'symbol': symbol,
                    'sector': sector,
                    'pre_event_return': pre_return,
                    'event_day_return': event_day_return,
                    'post_event_return': post_return,
                    'volume_spike': volume_spike,
                    'expected_impact': event.get('impact', 'unknown'),
                    'relevant_sector': sector.lower() in [s.lower() for s in event.get('sectors', [])]
                })
        
        print(f"✅ Calculated {len(event_impacts)} event-stock impact combinations")
        return event_impacts
    
    def build_predictive_patterns(self, event_impacts: List[Dict]) -> Dict:
        """
        Build predictive patterns from historical event analysis
        
        LEARNING: Pattern Recognition
        - Which event types have strongest market impact
        - Sector-specific event sensitivities
        - Recovery time patterns
        - Predictive indicators for future events
        """
        print("🔮 Building predictive patterns...")
        
        df = pd.DataFrame(event_impacts)
        
        patterns = {
            'event_type_impact': {},
            'sector_sensitivity': {},
            'recovery_patterns': {},
            'volume_indicators': {}
        }
        
        # Event type impact analysis
        for event_type in df['type'].unique():
            type_data = df[df['type'] == event_type]
            
            patterns['event_type_impact'][event_type] = {
                'avg_event_day_return': type_data['event_day_return'].mean(),
                'avg_post_event_return': type_data['post_event_return'].mean(),
                'volatility': type_data['event_day_return'].std(),
                'sample_size': len(type_data)
            }
        
        # Sector sensitivity analysis
        for sector in df['sector'].unique():
            sector_data = df[df['sector'] == sector]
            
            patterns['sector_sensitivity'][sector] = {
                'most_sensitive_events': sector_data.groupby('type')['event_day_return'].mean().abs().sort_values(ascending=False).head(3).to_dict(),
                'avg_recovery_time': self._estimate_recovery_time(sector_data),
                'volatility_during_events': sector_data['event_day_return'].std()
            }
        
        # Volume spike patterns
        high_volume_events = df[df['volume_spike'] > 2.0]  # 2x normal volume
        patterns['volume_indicators'] = {
            'high_volume_event_types': high_volume_events['type'].value_counts().to_dict(),
            'volume_spike_correlation': df[['volume_spike', 'event_day_return']].corr().iloc[0, 1]
        }
        
        print("✅ Predictive patterns built successfully")
        return patterns
    
    def _estimate_recovery_time(self, sector_data: pd.DataFrame) -> float:
        """Estimate average recovery time for a sector"""
        # Simplified recovery time calculation
        negative_events = sector_data[sector_data['event_day_return'] < -2]  # Events with >2% drop
        if len(negative_events) == 0:
            return 0
        
        # Look at how long it takes to recover (simplified)
        recovery_estimates = []
        for _, event in negative_events.iterrows():
            if event['post_event_return'] > abs(event['event_day_return']):
                recovery_estimates.append(10)  # Recovered within 10 days
            else:
                recovery_estimates.append(30)  # Took longer than 10 days
        
        return np.mean(recovery_estimates) if recovery_estimates else 30
    
    def generate_intelligence_report(self, market_data: Dict, events: List[Dict], 
                                   event_impacts: List[Dict], patterns: Dict) -> str:
        """
        Generate comprehensive intelligence report
        
        LEARNING: Synthesis & Reporting
        - How to synthesize complex data into actionable insights
        - What metrics matter most for different stakeholders
        - How to communicate uncertainty and confidence levels
        """
        print("📋 Generating comprehensive intelligence report...")
        
        report = []
        report.append("🎯 COMPREHENSIVE MARKET INTELLIGENCE REPORT")
        report.append("=" * 60)
        report.append(f"📅 Analysis Period: {self.start_date} to {self.end_date}")
        report.append(f"📊 Symbols Analyzed: {len(market_data)}")
        report.append(f"🌍 Major Events: {len(events)}")
        report.append(f"📈 Event-Stock Combinations: {len(event_impacts)}")
        report.append("")
        
        # Market Overview
        report.append("📊 MARKET OVERVIEW")
        report.append("-" * 30)
        total_returns = []
        for symbol, data in market_data.items():
            if not data['historical_data'].empty:
                total_return = (data['historical_data']['Close'].iloc[-1] / data['historical_data']['Close'].iloc[0] - 1) * 100
                total_returns.append(total_return)
                report.append(f"{symbol}: {total_return:+.1f}% (10+ year return)")
        
        if total_returns:
            report.append(f"Average Return: {np.mean(total_returns):+.1f}%")
        report.append("")
        
        # Event Impact Analysis
        report.append("🌍 EVENT IMPACT ANALYSIS")
        report.append("-" * 30)
        for event_type, stats in patterns['event_type_impact'].items():
            report.append(f"{event_type.upper()}:")
            report.append(f"  Avg Event Day Impact: {stats['avg_event_day_return']:+.2f}%")
            report.append(f"  Avg 10-Day Recovery: {stats['avg_post_event_return']:+.2f}%")
            report.append(f"  Volatility: {stats['volatility']:.2f}%")
            report.append(f"  Sample Size: {stats['sample_size']} events")
            report.append("")
        
        # Sector Sensitivity
        report.append("🏭 SECTOR SENSITIVITY ANALYSIS")
        report.append("-" * 30)
        for sector, stats in patterns['sector_sensitivity'].items():
            if stats['most_sensitive_events']:
                report.append(f"{sector.upper()}:")
                report.append(f"  Most Sensitive to: {list(stats['most_sensitive_events'].keys())[:2]}")
                report.append(f"  Avg Recovery Time: {stats['avg_recovery_time']:.0f} days")
                report.append(f"  Event Volatility: {stats['volatility_during_events']:.2f}%")
                report.append("")
        
        # Key Insights
        report.append("💡 KEY INSIGHTS & PREDICTIONS")
        report.append("-" * 30)
        
        # Find most impactful event types
        most_negative = min(patterns['event_type_impact'].items(), 
                          key=lambda x: x[1]['avg_event_day_return'])
        most_positive = max(patterns['event_type_impact'].items(), 
                          key=lambda x: x[1]['avg_event_day_return'])
        
        report.append(f"📉 Most Negative Event Type: {most_negative[0]} ({most_negative[1]['avg_event_day_return']:+.2f}%)")
        report.append(f"📈 Most Positive Event Type: {most_positive[0]} ({most_positive[1]['avg_event_day_return']:+.2f}%)")
        report.append("")
        
        # Volume insights
        vol_corr = patterns['volume_indicators']['volume_spike_correlation']
        report.append(f"📊 Volume-Return Correlation: {vol_corr:.3f}")
        if vol_corr > 0.1:
            report.append("   → High volume often accompanies large price moves")
        elif vol_corr < -0.1:
            report.append("   → High volume often accompanies price reversals")
        else:
            report.append("   → Volume and returns show weak correlation")
        report.append("")
        
        # Recommendations
        report.append("🎯 INVESTMENT RECOMMENDATIONS")
        report.append("-" * 30)
        report.append("1. DEFENSIVE STRATEGY:")
        report.append("   → Monitor political/policy events closely")
        report.append("   → Reduce exposure before known events (budget, elections)")
        report.append("   → Keep cash reserves for post-crash opportunities")
        report.append("")
        report.append("2. OPPORTUNISTIC STRATEGY:")
        report.append("   → Buy during panic selling (COVID, war events)")
        report.append("   → Focus on sectors with fast recovery patterns")
        report.append("   → Use volume spikes as entry/exit signals")
        report.append("")
        report.append("3. SECTOR ROTATION:")
        most_volatile_sector = max(patterns['sector_sensitivity'].items(), 
                                 key=lambda x: x[1]['volatility_during_events'])
        report.append(f"   → Avoid {most_volatile_sector[0]} during uncertainty")
        report.append("   → Focus on defensive sectors during global crises")
        report.append("   → Overweight cyclicals after policy announcements")
        
        return "\n".join(report)

# Example usage and testing
if __name__ == "__main__":
    print("🚀 LAUNCHING COMPREHENSIVE MARKET INTELLIGENCE")
    print("=" * 70)
    
    # Initialize the engine
    intelligence = ComprehensiveMarketIntelligence()
    
    # Define symbols to analyze (mix of sectors)
    indian_symbols = [
        "RELIANCE.NS",   # Oil & Gas
        "TCS.NS",        # IT Services  
        "HDFCBANK.NS",   # Banking
        "INFY.NS",       # IT Services
        "HINDUNILVR.NS", # FMCG
        "ITC.NS",        # FMCG/Tobacco
        "SBIN.NS",       # Banking
        "BHARTIARTL.NS", # Telecom
        "MARUTI.NS",     # Auto
        "ADANIGREEN.NS"  # Renewable Energy
    ]
    
    # Step 1: Get comprehensive market data
    print("STEP 1: Gathering 10+ years of market data...")
    market_data = intelligence.get_comprehensive_market_data(indian_symbols[:5])  # Start with 5 for demo
    
    # Step 2: Build major events timeline  
    print("\nSTEP 2: Building major events timeline...")
    major_events = intelligence.get_major_events_timeline()
    
    # Step 3: Analyze Google Trends (if available)
    print("\nSTEP 3: Analyzing Google Trends...")
    trend_keywords = ["stock market", "investment", "mutual funds", "SIP", "inflation"]
    trends_data = intelligence.analyze_google_trends(trend_keywords)
    
    # Step 4: Calculate event impacts
    print("\nSTEP 4: Calculating event impacts...")
    event_impacts = intelligence.calculate_event_impact(market_data, major_events)
    
    # Step 5: Build predictive patterns
    print("\nSTEP 5: Building predictive patterns...")
    patterns = intelligence.build_predictive_patterns(event_impacts)
    
    # Step 6: Generate comprehensive report
    print("\nSTEP 6: Generating intelligence report...")
    report = intelligence.generate_intelligence_report(market_data, major_events, event_impacts, patterns)
    
    print("\n" + "=" * 70)
    print(report)
    print("=" * 70)
    
    print("\n💡 NEXT STEPS:")
    print("✅ Extend to more symbols and longer time periods")
    print("✅ Add real-time news sentiment analysis")
    print("✅ Integrate economic indicators (GDP, inflation)")
    print("✅ Build automated alert system for new events")
    print("✅ Create predictive models for event-driven trading")
    
    print("\n🚀 BUSINESS OPPORTUNITY:")
    print("- First comprehensive event-correlation engine for Indian markets")
    print("- Institutional-grade research at retail prices")
    print("- Explainable AI for market movements")
    print("- Premium subscription: ₹999/month for full intelligence")
    print("- Target: Fund managers, HNI investors, financial advisors")
    
    print(f"\n🎯 This analysis covers {len(event_impacts)} event-stock combinations")
    print("Ready to build the most comprehensive market intelligence platform in India! 🇮🇳")
