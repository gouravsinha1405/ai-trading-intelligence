"""
Indian Market Portfolio Monitor - NSE/BSE Integration

LEARNING OBJECTIVES:
1. Indian Stock Exchange Integration (NSE/BSE)
2. Mutual Fund Tracking (SIP optimization)
3. INR Currency Handling
4. Indian Market Hours & Holidays
5. SEBI Compliance Considerations

INDIAN MARKET FEATURES:
- NSE/BSE stock symbols (.NS, .BO suffixes)
- Mutual fund NAV tracking
- Market timings (9:15 AM - 3:30 PM IST)
- Currency in INR (₹)
- Indian market indices (NIFTY, SENSEX)
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import requests
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

class IndianMarketPortfolio:
    """
    Portfolio Monitor specifically designed for Indian markets
    
    LEARNING FOCUS:
    - How Indian stock symbols work (.NS, .BO)
    - Mutual fund NAV integration
    - Currency handling (INR)
    - Market timing considerations
    """
    
    def __init__(self):
        self.positions = {}
        self.mutual_funds = {}
        self.currency = "₹"
        
        # Indian market indices for beta calculation
        self.market_indices = {
            'NIFTY': '^NSEI',
            'SENSEX': '^BSESN',
            'BANKNIFTY': '^NSEBANK'
        }
        
        print("🇮🇳 Indian Market Portfolio Monitor initialized")
    
    def add_indian_stock(self, symbol: str, quantity: float, avg_price: float, exchange: str = "NSE"):
        """
        Add Indian stock position
        
        LEARNING: Indian Stock Symbols
        - NSE stocks: RELIANCE.NS, TCS.NS, HDFCBANK.NS
        - BSE stocks: RELIANCE.BO, TCS.BO (same company, different exchange)
        - Choose NSE for better liquidity typically
        """
        try:
            # Format symbol for Indian exchanges
            if exchange.upper() == "NSE":
                formatted_symbol = f"{symbol}.NS"
            elif exchange.upper() == "BSE":
                formatted_symbol = f"{symbol}.BO"
            else:
                formatted_symbol = symbol  # Assume already formatted
            
            # Get current price
            ticker = yf.Ticker(formatted_symbol)
            current_data = ticker.history(period="1d")
            
            if current_data.empty:
                # Try alternative exchange if first fails
                alt_symbol = f"{symbol}.BO" if exchange == "NSE" else f"{symbol}.NS"
                ticker = yf.Ticker(alt_symbol)
                current_data = ticker.history(period="1d")
                formatted_symbol = alt_symbol
            
            if current_data.empty:
                raise ValueError(f"Cannot get data for {symbol} on {exchange}")
            
            current_price = current_data['Close'].iloc[-1]
            market_value = quantity * current_price
            unrealized_pnl = quantity * (current_price - avg_price)
            unrealized_pnl_pct = (current_price - avg_price) / avg_price * 100
            
            # Get additional Indian market info
            info = ticker.info
            company_name = info.get('longName', symbol)
            sector = info.get('sector', 'Unknown')
            
            self.positions[symbol] = {
                'symbol': formatted_symbol,
                'company_name': company_name,
                'sector': sector,
                'exchange': exchange,
                'quantity': quantity,
                'avg_price': avg_price,
                'current_price': current_price,
                'market_value': market_value,
                'unrealized_pnl': unrealized_pnl,
                'unrealized_pnl_pct': unrealized_pnl_pct
            }
            
            print(f"✅ Added {quantity} shares of {company_name} ({symbol})")
            print(f"   Exchange: {exchange} | Current: {self.currency}{current_price:.2f} | P&L: {unrealized_pnl_pct:+.1f}%")
            
        except Exception as e:
            print(f"❌ Error adding {symbol}: {e}")
    
    def add_mutual_fund_sip(self, fund_code: str, monthly_sip: float, start_date: str):
        """
        Add mutual fund SIP position
        
        LEARNING: Mutual Fund Integration
        - Most Indians invest through SIPs (Systematic Investment Plans)
        - NAV (Net Asset Value) changes daily
        - Need to track multiple purchases at different NAVs
        - SIP optimization is key for wealth building
        """
        try:
            # For now, we'll simulate MF data (real API integration would use AMFI/fund house APIs)
            # In production, you'd integrate with:
            # - MFU (MF Utilities) API
            # - Individual AMC APIs  
            # - Third-party providers like MFCentral
            
            self.mutual_funds[fund_code] = {
                'fund_code': fund_code,
                'fund_name': f"Fund {fund_code}",  # Would get from API
                'monthly_sip': monthly_sip,
                'start_date': start_date,
                'current_nav': 45.50,  # Would get from API
                'total_invested': 0,
                'current_value': 0,
                'units': 0
            }
            
            print(f"✅ Added SIP: {self.currency}{monthly_sip:,.0f}/month in {fund_code}")
            
        except Exception as e:
            print(f"❌ Error adding MF SIP {fund_code}: {e}")
    
    def calculate_indian_market_metrics(self):
        """
        Calculate metrics specific to Indian market
        
        LEARNING: Indian Market Characteristics
        - Higher volatility than developed markets
        - Sector concentration (IT, Banking, Pharma)
        - Currency considerations (INR depreciation)
        - Regulatory environment (SEBI rules)
        """
        if not self.positions:
            return {}
        
        total_value = sum(pos['market_value'] for pos in self.positions.values())
        
        # Sector allocation (important in Indian context)
        sector_allocation = {}
        for pos in self.positions.values():
            sector = pos.get('sector', 'Unknown')
            if sector not in sector_allocation:
                sector_allocation[sector] = 0
            sector_allocation[sector] += pos['market_value']
        
        # Convert to percentages
        for sector in sector_allocation:
            sector_allocation[sector] = (sector_allocation[sector] / total_value) * 100
        
        # Check for typical Indian market concentrations
        it_exposure = sector_allocation.get('Technology', 0)
        banking_exposure = sector_allocation.get('Financial Services', 0) + sector_allocation.get('Banks', 0)
        
        # Calculate beta vs NIFTY
        nifty_beta = self._calculate_nifty_beta()
        
        return {
            'total_value_inr': total_value,
            'sector_allocation': sector_allocation,
            'it_concentration': it_exposure,
            'banking_concentration': banking_exposure,
            'nifty_beta': nifty_beta,
            'high_risk_concentration': max(sector_allocation.values()) if sector_allocation else 0
        }
    
    def _calculate_nifty_beta(self):
        """Calculate portfolio beta vs NIFTY 50"""
        try:
            # Get NIFTY data
            nifty = yf.Ticker('^NSEI')
            nifty_data = nifty.history(period='3mo')['Close']
            nifty_returns = nifty_data.pct_change().dropna()
            
            # Calculate portfolio returns (simplified)
            portfolio_returns = []
            for symbol, pos in self.positions.items():
                try:
                    ticker = yf.Ticker(pos['symbol'])
                    stock_data = ticker.history(period='3mo')['Close']
                    stock_returns = stock_data.pct_change().dropna()
                    
                    # Weight by position size
                    weight = pos['market_value'] / sum(p['market_value'] for p in self.positions.values())
                    weighted_returns = stock_returns * weight
                    
                    if len(portfolio_returns) == 0:
                        portfolio_returns = weighted_returns.copy()
                    else:
                        # Align indices and add
                        aligned_returns = weighted_returns.reindex(portfolio_returns.index, fill_value=0)
                        portfolio_returns = portfolio_returns.add(aligned_returns, fill_value=0)
                        
                except Exception as e:
                    continue
            
            if len(portfolio_returns) == 0:
                return 1.0
            
            # Align portfolio and NIFTY returns
            aligned_portfolio = portfolio_returns.reindex(nifty_returns.index).dropna()
            aligned_nifty = nifty_returns.reindex(aligned_portfolio.index)
            
            if len(aligned_portfolio) < 20:
                return 1.0
            
            # Calculate beta
            covariance = np.cov(aligned_portfolio, aligned_nifty)[0, 1]
            market_variance = np.var(aligned_nifty)
            
            beta = covariance / market_variance if market_variance > 0 else 1.0
            return beta
            
        except Exception as e:
            print(f"Warning: Could not calculate NIFTY beta: {e}")
            return 1.0
    
    def generate_indian_market_alerts(self):
        """
        Generate alerts specific to Indian market conditions
        
        LEARNING: Indian Market Risk Factors
        - Sector concentration (especially IT/Banking)
        - Regulatory changes (budget impacts, policy changes)
        - Currency depreciation effects
        - Market timing (festivals, results seasons)
        """
        alerts = []
        metrics = self.calculate_indian_market_metrics()
        
        # IT sector concentration alert
        if metrics.get('it_concentration', 0) > 30:
            alerts.append({
                'type': 'SECTOR_CONCENTRATION',
                'level': 'HIGH',
                'message': f"High IT sector exposure: {metrics['it_concentration']:.1f}%",
                'recommendation': "Consider diversifying into other sectors (Pharma, FMCG, Energy)"
            })
        
        # Banking sector concentration alert  
        if metrics.get('banking_concentration', 0) > 40:
            alerts.append({
                'type': 'SECTOR_CONCENTRATION',
                'level': 'HIGH',
                'message': f"High banking sector exposure: {metrics['banking_concentration']:.1f}%",
                'recommendation': "Banking sector sensitive to interest rates and NPAs"
            })
        
        # Overall concentration alert
        if metrics.get('high_risk_concentration', 0) > 50:
            alerts.append({
                'type': 'CONCENTRATION_RISK',
                'level': 'CRITICAL',
                'message': f"Single sector dominance: {metrics['high_risk_concentration']:.1f}%",
                'recommendation': "Urgent: Diversify across multiple sectors"
            })
        
        # High beta alert (volatile vs NIFTY)
        nifty_beta = metrics.get('nifty_beta', 1.0)
        if nifty_beta > 1.5:
            alerts.append({
                'type': 'HIGH_BETA',
                'level': 'MEDIUM',
                'message': f"Portfolio beta vs NIFTY: {nifty_beta:.2f} (high volatility)",
                'recommendation': "Consider adding defensive stocks or bonds"
            })
        
        return alerts
    
    def get_indian_portfolio_summary(self):
        """Generate comprehensive Indian market portfolio summary"""
        if not self.positions and not self.mutual_funds:
            return "Portfolio is empty"
        
        # Stock positions summary
        stocks_summary = ""
        if self.positions:
            total_stocks_value = sum(pos['market_value'] for pos in self.positions.values())
            total_stocks_cost = sum(pos['quantity'] * pos['avg_price'] for pos in self.positions.values())
            stocks_pnl = total_stocks_value - total_stocks_cost
            stocks_pnl_pct = (stocks_pnl / total_stocks_cost * 100) if total_stocks_cost > 0 else 0
            
            stocks_summary = f"""
📈 EQUITY PORTFOLIO
{'='*50}
Total Value: {self.currency}{total_stocks_value:,.2f}
Total Invested: {self.currency}{total_stocks_cost:,.2f}
Total P&L: {self.currency}{stocks_pnl:,.2f} ({stocks_pnl_pct:+.2f}%)

📊 STOCK POSITIONS ({len(self.positions)}):
{'-'*50}"""
            
            for symbol, pos in sorted(self.positions.items(), key=lambda x: x[1]['market_value'], reverse=True):
                weight = pos['market_value'] / total_stocks_value * 100
                stocks_summary += f"""
{symbol} ({pos['exchange']}): {weight:.1f}% | {self.currency}{pos['market_value']:,.0f} | {pos['unrealized_pnl_pct']:+.1f}%
  {pos['company_name']} | {pos['sector']}"""
        
        # Mutual funds summary
        mf_summary = ""
        if self.mutual_funds:
            mf_summary = f"""

💰 MUTUAL FUNDS
{'='*50}"""
            for fund_code, mf in self.mutual_funds.items():
                mf_summary += f"""
{fund_code}: SIP {self.currency}{mf['monthly_sip']:,.0f}/month
  Current NAV: {self.currency}{mf['current_nav']:.2f}"""
        
        # Market analysis
        metrics = self.calculate_indian_market_metrics()
        alerts = self.generate_indian_market_alerts()
        
        analysis_summary = f"""

🔍 INDIAN MARKET ANALYSIS
{'='*50}
NIFTY Beta: {metrics.get('nifty_beta', 1.0):.2f}
Top Sector: {max(metrics.get('sector_allocation', {'None': 0}), key=metrics.get('sector_allocation', {'None': 0}).get)} ({max(metrics.get('sector_allocation', {}).values(), default=0):.1f}%)"""
        
        if alerts:
            analysis_summary += f"""

🚨 RISK ALERTS ({len(alerts)}):
{'-'*30}"""
            for alert in alerts[:3]:  # Show top 3 alerts
                analysis_summary += f"""
{alert['type']}: {alert['message']}
Action: {alert['recommendation']}"""
        
        return stocks_summary + mf_summary + analysis_summary

# Example usage with popular Indian stocks
if __name__ == "__main__":
    print("🇮🇳 LEARNING SESSION: Indian Market Portfolio")
    print("="*60)
    
    # Create Indian market portfolio
    portfolio = IndianMarketPortfolio()
    
    # Add popular Indian stocks
    print("📊 Building Indian stock portfolio...")
    
    indian_stocks = [
        ("RELIANCE", 100, 2200.0, "NSE"),    # Reliance Industries
        ("TCS", 50, 3200.0, "NSE"),          # Tata Consultancy Services  
        ("HDFCBANK", 75, 1500.0, "NSE"),     # HDFC Bank
        ("INFY", 60, 1400.0, "NSE"),         # Infosys
        ("ITC", 200, 250.0, "NSE"),          # ITC Limited
    ]
    
    for symbol, quantity, avg_price, exchange in indian_stocks:
        portfolio.add_indian_stock(symbol, quantity, avg_price, exchange)
    
    # Add mutual fund SIPs
    print("\n💰 Adding mutual fund SIPs...")
    portfolio.add_mutual_fund_sip("HDFC_TOP_100", 10000, "2024-01-01")
    portfolio.add_mutual_fund_sip("ICICI_BLUE_CHIP", 5000, "2024-01-01")
    
    # Display comprehensive summary
    print(portfolio.get_indian_portfolio_summary())
    
    print("\n" + "="*60)
    print("🎯 INDIAN MARKET CONCEPTS LEARNED:")
    print("✅ NSE/BSE stock symbol formatting")
    print("✅ Indian company and sector identification")
    print("✅ Mutual fund SIP tracking")
    print("✅ NIFTY beta calculation")
    print("✅ Indian market-specific risk alerts")
    print("✅ Sector concentration analysis")
    print("✅ Currency formatting (INR)")
    
    print("\n💡 INDIAN MARKET INSIGHTS:")
    print("- Sector diversification is crucial (avoid IT/Banking dominance)")
    print("- SIP optimization can significantly boost returns") 
    print("- Market volatility higher than global markets")
    print("- Regulatory changes have major impact")
    
    print("\n🚀 Next: Build SIP optimization and market timing alerts!")
    print("🎯 Revenue opportunity: 3 crore SIP investors need this protection!")
