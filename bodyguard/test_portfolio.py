"""
Test Portfolio Monitor
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Test our basic functionality without complex imports
import yfinance as yf
import pandas as pd
from datetime import datetime

print("🎓 LEARNING SESSION: Portfolio Monitoring System")
print("="*60)

# Simple portfolio tracking example
class SimplePortfolio:
    def __init__(self):
        self.positions = {}
    
    def add_position(self, symbol, quantity, avg_price):
        # Get current price
        try:
            ticker = yf.Ticker(symbol)
            current_data = ticker.history(period="1d")
            current_price = current_data['Close'].iloc[-1]
            
            market_value = quantity * current_price
            unrealized_pnl = quantity * (current_price - avg_price)
            unrealized_pnl_pct = (current_price - avg_price) / avg_price * 100
            
            self.positions[symbol] = {
                'quantity': quantity,
                'avg_price': avg_price,
                'current_price': current_price,
                'market_value': market_value,
                'unrealized_pnl': unrealized_pnl,
                'unrealized_pnl_pct': unrealized_pnl_pct
            }
            
            print(f"✅ Added {quantity} shares of {symbol} at ${avg_price:.2f}")
            print(f"   Current price: ${current_price:.2f} | P&L: {unrealized_pnl_pct:+.1f}%")
            
        except Exception as e:
            print(f"❌ Error adding {symbol}: {e}")
    
    def get_summary(self):
        if not self.positions:
            return "Portfolio is empty"
        
        total_value = sum(pos['market_value'] for pos in self.positions.values())
        total_cost = sum(pos['quantity'] * pos['avg_price'] for pos in self.positions.values())
        total_pnl = total_value - total_cost
        total_pnl_pct = (total_pnl / total_cost * 100) if total_cost > 0 else 0
        
        summary = f"""
🏦 PORTFOLIO SUMMARY
{'='*50}
Total Value: ${total_value:,.2f}
Total Cost: ${total_cost:,.2f}
Total P&L: ${total_pnl:,.2f} ({total_pnl_pct:+.2f}%)

📊 POSITIONS ({len(self.positions)}):
{'-'*50}"""
        
        for symbol, pos in self.positions.items():
            weight = pos['market_value'] / total_value * 100
            summary += f"""
{symbol}: {weight:.1f}% | ${pos['market_value']:,.0f} | {pos['unrealized_pnl_pct']:+.1f}%"""
        
        return summary

# Test the portfolio
print("📊 Building example portfolio...")

portfolio = SimplePortfolio()

# Add some example positions
example_positions = [
    ("AAPL", 50, 150.0),   # 50 shares at $150 avg
    ("GOOGL", 20, 120.0),  # 20 shares at $120 avg
    ("MSFT", 30, 300.0),   # 30 shares at $300 avg
]

for symbol, quantity, avg_price in example_positions:
    portfolio.add_position(symbol, quantity, avg_price)

# Display results
print(portfolio.get_summary())

print("\n" + "="*60)
print("🎯 PORTFOLIO CONCEPTS LEARNED:")
print("✅ Real-time price fetching with yfinance")
print("✅ Portfolio construction and position tracking")
print("✅ Position weights and market values calculation")
print("✅ Unrealized P&L calculation")
print("✅ Portfolio summary generation")
print("✅ Error handling for missing data")

print("\n💡 KEY INSIGHTS:")
print("- Each position shows current market value vs your cost basis")
print("- Portfolio weights show concentration (diversification)")
print("- P&L percentages show which positions are winners/losers")
print("- This forms the foundation for risk monitoring!")

print("\n🚀 Next: Build AI-powered risk alerts on top of this foundation!")
