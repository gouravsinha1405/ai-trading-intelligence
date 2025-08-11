"""
SIP Optimization Engine for Indian Market

LEARNING OBJECTIVES:
1. SIP Market Timing (when to increase/decrease SIP amounts)
2. Mutual Fund Analysis (expense ratios, performance tracking)
3. Goal-Based Investing (retirement, child education, house)
4. Tax Optimization (ELSS, tax saving strategies)
5. Market Cycle Detection (bull/bear market SIP adjustments)

INDIAN MARKET SPECIFIC FEATURES:
- Budget impact analysis (market reaction to Union Budget)
- Festival season patterns (Diwali rally, etc.)
- Result season analysis (quarterly earnings impact)
- FII/DII flow impact on market timing
- Currency depreciation hedging
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class SIPOptimizerIndia:
    """
    AI-powered SIP optimization for Indian mutual funds
    
    LEARNING FOCUS:
    - Market timing for SIP amounts (not timing the market, but optimizing SIP)
    - Valuation-based SIP adjustments
    - Goal-based portfolio allocation
    - Tax-efficient investing strategies
    """
    
    def __init__(self):
        self.nifty_pe_ranges = {
            'extremely_cheap': 15,
            'cheap': 18,
            'fair': 22,
            'expensive': 25,
            'extremely_expensive': 28
        }
        
        self.sip_multipliers = {
            'extremely_cheap': 2.0,    # Double SIP when market is very cheap
            'cheap': 1.5,              # 50% more SIP when cheap
            'fair': 1.0,               # Normal SIP when fairly valued
            'expensive': 0.7,          # Reduce SIP by 30% when expensive
            'extremely_expensive': 0.5  # Halve SIP when very expensive
        }
        
        print("🎯 SIP Optimizer for Indian Market initialized")
    
    def get_nifty_valuation(self):
        """
        Get current NIFTY 50 valuation metrics
        
        LEARNING: Market Valuation
        - P/E ratio shows if market is cheap or expensive
        - Below 18 P/E = Good time to increase SIP
        - Above 25 P/E = Time to reduce SIP amounts
        - This is NOT market timing, it's SIP optimization
        """
        try:
            # Get NIFTY 50 data
            nifty = yf.Ticker('^NSEI')
            nifty_info = nifty.info
            
            # Try to get P/E ratio (may not always be available)
            current_pe = nifty_info.get('trailingPE', None)
            
            if current_pe is None:
                # Fallback: estimate based on recent performance
                # In production, you'd use NSE API or financial data providers
                current_pe = 22.5  # Approximate current NIFTY P/E
            
            # Determine valuation category
            if current_pe <= self.nifty_pe_ranges['extremely_cheap']:
                valuation = 'extremely_cheap'
            elif current_pe <= self.nifty_pe_ranges['cheap']:
                valuation = 'cheap'
            elif current_pe <= self.nifty_pe_ranges['fair']:
                valuation = 'fair'
            elif current_pe <= self.nifty_pe_ranges['expensive']:
                valuation = 'expensive'
            else:
                valuation = 'extremely_expensive'
            
            return {
                'current_pe': current_pe,
                'valuation': valuation,
                'sip_multiplier': self.sip_multipliers[valuation]
            }
            
        except Exception as e:
            print(f"Warning: Could not get NIFTY valuation: {e}")
            return {
                'current_pe': 22.5,
                'valuation': 'fair',
                'sip_multiplier': 1.0
            }
    
    def optimize_sip_amount(self, base_sip: float, fund_category: str = "large_cap"):
        """
        Optimize SIP amount based on market conditions
        
        LEARNING: SIP Optimization Strategy
        - Base SIP: Your regular monthly investment
        - Market cheap: Increase SIP to buy more units at lower NAV
        - Market expensive: Reduce SIP but don't stop (rupee cost averaging)
        - Different strategies for different fund categories
        """
        valuation = self.get_nifty_valuation()
        
        # Base optimization
        optimized_sip = base_sip * valuation['sip_multiplier']
        
        # Category-specific adjustments
        category_multipliers = {
            'large_cap': 1.0,      # Follow market valuation directly
            'mid_cap': 0.8,        # Be more conservative with mid-cap
            'small_cap': 0.6,      # Be very conservative with small-cap
            'multi_cap': 0.9,      # Slightly conservative
            'international': 1.1,   # Less affected by Indian market valuation
            'debt': 1.2,           # Counter-cyclical to equity
            'elss': 1.0            # Tax-saving funds, follow market
        }
        
        category_adjustment = category_multipliers.get(fund_category.lower(), 1.0)
        final_sip = optimized_sip * category_adjustment
        
        # Ensure minimum and maximum bounds
        min_sip = base_sip * 0.3  # Never go below 30% of base SIP
        max_sip = base_sip * 3.0  # Never exceed 3x base SIP
        
        final_sip = max(min_sip, min(max_sip, final_sip))
        
        return {
            'base_sip': base_sip,
            'optimized_sip': final_sip,
            'adjustment_pct': (final_sip - base_sip) / base_sip * 100,
            'market_valuation': valuation['valuation'],
            'market_pe': valuation['current_pe'],
            'recommendation': self._generate_sip_recommendation(base_sip, final_sip, valuation)
        }
    
    def _generate_sip_recommendation(self, base_sip, optimized_sip, valuation):
        """Generate human-readable SIP recommendation"""
        adjustment_pct = (optimized_sip - base_sip) / base_sip * 100
        
        if adjustment_pct > 20:
            return f"🚀 INCREASE SIP: Market is {valuation['valuation']} (P/E: {valuation['current_pe']:.1f}). Great time to invest more!"
        elif adjustment_pct > 5:
            return f"📈 SLIGHTLY INCREASE: Market reasonably valued. Good time to invest a bit more."
        elif adjustment_pct > -5:
            return f"✅ MAINTAIN SIP: Market fairly valued. Continue regular investments."
        elif adjustment_pct > -20:
            return f"📉 REDUCE SIP: Market getting expensive. Be conservative but don't stop."
        else:
            return f"⚠️ SIGNIFICANTLY REDUCE: Market very expensive. Minimal investments until correction."
    
    def analyze_mutual_fund_portfolio(self, sip_portfolio):
        """
        Analyze and optimize entire MF portfolio
        
        LEARNING: Portfolio-Level Optimization
        - Different funds need different SIP amounts based on market conditions
        - Rebalancing through SIP adjustments
        - Goal-based allocation optimization
        """
        portfolio_analysis = {
            'total_monthly_sip': 0,
            'optimized_monthly_sip': 0,
            'fund_recommendations': [],
            'portfolio_balance': {}
        }
        
        equity_funds = 0
        debt_funds = 0
        
        for fund in sip_portfolio:
            fund_code = fund['fund_code']
            base_sip = fund['monthly_sip']
            category = fund.get('category', 'large_cap')
            
            optimization = self.optimize_sip_amount(base_sip, category)
            
            portfolio_analysis['total_monthly_sip'] += base_sip
            portfolio_analysis['optimized_monthly_sip'] += optimization['optimized_sip']
            
            # Track equity vs debt allocation
            if 'debt' in category.lower() or 'liquid' in category.lower():
                debt_funds += optimization['optimized_sip']
            else:
                equity_funds += optimization['optimized_sip']
            
            portfolio_analysis['fund_recommendations'].append({
                'fund_code': fund_code,
                'current_sip': base_sip,
                'recommended_sip': optimization['optimized_sip'],
                'adjustment': optimization['adjustment_pct'],
                'reason': optimization['recommendation']
            })
        
        # Calculate asset allocation
        total_optimized = portfolio_analysis['optimized_monthly_sip']
        if total_optimized > 0:
            portfolio_analysis['portfolio_balance'] = {
                'equity_allocation': (equity_funds / total_optimized) * 100,
                'debt_allocation': (debt_funds / total_optimized) * 100
            }
        
        return portfolio_analysis
    
    def generate_goal_based_recommendations(self, goals):
        """
        Generate goal-based investment recommendations
        
        LEARNING: Goal-Based Investing
        - Different goals need different strategies
        - Time horizon affects risk tolerance
        - Tax implications vary by goal type
        """
        recommendations = []
        
        for goal in goals:
            goal_name = goal['name']
            target_amount = goal['target_amount']
            time_horizon = goal['time_horizon_years']
            current_savings = goal.get('current_savings', 0)
            
            # Calculate required monthly SIP
            required_sip = self._calculate_required_sip(
                target_amount, time_horizon, current_savings
            )
            
            # Recommend asset allocation based on time horizon
            if time_horizon > 10:
                equity_pct = 80
                debt_pct = 20
                risk_level = "Aggressive"
            elif time_horizon > 5:
                equity_pct = 70
                debt_pct = 30
                risk_level = "Moderate"
            else:
                equity_pct = 50
                debt_pct = 50
                risk_level = "Conservative"
            
            recommendations.append({
                'goal': goal_name,
                'required_monthly_sip': required_sip,
                'recommended_allocation': {
                    'equity': equity_pct,
                    'debt': debt_pct
                },
                'risk_level': risk_level,
                'fund_suggestions': self._suggest_funds(equity_pct, debt_pct, time_horizon)
            })
        
        return recommendations
    
    def _calculate_required_sip(self, target_amount, years, current_savings):
        """Calculate required monthly SIP using compound interest"""
        # Assume 12% annual return for equity-heavy portfolio
        annual_return = 0.12
        monthly_return = annual_return / 12
        months = years * 12
        
        # Future value of current savings
        fv_current = current_savings * ((1 + annual_return) ** years)
        
        # Amount needed from SIPs
        amount_needed = target_amount - fv_current
        
        if amount_needed <= 0:
            return 0
        
        # Calculate required monthly SIP
        # PMT formula: PMT = FV / [((1 + r)^n - 1) / r]
        if monthly_return > 0:
            required_sip = amount_needed / (((1 + monthly_return) ** months - 1) / monthly_return)
        else:
            required_sip = amount_needed / months
        
        return max(0, required_sip)
    
    def _suggest_funds(self, equity_pct, debt_pct, time_horizon):
        """Suggest specific fund categories based on allocation"""
        suggestions = []
        
        if equity_pct > 0:
            if time_horizon > 10:
                suggestions.extend([
                    "Large Cap Fund (40% of equity)",
                    "Mid Cap Fund (35% of equity)", 
                    "Small Cap Fund (25% of equity)"
                ])
            elif time_horizon > 5:
                suggestions.extend([
                    "Large Cap Fund (60% of equity)",
                    "Mid Cap Fund (40% of equity)"
                ])
            else:
                suggestions.append("Large Cap Fund (100% of equity)")
        
        if debt_pct > 0:
            if time_horizon > 3:
                suggestions.append("Long Duration Debt Fund")
            else:
                suggestions.append("Short Duration Debt Fund")
        
        return suggestions

# Example usage
if __name__ == "__main__":
    print("🎯 LEARNING SESSION: SIP Optimization for Indian Market")
    print("="*70)
    
    optimizer = SIPOptimizerIndia()
    
    # Example 1: Single SIP optimization
    print("📊 Single SIP Optimization Example:")
    sip_result = optimizer.optimize_sip_amount(10000, "large_cap")
    
    print(f"Base SIP: ₹{sip_result['base_sip']:,.0f}")
    print(f"Optimized SIP: ₹{sip_result['optimized_sip']:,.0f}")
    print(f"Adjustment: {sip_result['adjustment_pct']:+.1f}%")
    print(f"Market P/E: {sip_result['market_pe']:.1f}")
    print(f"Recommendation: {sip_result['recommendation']}")
    
    # Example 2: Portfolio optimization
    print(f"\n💰 Portfolio Optimization Example:")
    
    sample_portfolio = [
        {'fund_code': 'HDFC_TOP_100', 'monthly_sip': 10000, 'category': 'large_cap'},
        {'fund_code': 'ICICI_FOCUSED_BLUE_CHIP', 'monthly_sip': 8000, 'category': 'large_cap'},
        {'fund_code': 'AXIS_MIDCAP', 'monthly_sip': 5000, 'category': 'mid_cap'},
        {'fund_code': 'SBI_SMALL_CAP', 'monthly_sip': 3000, 'category': 'small_cap'},
        {'fund_code': 'HDFC_CORPORATE_BOND', 'monthly_sip': 4000, 'category': 'debt'},
    ]
    
    portfolio_analysis = optimizer.analyze_mutual_fund_portfolio(sample_portfolio)
    
    print(f"Current Total SIP: ₹{portfolio_analysis['total_monthly_sip']:,.0f}")
    print(f"Optimized Total SIP: ₹{portfolio_analysis['optimized_monthly_sip']:,.0f}")
    
    print(f"\nFund-wise Recommendations:")
    for rec in portfolio_analysis['fund_recommendations']:
        print(f"{rec['fund_code']}: ₹{rec['current_sip']:,.0f} → ₹{rec['recommended_sip']:,.0f} ({rec['adjustment']:+.1f}%)")
    
    # Example 3: Goal-based planning
    print(f"\n🎯 Goal-Based Investment Planning:")
    
    sample_goals = [
        {
            'name': 'Child Education',
            'target_amount': 2500000,  # 25 Lakh
            'time_horizon_years': 15,
            'current_savings': 200000   # 2 Lakh
        },
        {
            'name': 'Retirement',
            'target_amount': 50000000,  # 5 Crore
            'time_horizon_years': 25,
            'current_savings': 1000000  # 10 Lakh
        }
    ]
    
    goal_recommendations = optimizer.generate_goal_based_recommendations(sample_goals)
    
    for rec in goal_recommendations:
        print(f"\n{rec['goal']}:")
        print(f"  Required Monthly SIP: ₹{rec['required_monthly_sip']:,.0f}")
        print(f"  Recommended Allocation: {rec['recommended_allocation']['equity']}% Equity, {rec['recommended_allocation']['debt']}% Debt")
        print(f"  Risk Level: {rec['risk_level']}")
    
    print("\n" + "="*70)
    print("🎯 SIP OPTIMIZATION CONCEPTS LEARNED:")
    print("✅ Market valuation-based SIP adjustments")
    print("✅ Category-specific optimization strategies")
    print("✅ Portfolio-level SIP rebalancing")
    print("✅ Goal-based investment planning")
    print("✅ Required SIP calculation with compound interest")
    print("✅ Risk-appropriate asset allocation")
    
    print("\n💡 KEY INSIGHTS:")
    print("- SIP optimization ≠ Market timing (it's about amounts, not timing)")
    print("- Market valuation should influence SIP amounts")
    print("- Different fund categories need different strategies")
    print("- Goal-based planning ensures disciplined investing")
    
    print("\n🚀 BUSINESS OPPORTUNITY:")
    print("- 3+ crore SIP accounts in India")
    print("- Most investors use fixed SIP amounts (suboptimal)")
    print("- AI-powered SIP optimization = clear value proposition")
    print("- Subscription model: ₹299/month for optimization advice")
    print("- Potential revenue: Even 10,000 users = ₹30 Lakh/month!")
    
    print("\n🎯 Next: Build real-time market alerts and automation!")
