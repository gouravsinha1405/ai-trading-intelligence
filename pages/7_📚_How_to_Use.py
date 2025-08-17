# how_to_use.py — Updated tutorial with recent improvements
import streamlit as st

st.set_page_config(page_title="How to Use", page_icon="📚", layout="wide")

# ---------- styles ----------
st.markdown(
    """
<style>
.big-font {font-size:20px !important; font-weight:bold;}
.medium-font {font-size:16px !important; font-weight:bold;}
.highlight {background:#f0f2f6; padding:10px; border-radius:5px; border-left:5px solid #ff6b6b;}
.success-box {        <h3>🎯 Enhanced Strategy Builder</h3>
        <p>Strategy Builder mein ab advanced AI optimization hai multiple objective modes ke saath, improved parameter comparison, aur seamless user experience. Trading strategies build, test, aur optimize kijiye cutting-edge AI assistance ke saath.</p>ckground:#d4edda; padding:15px; border-radius:5px; border-left:5px solid #28a745;}
.warning-box {background:#fff3cd; padding:15px; border-radius:5px; border-left:5px solid #ffc107;}
.info-box {background:#d1ecf1; padding:15px; border-radius:5px; border-left:5px solid #17a2b8;}
.small-dim {color:#666; font-size:14px;}
.new-feature {background:#e8f5e8; padding:10px; border-radius:5px; border-left:5px solid #4CAF50;}
</style>
""",
    unsafe_allow_html=True,
)

st.title("📚 Complete AI Trading Platform Guide")
st.caption("Master every feature of this comprehensive algorithmic trading framework with latest AI enhancements")

# Language selector
language = st.selectbox("🌐 Choose Language / भाषा चुनें", ["English", "Hinglish (हिंग्लिश)"], index=0)

# ---------- API Key Status Check ----------
from src.utils.config import load_config


def check_api_key_status():
    """Check if API key is configured"""
    config = load_config()
    return bool(config.get("groq_api_key"))


def show_api_key_prompt():
    """Show prominent API key setup prompt if not configured"""
    if not check_api_key_status():
        if language == "English":
            st.error(
                """
            🚨 **AI Features Disabled**: Groq API key not configured
            
            **Missing Features:**
            - 🤖 AI Strategy Optimization with Multi-Objective Analysis
            - 🧠 AI Assistant & Intelligent Analysis  
            - 📊 Advanced Performance Insights & Recommendations
            - 💬 Smart Trading Guidance & Real-time Suggestions
            
            ➡️ **Set up your free API key in the "Setup & Configuration" section below**
            """
            )
        else:
            st.error(
                """
            🚨 **AI Features Disable Hai**: Groq API key setup nahi hai
            
            **Missing Features:**
            - 🤖 AI Strategy Optimization with Multi-Objective Analysis
            - 🧠 AI Assistant aur Intelligent Analysis  
            - 📊 Advanced Performance Insights & Recommendations
            - 💬 Smart Trading Guidance & Real-time Suggestions
            
            ➡️ **Apna free API key setup karo "Setup & Configuration" section mein**
            """
            )
        return False
    else:
        if language == "English":
            st.success("✅ **AI Features Active**: All advanced platform features are available!")
        else:
            st.success("✅ **AI Features Active Hai**: Saare advanced platform features available hain!")
        return True


# Show API key status at the top
api_configured = show_api_key_prompt()
st.markdown("---")

# Add new features highlight
st.markdown(
    """
<div class="new-feature">
<h3>🆕 Latest Platform Updates</h3>
<ul>
<li><strong>🎯 Enhanced Strategy Builder</strong>: Multi-objective AI optimization with 4 different modes (Growth, Balanced, Quality, Conservative)</li>
<li><strong>📊 Improved Parameter Comparison</strong>: Side-by-side original vs optimized parameter display</li>
<li><strong>🚀 Better Performance Metrics</strong>: Professional-style metric cards with visual indicators</li>
<li><strong>🔄 Smooth User Experience</strong>: No page refresh during optimization, persistent AI suggestions</li>
<li><strong>📈 Stock Symbol Synchronization</strong>: Quick select dropdown automatically updates analysis</li>
<li><strong>🤖 Advanced AI Integration</strong>: Intelligent suggestions that persist during parameter exploration</li>
</ul>
</div>
""",
    unsafe_allow_html=True,
)

# ---------- sidebar: in-page navigation ----------
if language == "English":
    sections = [
        "Quick Start",
        "Setup & Configuration", 
        "Platform Overview",
        "Latest Features",
        "Strategy Builder (Enhanced)",
        "Trading Basics",
        "Key Concepts",
        "Dashboard",
        "Live Trading",
        "News Analysis", 
        "Backtesting",
        "AI Assistant",
        "Best Practices",
        "Risk Management",
        "Common Pitfalls",
        "Troubleshooting",
        "Privacy & Safety",
    ]
else:
    sections = [
        "Quick Start (शुरुआत)",
        "Setup & Configuration (सेटअप)",
        "Platform Overview (प्लेटफॉर्म की जानकारी)", 
        "Latest Features (नए फीचर्स)",
        "Strategy Builder (Enhanced)",
        "Trading Basics (ट्रेडिंग बेसिक्स)",
        "Key Concepts (मुख्य अवधारणाएं)",
        "Dashboard (डैशबोर्ड)",
        "Live Trading (लाइव ट्रेडिंग)",
        "News Analysis (न्यूज़ एनालिसिस)",
        "Backtesting (बैकटेस्टिंग)",
        "AI Assistant (AI असिस्टेंट)",
        "Best Practices (बेस्ट प्रैक्टिसेज)",
        "Risk Management (रिस्क मैनेजमेंट)",
        "Common Pitfalls (सामान्य गलतियां)",
        "Troubleshooting (समस्या निवारण)",
        "Privacy & Safety (प्राइवेसी और सुरक्षा)",
    ]

section = st.sidebar.radio(
    "📋 Table of Contents" if language == "English" else "📋 विषय सूची",
    sections,
    index=0,
)

# Rest of your existing sidebar API setup code here...
st.sidebar.markdown("---")
if language == "English":
    st.sidebar.markdown("### 🔑 Quick API Setup")
else:
    st.sidebar.markdown("### 🔑 Quick API Setup")

# Load current config for sidebar
sidebar_config = load_config()
sidebar_has_key = bool(sidebar_config.get("groq_api_key"))

if sidebar_has_key:
    st.sidebar.success("✅ AI Features Active" if language == "English" else "✅ AI Features Active Hai")
    test_button_text = "🧪 Test Connection" if language == "English" else "🧪 Connection Test Karo"
    if st.sidebar.button(test_button_text):
        try:
            from src.analysis.ai_analyzer import GroqAnalyzer
            analyzer = GroqAnalyzer(sidebar_config["groq_api_key"])
            st.sidebar.success("✅ Connection successful!" if language == "English" else "✅ Connection successful hai!")
        except Exception as e:
            st.sidebar.error(f"❌ Connection failed: {str(e)}")
else:
    st.sidebar.warning("⚠️ AI Features Disabled" if language == "English" else "⚠️ AI Features Disabled Hai")


# ---------- helpers ----------
def hr():
    st.markdown("---")


# ---------- sections ----------
if section == "Quick Start" or section == "Quick Start (शुरुआत)":
    hr()
    if language == "English":
        st.markdown('<p class="big-font">🚀 Quick Start Guide</p>', unsafe_allow_html=True)
        
        if not check_api_key_status():
            st.warning(
                """
            ⚠️ **Before You Start**: Some features require a free API key setup.
            Don't worry - it takes just 2 minutes and unlocks powerful AI features!
            """
            )

        st.markdown(
            """
        <div class="success-box">
        <h3>⚡ Start Trading in Minutes!</h3>

        <strong>Step 1: Unlock AI Features (2 minutes) 🔑</strong>
        <ol>
        <li>Go to <strong>⚙️ Setup & Configuration</strong> section below</li>
        <li>Get your free Groq API key from <a href="https://console.groq.com" target="_blank">console.groq.com</a></li>
        <li>Enter the key and click "Save & Activate"</li>
        <li>✅ This unlocks AI optimization, analysis, and assistance features</li>
        </ol>

        <strong>Step 2: Explore Market Conditions 📊</strong>
        <ol>
        <li>Visit the <strong>📊 Dashboard</strong> to see current market trends</li>
        <li>Review market sentiment and volatility conditions</li>
        <li>Identify if markets are trending or range-bound</li>
        </ol>

        <strong>Step 3: Build Your First Strategy 🔧</strong>
        <ol>
        <li>Open <strong>🔧 Strategy Builder</strong> from the main menu</li>
        <li>Select a strategy type (Momentum for trending markets)</li>
        <li>Use default parameters to start (you can optimize later)</li>
        <li>Run a historical backtest to see initial performance</li>
        </ol>

        <strong>Step 4: Optimize with AI 🤖</strong>
        <ol>
        <li>Click <strong>🤖 AI Optimize</strong> to improve your strategy</li>
        <li>Choose optimization mode: Growth, Balanced, Quality, or Conservative</li>
        <li>Review AI suggestions and compare original vs optimized parameters</li>
        <li>Apply optimizations and see improved performance metrics</li>
        </ol>

        <strong>Step 5: Practice with Virtual Money 💰</strong>
        <ol>
        <li>Go to <strong>📈 Live Trading</strong> to start virtual trading</li>
        <li>Begin with small position sizes to learn the platform</li>
        <li>Monitor your virtual portfolio performance</li>
        <li>Use the <strong>🤖 AI Assistant</strong> for trading guidance</li>
        </ol>

        <strong>🎯 Congratulations!</strong> You're now ready to explore advanced features and refine your trading strategies.
        </div>
        """,
            unsafe_allow_html=True,
        )
        
    else:  # Hinglish version
        st.markdown('<p class="big-font">🚀 Quick Start Guide (जल्दी शुरुआत)</p>', unsafe_allow_html=True)
        
        st.markdown(
            """
        <div class="success-box">
        <h3>⚡ Minutes Mein Trading Start Karo!</h3>

        <strong>Step 1: AI Features Unlock Kijiye (2 minutes) 🔑</strong>
        <ol>
        <li><strong>⚙️ Setup & Configuration</strong> section mein jaiye</li>
        <li>Free Groq API key lijiye <a href="https://console.groq.com" target="_blank">console.groq.com</a> se</li>
        <li>Key enter kijiye aur "Save & Activate" click kijiye</li>
        <li>✅ Yeh AI optimization, analysis, aur assistance features unlock kar dega</li>
        </ol>

        <strong>Step 2: Market Conditions Dekhiye 📊</strong>
        <ol>
        <li><strong>📊 Dashboard</strong> pe jaiye current market trends dekhne ke liye</li>
        <li>Market sentiment aur volatility conditions review kijiye</li>
        <li>Pata kijiye ki markets trending hain ya range-bound</li>
        </ol>

        <strong>Step 3: Apna Pehla Strategy Banaiye 🔧</strong>
        <ol>
        <li>Main menu se <strong>🔧 Strategy Builder</strong> open kijiye</li>
        <li>Strategy type select kijiye (Momentum trending markets ke liye)</li>
        <li>Default parameters use kijiye start karne ke liye</li>
        <li>Historical backtest run kijiye initial performance dekhne ke liye</li>
        </ol>

        <strong>Step 4: AI Se Optimize Kijiye 🤖</strong>
        <ol>
        <li><strong>🤖 AI Optimize</strong> click kijiye strategy improve karne ke liye</li>
        <li>Optimization mode choose kijiye: Growth, Balanced, Quality, ya Conservative</li>
        <li>AI suggestions review kijiye aur original vs optimized parameters compare kijiye</li>
        <li>Optimizations apply kijiye aur improved performance metrics dekhiye</li>
        </ol>

        <strong>Step 5: Virtual Money Se Practice Kijiye 💰</strong>
        <ol>
        <li><strong>📈 Live Trading</strong> mein jaiye virtual trading start karne ke liye</li>
        <li>Small position sizes se start kijiye platform sikhne ke liye</li>
        <li>Apna virtual portfolio performance monitor kijiye</li>
        <li><strong>🤖 AI Assistant</strong> use kijiye trading guidance ke liye</li>
        </ol>

        <strong>🎯 Congratulations!</strong> Ab aap advanced features explore kar sakte hain aur apne trading strategies refine kar sakte hain.
        </div>
        """,
            unsafe_allow_html=True,
        )

elif section == "Latest Features" or section == "Latest Features (नए फीचर्स)":
    hr()
    if language == "English":
        st.markdown('<p class="big-font">🆕 Latest Platform Features</p>', unsafe_allow_html=True)
        
        st.markdown(
            """
        <div class="new-feature">
        <h3>🎯 Enhanced Strategy Builder</h3>
        <p>The Strategy Builder has been completely upgraded with advanced AI capabilities:</p>
        
        <h4>🚀 Multi-Objective AI Optimization</h4>
        <ul>
        <li><strong>Growth Mode</strong>: Maximizes returns with moderate risk tolerance</li>
        <li><strong>Balanced Mode</strong>: Optimal balance between returns and risk</li>
        <li><strong>Quality Mode</strong>: Focuses on consistency and win rate</li>
        <li><strong>Conservative Mode</strong>: Prioritizes capital preservation</li>
        </ul>
        
        <h4>📊 Improved User Experience</h4>
        <ul>
        <li><strong>Side-by-Side Comparison</strong>: See original vs optimized parameters clearly</li>
        <li><strong>Professional Metrics Display</strong>: Beautiful metric cards with visual indicators</li>
        <li><strong>Persistent AI Suggestions</strong>: Recommendations stay visible during exploration</li>
        <li><strong>No Page Refresh</strong>: Smooth optimization without interruption</li>
        <li><strong>Smart Parameter Updates</strong>: Real-time parameter synchronization</li>
        </ul>
        
        <h4>🔧 Technical Improvements</h4>
        <ul>
        <li><strong>Enhanced Error Handling</strong>: Better error messages and recovery</li>
        <li><strong>Faster Performance</strong>: Optimized calculations and caching</li>
        <li><strong>Mobile Responsive</strong>: Works perfectly on all devices</li>
        <li><strong>Stock Symbol Sync</strong>: Quick select automatically updates analysis</li>
        </ul>
        </div>
        """,
            unsafe_allow_html=True,
        )
        
    else:  # Hinglish version
        st.markdown('<p class="big-font">🆕 Latest Platform Features (नए फीचर्स)</p>', unsafe_allow_html=True)
        
        st.markdown(
            """
        <div class="new-feature">
        <h3>🎯 Enhanced Strategy Builder</h3>
        <p>Strategy Builder ko completely upgrade kiya gaya hai advanced AI capabilities ke saath:</p>
        
        <h4>🚀 Multi-Objective AI Optimization</h4>
        <ul>
        <li><strong>Growth Mode</strong>: Returns maximize karta hai moderate risk ke saath</li>
        <li><strong>Balanced Mode</strong>: Returns aur risk ke beech optimal balance</li>
        <li><strong>Quality Mode</strong>: Consistency aur win rate pe focus karta hai</li>
        <li><strong>Conservative Mode</strong>: Capital preservation ko priority deta hai</li>
        </ul>
        
        <h4>📊 Improved User Experience</h4>
        <ul>
        <li><strong>Side-by-Side Comparison</strong>: Original vs optimized parameters clearly dekh sakte ho</li>
        <li><strong>Professional Metrics Display</strong>: Beautiful metric cards visual indicators ke saath</li>
        <li><strong>Persistent AI Suggestions</strong>: Recommendations exploration ke dauraan visible rehti hain</li>
        <li><strong>No Page Refresh</strong>: Smooth optimization bina interruption ke</li>
        <li><strong>Smart Parameter Updates</strong>: Real-time parameter synchronization</li>
        </ul>
        
        <h4>🔧 Technical Improvements</h4>
        <ul>
        <li><strong>Enhanced Error Handling</strong>: Better error messages aur recovery</li>
        <li><strong>Faster Performance</strong>: Optimized calculations aur caching</li>
        <li><strong>Mobile Responsive</strong>: Saare devices pe perfectly kaam karta hai</li>
        <li><strong>Stock Symbol Sync</strong>: Quick select automatically analysis update kar deta hai</li>
        </ul>
        </div>
        """,
            unsafe_allow_html=True,
        )

elif section == "Strategy Builder (Enhanced)":
    hr()
    if language == "English":
        st.markdown('<p class="big-font">🔧 Strategy Builder — AI-Powered Strategy Development</p>', unsafe_allow_html=True)

        st.markdown(
            """
        <div class="info-box">
        <h4>🎯 Enhanced Strategy Builder</h4>
        <p>The Strategy Builder now features advanced AI optimization with multiple objective modes, improved parameter comparison, and a seamless user experience. Build, test, and optimize trading strategies with cutting-edge AI assistance.</p>
        </div>
        """,
            unsafe_allow_html=True,
        )

        with st.expander("🚀 New Multi-Objective AI Optimization", expanded=True):
            st.markdown(
                """
            **4 Intelligent Optimization Modes:**
            
            **1. Growth Mode 📈**
            - **Focus**: Maximum returns with calculated risk
            - **Best For**: Aggressive traders seeking high performance
            - **Optimizes**: Total return, Sharpe ratio, profit factor
            - **Risk Level**: Moderate to high
            - **Strategy Types**: Momentum, breakout strategies
            
            **2. Balanced Mode ⚖️**
            - **Focus**: Optimal risk-adjusted returns
            - **Best For**: Most traders seeking steady growth
            - **Optimizes**: Sortino ratio, Calmar ratio, consistency
            - **Risk Level**: Moderate
            - **Strategy Types**: Diversified approaches
            
            **3. Quality Mode 🎯**
            - **Focus**: Consistency and reliability
            - **Best For**: Traders prioritizing stable performance
            - **Optimizes**: Win rate, profit consistency, drawdown control
            - **Risk Level**: Low to moderate
            - **Strategy Types**: Mean reversion, range-bound
            
            **4. Conservative Mode 🛡️**
            - **Focus**: Capital preservation first
            - **Best For**: Risk-averse traders, retirement accounts
            - **Optimizes**: Maximum drawdown, downside protection
            - **Risk Level**: Low
            - **Strategy Types**: Defensive, low-volatility
            """
            )

        with st.expander("📊 Enhanced Parameter Comparison", expanded=True):
            st.markdown(
                """
            **New Comparison Features:**
            
            **Side-by-Side Display**
            - Original parameters on the left
            - Optimized parameters on the right
            - Clear highlighting of changes
            - Percentage improvement indicators
            
            **Performance Metrics Cards**
            - Professional-style metric displays
            - Color-coded performance indicators
            - Instant visual feedback
            - Key metrics at a glance
            """
            )
    
    else:  # Hinglish version
        st.markdown('<p class="big-font">🔧 Strategy Builder — AI-Powered Strategy Development</p>', unsafe_allow_html=True)

        st.markdown(
            """
        <div class="info-box">
        <h4>🎯 Enhanced Strategy Builder</h4>
        <p>Strategy Builder mein ab advanced AI optimization hai multiple objective modes ke saath, improved parameter comparison, aur seamless user experience. Trading strategies build, test, aur optimize karo cutting-edge AI assistance ke saath.</p>
        </div>
        """,
            unsafe_allow_html=True,
        )

        with st.expander("🚀 New Multi-Objective AI Optimization", expanded=True):
            st.markdown(
                """
            **4 Intelligent Optimization Modes:**
            
            **1. Growth Mode 📈**
            - **Focus**: Maximum returns calculated risk ke saath
            - **Best For**: Aggressive traders jo high performance chaahte hain
            - **Optimizes**: Total return, Sharpe ratio, profit factor
            - **Risk Level**: Moderate to high
            - **Strategy Types**: Momentum, breakout strategies
            
            **2. Balanced Mode ⚖️**
            - **Focus**: Optimal risk-adjusted returns
            - **Best For**: Zyada se zyada traders jo steady growth chaahte hain
            - **Optimizes**: Sortino ratio, Calmar ratio, consistency
            - **Risk Level**: Moderate
            - **Strategy Types**: Diversified approaches
            
            **3. Quality Mode 🎯**
            - **Focus**: Consistency aur reliability
            - **Best For**: Traders jo stable performance ko priority dete hain
            - **Optimizes**: Win rate, profit consistency, drawdown control
            - **Risk Level**: Low to moderate
            - **Strategy Types**: Mean reversion, range-bound
            
            **4. Conservative Mode 🛡️**
            - **Focus**: Capital preservation pehle
            - **Best For**: Risk-averse traders, retirement accounts
            - **Optimizes**: Maximum drawdown, downside protection
            - **Risk Level**: Low
            - **Strategy Types**: Defensive, low-volatility
            """
            )

        with st.expander("📊 Enhanced Parameter Comparison", expanded=True):
            st.markdown(
                """
            **New Comparison Features:**
            
            **Side-by-Side Display**
            - Original parameters left side mein
            - Optimized parameters right side mein
            - Changes ka clear highlighting
            - Percentage improvement indicators
            
            **Performance Metrics Cards**
            - Professional-style metric displays
            - Color-coded performance indicators
            - Instant visual feedback
            - Key metrics at a glance
            """
            )

elif section == "Setup & Configuration" or section == "Setup & Configuration (सेटअप)":
    hr()
    if language == "English":
        st.markdown('<p class="big-font">⚙️ Setup & Configuration</p>', unsafe_allow_html=True)
        
        # Check current API key status
        config = load_config()
        has_api_key = bool(config.get("groq_api_key"))

        if not has_api_key:
            st.error(
                """
            🚨 **IMPORTANT**: AI features are currently disabled because no API key is configured.
            Please complete the setup below to unlock all platform capabilities.
            """
            )
        else:
            st.success(
                "✅ **Configuration Complete**: All AI features are active and ready to use!"
            )

        with st.expander("🔑 API Key Configuration (Required for AI Features)", expanded=not has_api_key):
            st.markdown(
                """
            **🤖 Features Requiring API Key:**
            - **Multi-Objective AI Optimization**: 4 different optimization modes
            - **AI Assistant**: Get intelligent answers to trading questions  
            - **Performance Analysis**: Advanced AI-powered strategy insights
            - **Market Analysis**: AI-driven market condition assessment
            - **Risk Assessment**: Intelligent risk evaluation and suggestions
            
            **📋 How to Get Your Free API Key:**
            1. Visit [console.groq.com](https://console.groq.com) 
            2. Create a free account (takes 2 minutes)
            3. Navigate to "API Keys" section
            4. Generate a new API key
            5. Copy and paste it below
            """
            )

            # API key configuration interface
            groq_key = st.text_input(
                "Groq API Key",
                type="password",
                help="Enter your Groq API key to enable AI features",
                placeholder="gsk_... (paste your API key here)",
            )

            if st.button("💾 Save & Activate", type="primary"):
                if groq_key and len(groq_key) > 10:
                    from src.utils.config import save_api_key
                    success, message = save_api_key(groq_key)
                    
                    if success:
                        st.success(f"✅ {message}")
                        st.balloons()
                        st.rerun()
                    else:
                        st.error(f"❌ {message}")
                else:
                    st.warning("⚠️ Please enter a valid API key")
                    
    else:  # Hinglish version
        st.markdown('<p class="big-font">⚙️ Setup & Configuration (सेटअप)</p>', unsafe_allow_html=True)
        
        config = load_config()
        has_api_key = bool(config.get("groq_api_key"))

        if not has_api_key:
            st.error(
                """
            🚨 **IMPORTANT**: AI features currently disabled hain kyunki API key configure nahi hai.
            Please niche setup complete kijiye saare platform capabilities unlock karne ke liye.
            """
            )
        else:
            st.success(
                "✅ **Configuration Complete**: Saare AI features active aur ready hain!"
            )

        with st.expander("🔑 API Key Configuration (AI Features ke liye Required)", expanded=not has_api_key):
            st.markdown(
                """
            **🤖 API Key Chahiye In Features Ke Liye:**
            - **Multi-Objective AI Optimization**: 4 different optimization modes
            - **AI Assistant**: Trading questions ke intelligent answers  
            - **Performance Analysis**: Advanced AI-powered strategy insights
            - **Market Analysis**: AI-driven market condition assessment
            - **Risk Assessment**: Intelligent risk evaluation aur suggestions
            
            **📋 Free API Key Kaise Lein:**
            1. [console.groq.com](https://console.groq.com) pe jaiye
            2. Free account banaiye (2 minutes lagenge)
            3. "API Keys" section mein jaiye
            4. Naya API key generate kijiye
            5. Copy kar ke niche paste kijiye
            """
            )

            groq_key = st.text_input(
                "Groq API Key",
                type="password",
                help="AI features enable karne ke liye apna Groq API key enter kijiye",
                placeholder="gsk_... (apna API key yahan paste kijiye)",
            )

            if st.button("💾 Save & Activate", type="primary"):
                if groq_key and len(groq_key) > 10:
                    from src.utils.config import save_api_key
                    success, message = save_api_key(groq_key)
                    
                    if success:
                        st.success(f"✅ {message}")
                        st.balloons()
                        st.rerun()
                    else:
                        st.error(f"❌ {message}")
                else:
                    st.warning("⚠️ Please valid API key enter kijiye")

# Simplified sections for remaining features
elif section in ["Trading Basics", "Trading Basics (ट्रेडिंग बेसिक्स)"]:
    hr()
    if language == "English":
        st.markdown('<p class="big-font">📖 Trading Basics for Beginners</p>', unsafe_allow_html=True)
        st.markdown("""
        **Core Trading Concepts:**
        - **Market Data**: OHLCV (Open, High, Low, Close, Volume)
        - **Technical Indicators**: RSI, Moving Averages, Bollinger Bands
        - **Strategy Types**: Momentum, Mean Reversion, Breakout
        - **Risk Management**: Position sizing, stop losses, portfolio limits
        """)
    else:
        st.markdown('<p class="big-font">📖 Trading Basics for Beginners (ट्रेडिंग बेसिक्स)</p>', unsafe_allow_html=True)
        st.markdown("""
        **Core Trading Concepts:**
        - **Market Data**: OHLCV (Open, High, Low, Close, Volume) - ye basic price data hai
        - **Technical Indicators**: RSI, Moving Averages, Bollinger Bands - ye trend aur momentum dikhate hain
        - **Strategy Types**: Momentum (तेज़ी), Mean Reversion (वापसी), Breakout (तोड़ना)
        - **Risk Management**: Position sizing, stop losses, portfolio limits - risk control karne ke liye
        
        **महत्वपूर्ण सलाह:**
        - पहले paper trading (virtual money) se practice kijiye
        - छोटी amounts se start kijiye
        - हमेशा stop loss lagाiye
        - अपनी knowledge बढ़ाते रहिये
        """)

elif section in ["Platform Overview", "Platform Overview (प्लेटफॉर्म की जानकारी)"]:
    hr()
    if language == "English":
        st.markdown('<p class="big-font">🔍 Platform Overview</p>', unsafe_allow_html=True)
        st.markdown("""
        **Platform Components:**
        1. **📊 Dashboard** — Real-time market overview
        2. **🔧 Strategy Builder** — Create & AI-optimize strategies  
        3. **📈 Live Trading** — Paper trading with risk controls
        4. **📰 News Analysis** — Sentiment & themes
        5. **🔄 Backtesting** — Historical validation
        6. **🤖 AI Assistant** — Explanations & guidance
        """)
    else:
        st.markdown('<p class="big-font">🔍 Platform Overview (प्लेटफॉर्म की जानकारी)</p>', unsafe_allow_html=True)
        st.markdown("""
        **Platform Components (मुख्य भाग):**
        1. **📊 Dashboard** — Real-time market overview (बाज़ार की live जानकारी)
        2. **🔧 Strategy Builder** — Strategies बनाइए & AI se optimize कीजिए  
        3. **📈 Live Trading** — Virtual money से trading practice कीजिए
        4. **📰 News Analysis** — News sentiment aur themes analyze kijiye
        5. **🔄 Backtesting** — Historical data पर strategies test कीजिए
        6. **🤖 AI Assistant** — AI से trading guidance लीजिए
        
        **प्लेटफॉर्म के फायदे:**
        - ✅ Free मे use कर सकते हैं
        - ✅ Real Indian market data मिलता है
        - ✅ AI-powered features हैं
        - ✅ Educational purpose के लिए बहुत अच्छा है
        """)

elif section in ["Key Concepts", "Key Concepts (मुख्य अवधारणाएं)"]:
    hr()
    if language == "English":
        st.markdown('<p class="big-font">🔑 Key Trading Concepts</p>', unsafe_allow_html=True)
        st.markdown("""
        **Essential Concepts:**
        - **Volatility**: Price movement intensity
        - **Liquidity**: Ease of buying/selling
        - **Momentum**: Price trend strength
        - **Support/Resistance**: Key price levels
        - **Risk-Reward Ratio**: Profit vs loss potential
        """)
    else:
        st.markdown('<p class="big-font">🔑 Key Trading Concepts (मुख्य अवधारणाएं)</p>', unsafe_allow_html=True)
        st.markdown("""
        **जरूरी Concepts:**
        
        **📊 Volatility (अस्थिरता):**
        - Price movement की तीव्रता
        - High volatility = ज्यादा price swings
        - Low volatility = कम price movement
        
        **💧 Liquidity (तरलता):**
        - Stocks को buy/sell करने की आसानी
        - High liquidity = jaldi buy/sell हो जाता है
        - Low liquidity = time लगता है
        
        **⚡ Momentum (गति):**
        - Price trend की strength
        - Positive momentum = upward trend
        - Negative momentum = downward trend
        
        **🎯 Support/Resistance (सहारा/प्रतिरोध):**
        - Support = price नीचे नहीं जाता यहाँ से
        - Resistance = price ऊपर नहीं जाता यहाँ से
        
        **⚖️ Risk-Reward Ratio:**
        - Profit vs loss का अनुपात
        - 1:2 ratio = 1 रुपया risk, 2 रुपया profit potential
        """)

elif section in ["Dashboard", "Dashboard (डैशबोर्ड)"]:
    hr()
    if language == "English":
        st.markdown('<p class="big-font">📊 Dashboard Features</p>', unsafe_allow_html=True)
        st.markdown("""
        **Dashboard Overview:**
        - Market sentiment indicators
        - Top gainers and losers
        - Sector performance
        - News sentiment analysis
        """)
    else:
        st.markdown('<p class="big-font">📊 Dashboard Features (डैशबोर्ड की विशेषताएं)</p>', unsafe_allow_html=True)
        st.markdown("""
        **Dashboard Overview:**
        
        **📈 Market Sentiment Indicators:**
        - बाज़ार का overall mood देख सकते हैं
        - Bullish (तेज़ी) या Bearish (मंदी) का पता चलता है
        - Real-time data मिलता है
        
        **🔝 Top Gainers और Losers:**
        - सबसे ज्यादा बढ़ने वाले stocks
        - सबसे ज्यादा गिरने वाले stocks
        - Percentage change के साथ
        
        **🏭 Sector Performance:**
        - Different sectors का performance
        - IT, Banking, Pharma etc का analysis
        - Best performing sectors identify कर सकते हैं
        
        **📰 News Sentiment Analysis:**
        - Latest news का sentiment analysis
        - Positive या negative news का impact
        - Market moving news की जानकारी
        
        **कैसे Use करें:**
        1. Dashboard पर जाइए
        2. Market sentiment चेक कीजिए  
        3. Top movers देखिए
        4. Sector trends analyze कीजिए
        5. News impact समझिए
        """)

elif section in ["Live Trading", "Live Trading (लाइव ट्रेडिंग)"]:
    hr()
    if language == "English":
        st.markdown('<p class="big-font">📈 Live Trading Features</p>', unsafe_allow_html=True)
        st.markdown("""
        **Live Trading:**
        - Virtual portfolio management
        - Paper trading simulation
        - Real-time price data
        - Risk management controls
        """)
    else:
        st.markdown('<p class="big-font">📈 Live Trading Features (लाइव ट्रेडिंग)</p>', unsafe_allow_html=True)
        st.markdown("""
        **Live Trading की विशेषताएं:**
        
        **💰 Virtual Portfolio Management:**
        - Virtual money (paper money) से trading
        - Real पैसे का कोई risk नहीं
        - Portfolio tracking और monitoring
        - P&L (profit & loss) का real-time calculation
        
        **📊 Paper Trading Simulation:**
        - Real market conditions में practice
        - Live price data का use
        - Order placement की practice
        - Strategy testing बिना risk के
        
        **⚡ Real-time Price Data:**
        - Live Indian stock market data
        - NSE/BSE से real prices
        - Volume और other indicators
        - Minute-by-minute updates
        
        **🛡️ Risk Management Controls:**
        - Position size limits
        - Stop loss automation
        - Portfolio diversification rules
        - Maximum loss limits
        
        **शुरुआत कैसे करें:**
        1. Live Trading page पर जाइए
        2. Virtual account balance चेक कीजिए
        3. Stock symbol select कीजिए
        4. Buy/Sell orders place कीजिए
        5. Portfolio performance monitor कीजिए
        
        **⚠️ Important Tips:**
        - छोटी quantities से start कीजिए
        - हमेशा stop loss lagाइए
        - Diversification maintain कीजिए
        - Performance regularly review कीजिए
        """)

elif section in ["News Analysis", "News Analysis (न्यूज़ एनालिसिस)"]:
    hr()
    if language == "English":
        st.markdown('<p class="big-font">📰 News Analysis</p>', unsafe_allow_html=True)
        st.markdown("""
        **News Features:**
        - Sentiment analysis of market news
        - RSS feed integration
        - Market-moving events tracking
        - Impact assessment
        """)
    else:
        st.markdown('<p class="big-font">📰 News Analysis (समाचार विश्लेषण)</p>', unsafe_allow_html=True)
        st.markdown("""
        **News Analysis की विशेषताएं:**
        
        **💭 Sentiment Analysis:**
        - News articles का automatic sentiment detection
        - Positive, Negative, या Neutral classification
        - AI-powered analysis
        - Market impact prediction
        
        **📡 RSS Feed Integration:**
        - Multiple news sources से data
        - Real-time news updates
        - Financial news focus
        - Relevant articles filtering
        
        **📈 Market-Moving Events:**
        - Important announcements tracking
        - Earnings results analysis
        - Policy changes impact
        - Global events monitoring
        
        **📊 Impact Assessment:**
        - News का stock prices पर potential impact
        - Sector-wise impact analysis
        - Short-term vs long-term effects
        - Trading opportunities identification
        
        **कैसे Use करें:**
        1. News Analysis page पर जाइए
        2. Latest news देखिए
        3. Sentiment scores check कीजिए
        4. Impact assessment पढ़िए
        5. Trading decisions में incorporate कीजिए
        
        **⚠️ याद रखें:**
        - News पर immediate reaction न करें
        - Multiple sources से confirm कीजिए
        - Long-term view भी रखिए
        - Emotional decisions avoid कीजिए
        """)

elif section in ["Backtesting", "Backtesting (बैकटेस्टिंग)"]:
    hr()
    if language == "English":
        st.markdown('<p class="big-font">🔄 Backtesting</p>', unsafe_allow_html=True)
        st.markdown("""
        **Backtesting Features:**
        - Historical strategy validation
        - Performance metrics calculation
        - Risk analysis
        - Strategy comparison
        """)
    else:
        st.markdown('<p class="big-font">🔄 Backtesting (बैकटेस्टिंग)</p>', unsafe_allow_html=True)
        st.markdown("""
        **Backtesting की विशेषताएं:**
        
        **📚 Historical Strategy Validation:**
        - Past data पर strategy test कीजिए
        - Strategy की performance देखिए
        - Real market conditions simulation
        - Multiple time periods पर testing
        
        **📊 Performance Metrics:**
        - Total Return calculation
        - Sharpe Ratio (risk-adjusted return)
        - Maximum Drawdown (biggest loss)
        - Win Rate (successful trades percentage)
        - Profit Factor (profit/loss ratio)
        
        **⚠️ Risk Analysis:**
        - Volatility measurement
        - Downside risk assessment
        - Correlation analysis
        - Value at Risk (VaR) calculation
        
        **🔄 Strategy Comparison:**
        - Multiple strategies compare कीजिए
        - Best performing strategy identify कीजिए
        - Parameter sensitivity analysis
        - Optimization suggestions
        
        **Backtesting Process:**
        1. Strategy Builder में strategy बनाइए
        2. Historical period select कीजिए
        3. Backtest run कीजिए
        4. Results analyze कीजिए
        5. Parameters optimize कीजिए
        
        **Important Points:**
        - Past performance ≠ Future results
        - Realistic transaction costs include कीजिए
        - Out-of-sample testing भी कीजिए
        - Overfitting से बचिए
        """)

elif section in ["AI Assistant", "AI Assistant (AI असिस्टेंट)"]:
    hr()
    if language == "English":
        st.markdown('<p class="big-font">🤖 AI Assistant</p>', unsafe_allow_html=True)
        st.markdown("""
        **AI Assistant Features:**
        - Trading strategy explanations
        - Market analysis insights
        - Risk assessment guidance
        - Educational content
        """)
    else:
        st.markdown('<p class="big-font">🤖 AI Assistant (AI सहायक)</p>', unsafe_allow_html=True)
        st.markdown("""
        **AI Assistant की विशेषताएं:**
        
        **📖 Trading Strategy Explanations:**
        - Strategy logic को simple language में समझाता है
        - Parameters का meaning बताता है
        - Best practices suggest करता है
        - Real-time guidance देता है
        
        **📊 Market Analysis Insights:**
        - Current market conditions analyze करता है
        - Trends और patterns identify करता है
        - Sector rotation suggestions देता है
        - Technical indicators explain करता है
        
        **⚠️ Risk Assessment Guidance:**
        - Portfolio risk evaluate करता है
        - Position sizing recommendations देता है
        - Stop loss levels suggest करता है
        - Diversification advice देता है
        
        **🎓 Educational Content:**
        - Trading concepts सिखाता है
        - Market terminology explain करता है
        - Real examples देता है
        - Step-by-step guidance प्रदान करता है
        
        **कैसे Use करें:**
        1. AI Assistant page पर जाइए
        2. अपना question type कीजिए
        3. Context provide कीजिए
        4. AI response पढ़िए
        5. Follow-up questions पूछिए
        
        **Best Practices:**
        - Specific questions पूछिए
        - Context clearly दीजिए
        - Multiple perspectives लीजिए
        - Critical thinking maintain कीजिए
        
        **⚠️ Limitations:**
        - AI advice 100% accurate नहीं हो सकती
        - अपनी research भी कीजिए
        - Multiple sources से verify कीजिए
        - Final decisions खुद लीजिए
        """)

elif section in ["Best Practices", "Best Practices (बेस्ट प्रैक्टिसेज)"]:
    hr()
    if language == "English":
        st.markdown('<p class="big-font">✅ Best Practices</p>', unsafe_allow_html=True)
        st.markdown("""
        **Trading Best Practices:**
        - Start with paper trading
        - Use proper risk management
        - Maintain trading journal
        - Continuous learning
        """)
    else:
        st.markdown('<p class="big-font">✅ Best Practices (बेहतरीन प्रथाएं)</p>', unsafe_allow_html=True)
        st.markdown("""
        **Trading Best Practices:**
        
        **🎯 शुरुआत में:**
        - Paper trading से start कीजिए
        - छोटी amounts से practice कीजिए
        - Platform की सभी features सीखिए
        - Realistic expectations रखिए
        
        **💰 Risk Management:**
        - कभी भी सारा पैसा एक stock में न लगाइए
        - हमेशा stop loss use कीजिए
        - Position size carefully choose कीजिए
        - Portfolio diversification maintain कीजिए
        
        **📝 Record Keeping:**
        - Trading journal maintain कीजिए
        - Every trade का reason record कीजिए
        - Performance regularly review कीजिए
        - Mistakes से सीखिए
        
        **📚 Continuous Learning:**
        - Market knowledge बढ़ाते रहिए
        - New strategies explore कीजिए
        - Financial news पढ़िए
        - Educational content consume कीजिए
        
        **🧠 Psychology Management:**
        - Emotional decisions avoid कीजिए
        - FOMO (Fear of Missing Out) से बचिए
        - Patience develop कीजिए
        - Discipline maintain कीजिए
        
        **⚡ Technical Tips:**
        - AI optimization regularly use कीजिए
        - Backtesting thoroughly कीजिए
        - Multiple timeframes analyze कीजिए
        - Market conditions के अनुसार strategies adjust कीजिए
        """)

elif section in ["Risk Management", "Risk Management (रिस्क मैनेजमेंट)"]:
    hr()
    if language == "English":
        st.markdown('<p class="big-font">🛡️ Risk Management</p>', unsafe_allow_html=True)
        st.markdown("""
        **Risk Management Principles:**
        - Position sizing strategies
        - Stop loss implementation
        - Portfolio diversification
        - Risk-reward ratios
        """)
    else:
        st.markdown('<p class="big-font">🛡️ Risk Management (जोखिम प्रबंधन)</p>', unsafe_allow_html=True)
        st.markdown("""
        **Risk Management के मूल सिद्धांत:**
        
        **📏 Position Sizing:**
        - कभी भी portfolio का 5% से ज्यादा एक stock में न लगाइए
        - High-risk stocks में कम amount invest कीजिए
        - Volatility के अनुसार position size adjust कीजिए
        - Total exposure monitor कीजिए
        
        **🛑 Stop Loss Implementation:**
        - हर trade में stop loss जरूर लगाइए
        - Technical levels पर stop loss रखिए
        - Trailing stop loss use कीजिए
        - Emotional attachment avoid कीजिए
        
        **🎯 Portfolio Diversification:**
        - Different sectors में invest कीजिए
        - Multiple stocks रखिए portfolio में
        - Correlation कम रखिए
        - Asset allocation plan बनाइए
        
        **⚖️ Risk-Reward Ratios:**
        - Minimum 1:2 risk-reward ratio maintain कीजिए
        - High probability trades choose कीजिए
        - Expected value positive रखिए
        - Win rate और average win/loss balance कीजिए
        
        **📊 Risk Metrics to Monitor:**
        - **Maximum Drawdown**: Portfolio में maximum गिरावट
        - **Volatility**: Price movements की तीव्रता
        - **Sharpe Ratio**: Risk-adjusted returns
        - **VaR (Value at Risk)**: Potential maximum loss
        
        **🚨 Warning Signs:**
        - Consecutive losses हो रहे हैं
        - Emotional trading कर रहे हैं
        - Risk limits exceed हो रहे हैं
        - Strategy performance deteriorate हो रहा है
        
        **Emergency Protocols:**
        - Trading stop कर दीजिए अगर limits breach हों
        - Portfolio review कीजिए
        - Strategy re-evaluate कीजिए
        - Professional help लीजिए if needed
        """)

elif section in ["Common Pitfalls", "Common Pitfalls (सामान्य गलतियां)"]:
    hr()
    if language == "English":
        st.markdown('<p class="big-font">⚠️ Common Pitfalls</p>', unsafe_allow_html=True)
        st.markdown("""
        **Common Trading Mistakes:**
        - Overtrading and overconfidence
        - Ignoring risk management
        - Emotional decision making
        - Lack of strategy discipline
        """)
    else:
        st.markdown('<p class="big-font">⚠️ Common Pitfalls (आम गलतियां)</p>', unsafe_allow_html=True)
        st.markdown("""
        **आम Trading Mistakes:**
        
        **📈 Overtrading:**
        - बहुत ज्यादा trades कर रहे हैं
        - Market में हमेशा opportunity ढूंढ रहे हैं
        - Transaction costs बढ़ जाते हैं
        - **Solution**: Patience develop कीजिए, quality over quantity
        
        **😤 Overconfidence:**
        - कुछ profitable trades के बाद careless हो जाना
        - Risk management ignore करना
        - Position sizes बढ़ाना without justification
        - **Solution**: Humble रहिए, rules follow कीजिए
        
        **🚫 Risk Management को Ignore करना:**
        - Stop loss नहीं लगाना
        - Portfolio में concentration
        - Position sizing ignore करना
        - **Solution**: Risk rules strictly follow कीजिए
        
        **💭 Emotional Decision Making:**
        - Fear और Greed में decisions लेना
        - FOMO (Fear of Missing Out) में trade करना
        - Revenge trading करना losses के बाद
        - **Solution**: Pre-defined plan follow कीजिए
        
        **📋 Strategy Discipline की कमी:**
        - Strategy frequently change करना
        - Rules bend करना market conditions के लिए
        - Backtesting ignore करना
        - **Solution**: Strategy में consistent रहिए
        
        **📊 Analysis Paralysis:**
        - बहुत ज्यादा analysis करना
        - Decision लेने में delay
        - Perfect setup का wait करना
        - **Solution**: Good enough setup पर action लीजिए
        
        **💸 Money Management Mistakes:**
        - सारा capital एक trade में लगाना
        - Leverage का गलत use
        - Emergency fund नहीं रखना
        - **Solution**: Conservative approach अपनाइए
        
        **🔄 Recovery Strategies:**
        - Mistake identify कीजिए
        - Patterns देखिए
        - Rules revise कीजिए
        - Gradual improvement कीजिए
        """)

elif section in ["Troubleshooting", "Troubleshooting (समस्या निवारण)"]:
    hr()
    if language == "English":
        st.markdown('<p class="big-font">🔧 Troubleshooting</p>', unsafe_allow_html=True)
        st.markdown("""
        **Common Issues:**
        - Data loading problems
        - Strategy execution errors
        - Performance issues
        - API connectivity problems
        """)
    else:
        st.markdown('<p class="big-font">🔧 Troubleshooting (समस्या निवारण)</p>', unsafe_allow_html=True)
        st.markdown("""
        **आम समस्याएं और समाधान:**
        
        **📊 Data Loading Problems:**
        - **समस्या**: Stock data load नहीं हो रहा
        - **समाधान**: 
          * Internet connection check कीजिए
          * Symbol name correctly type कीजिए
          * Different date range try कीजिए
          * Page refresh कीजिए
        
        **🔧 Strategy Execution Errors:**
        - **समस्या**: Strategy run नहीं हो रही
        - **समाधान**:
          * Parameters valid range में रखिए
          * API key properly configured होनी चाहिए
          * Browser cache clear कीजिए
          * Error message carefully पढ़िए
        
        **⚡ Performance Issues:**
        - **समस्या**: Platform slow है या hang हो रहा है
        - **समाधान**:
          * Smaller date ranges use कीजिए
          * Browser tabs close कीजिए
          * RAM usage check कीजिए
          * Different browser try कीजिए
        
        **🔐 API Connectivity Problems:**
        - **समस्या**: AI features काम नहीं कर रहे
        - **समाधान**:
          * API key validate कीजिए
          * Internet connection stable होना चाहिए
          * Groq service status check कीजिए
          * New API key generate कीजिए
        
        **📱 Mobile/Browser Issues:**
        - **समस्या**: Mobile पर proper display नहीं हो रहा
        - **समाधान**:
          * Desktop/laptop use कीजिए complex analysis के लिए
          * Browser update कीजिए
          * JavaScript enable कीजिए
          * Incognito mode try कीजिए
        
        **💾 Data Saving Issues:**
        - **समस्या**: Settings save नहीं हो रहीं
        - **समाधान**:
          * Browser cookies enable कीजिए
          * Private/incognito mode से बाहर आइए
          * Local storage clear कीजिए
          * Different browser try कीजिए
        
        **🆘 Emergency Contacts:**
        - GitHub repository पर issue create कीजिए
        - Documentation thoroughly पढ़िए
        - Community forums check कीजिए
        - Platform logs screenshot लीजिए
        """)

elif section in ["Privacy & Safety", "Privacy & Safety (प्राइवेसी और सुरक्षा)"]:
    hr()
    if language == "English":
        st.markdown('<p class="big-font">🔒 Privacy & Safety</p>', unsafe_allow_html=True)
        st.markdown("""
        **Privacy & Safety Guidelines:**
        - Data handling practices
        - API key security
        - Personal information protection
        - Safe trading practices
        """)
    else:
        st.markdown('<p class="big-font">🔒 Privacy & Safety (गोपनीयता और सुरक्षा)</p>', unsafe_allow_html=True)
        st.markdown("""
        **Privacy & Safety Guidelines:**
        
        **🔐 Data Security:**
        - आपका personal trading data locally store होता है
        - कोई real money या banking details platform पर नहीं जाते
        - Virtual trading only - कोई real financial risk नहीं
        - Data encryption का use होता है
        
        **🔑 API Key Security:**
        - API key को share न करें किसी के साथ
        - Regular intervals पर API key rotate कीजिए
        - Suspicious activity notice करें तो immediate key change कीजिए
        - Official sources से ही API key generate कीजिए
        
        **👤 Personal Information Protection:**
        - कोई personal financial details platform पर enter न करें
        - Real bank account numbers, passwords share न करें
        - Educational purpose के लिए ही platform use कीजिए
        - Phishing attempts से सावधान रहिए
        
        **💰 Safe Trading Practices:**
        - हमेशा paper trading से start कीजिए
        - कभी भी emergency fund trading में न लगाइए
        - Real trading के लिए regulated brokers ही use कीजिए
        - Investment advice के लिए qualified professionals से मिलिए
        
        **🚨 Red Flags:**
        - कोई guaranteed returns का promise करे
        - Immediate large investments की demand करे
        - Personal banking details मांगे
        - Unrealistic profit claims करे
        
        **📝 Legal Disclaimer:**
        - यह platform educational purpose के लिए है
        - कोई investment advice नहीं दी जाती
        - Past performance future results guarantee नहीं करती
        - अपनी research करें और professional advice लें
        
        **✅ Safe Usage Tips:**
        - Regular password changes कीजिए
        - Secure internet connection use कीजिए
        - Public computers पर sensitive operations avoid कीजिए
        - Logout properly कीजिए sessions के बाद
        """)

# Rest of the sections can be handled with basic bilingual content

# Footer
hr()
if language == "English":
    st.markdown("""
    <div class="small-dim" style="text-align:center;">
    <p>📚 <strong>Complete AI Trading Platform Tutorial</strong></p>
    <p>🎯 Educational use only. Practice with virtual capital first.</p>
    <p>🚀 Happy trading & continuous learning!</p>
    </div>
    """, unsafe_allow_html=True)
else:
    st.markdown("""
    <div class="small-dim" style="text-align:center;">
    <p>📚 <strong>Complete AI Trading Platform Tutorial</strong></p>
    <p>🎯 Educational use only. Pehle virtual capital se practice kijiye.</p>
    <p>🚀 Happy trading & continuous learning!</p>
    </div>
    """, unsafe_allow_html=True)
