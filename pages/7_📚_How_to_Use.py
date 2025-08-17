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
.success-box {background:#d4edda; padding:15px; border-radius:5px; border-left:5px solid #28a745;}
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

        <strong>Step 1: AI Features Unlock Karo (2 minutes) 🔑</strong>
        <ol>
        <li><strong>⚙️ Setup & Configuration</strong> section mein jao</li>
        <li>Free Groq API key lo <a href="https://console.groq.com" target="_blank">console.groq.com</a> se</li>
        <li>Key enter karo aur "Save & Activate" click karo</li>
        <li>✅ Ye AI optimization, analysis, aur assistance features unlock kar dega</li>
        </ol>

        <strong>Step 2: Market Conditions Dekho 📊</strong>
        <ol>
        <li><strong>📊 Dashboard</strong> pe jao current market trends dekhne ke liye</li>
        <li>Market sentiment aur volatility conditions review karo</li>
        <li>Pata karo ki markets trending hai ya range-bound</li>
        </ol>

        <strong>Step 3: Apna Pehla Strategy Banao 🔧</strong>
        <ol>
        <li>Main menu se <strong>🔧 Strategy Builder</strong> open karo</li>
        <li>Strategy type select karo (Momentum trending markets ke liye)</li>
        <li>Default parameters use karo start karne ke liye</li>
        <li>Historical backtest run karo initial performance dekhne ke liye</li>
        </ol>

        <strong>Step 4: AI Se Optimize Karo 🤖</strong>
        <ol>
        <li><strong>🤖 AI Optimize</strong> click karo strategy improve karne ke liye</li>
        <li>Optimization mode choose karo: Growth, Balanced, Quality, ya Conservative</li>
        <li>AI suggestions review karo aur original vs optimized parameters compare karo</li>
        <li>Optimizations apply karo aur improved performance metrics dekho</li>
        </ol>

        <strong>Step 5: Virtual Money Se Practice Karo 💰</strong>
        <ol>
        <li><strong>📈 Live Trading</strong> mein jao virtual trading start karne ke liye</li>
        <li>Small position sizes se start karo platform sikhne ke liye</li>
        <li>Apna virtual portfolio performance monitor karo</li>
        <li><strong>🤖 AI Assistant</strong> use karo trading guidance ke liye</li>
        </ol>

        <strong>🎯 Congratulations!</strong> Ab aap advanced features explore kar sakte ho aur apne trading strategies refine kar sakte ho.
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
            🚨 **IMPORTANT**: AI features currently disabled hai kyunki API key configure nahi hai.
            Please niche setup complete karo saare platform capabilities unlock karne ke liye.
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
            
            **📋 Free API Key Kaise Le:**
            1. [console.groq.com](https://console.groq.com) pe jao
            2. Free account banao (2 minutes lagega)
            3. "API Keys" section mein jao
            4. Naya API key generate karo
            5. Copy kar ke niche paste karo
            """
            )

            groq_key = st.text_input(
                "Groq API Key",
                type="password",
                help="AI features enable karne ke liye apna Groq API key enter karo",
                placeholder="gsk_... (apna API key yahan paste karo)",
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
                    st.warning("⚠️ Please valid API key enter karo")

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
        - **Market Data**: OHLCV (Open, High, Low, Close, Volume)
        - **Technical Indicators**: RSI, Moving Averages, Bollinger Bands
        - **Strategy Types**: Momentum, Mean Reversion, Breakout
        - **Risk Management**: Position sizing, stop losses, portfolio limits
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
        **Platform Components:**
        1. **📊 Dashboard** — Real-time market overview
        2. **🔧 Strategy Builder** — Create & AI-optimize strategies  
        3. **📈 Live Trading** — Paper trading with risk controls
        4. **📰 News Analysis** — Sentiment & themes
        5. **🔄 Backtesting** — Historical validation
        6. **🤖 AI Assistant** — Explanations & guidance
        """)

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
    <p>🎯 Educational use only. Pehle virtual capital se practice karo.</p>
    <p>🚀 Happy trading & continuous learning!</p>
    </div>
    """, unsafe_allow_html=True)
