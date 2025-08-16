# how_to_use.py — single-page tutorial
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
</style>
""",
    unsafe_allow_html=True,
)

st.title("📚 Complete Trading Platform Tutorial")
st.caption("Master every feature of this comprehensive algorithmic trading framework")

# ---------- API Key Status Check ----------
from src.utils.config import load_config


def check_api_key_status():
    """Check if API key is configured"""
    config = load_config()
    return bool(config.get("groq_api_key"))


def show_api_key_prompt():
    """Show prominent API key setup prompt if not configured"""
    if not check_api_key_status():
        st.error(
            """
        🚨 **AI Features Disabled**: Groq API key not configured
        
        **Missing Features:**
        - 🤖 AI Strategy Optimization
        - 🧠 AI Assistant & Analysis  
        - 📊 Advanced Performance Insights
        - 💬 Intelligent Trading Guidance
        
        ➡️ **Set up your free API key in the "Setup & Configuration" section below**
        """
        )
        return False
    else:
        st.success("✅ **AI Features Active**: All platform features are available!")
        return True


# Show API key status at the top
api_configured = show_api_key_prompt()
st.markdown("---")

# ---------- sidebar: in-page navigation only ----------
section = st.sidebar.radio(
    "📋 Table of Contents",
    [
        "Quick Start",
        "Setup & Configuration",
        "Platform Overview",
        "Trading Basics",
        "Key Concepts",
        "Dashboard",
        "Strategy Builder",
        "Live Trading",
        "News Analysis",
        "Backtesting",
        "AI Assistant",
        "Best Practices",
        "Risk Management",
        "Common Pitfalls",
        "Troubleshooting",
        "Privacy & Safety",
    ],
    index=0,
)

# ---------- API Key Configuration in Sidebar ----------
st.sidebar.markdown("---")
st.sidebar.markdown("### 🔑 Quick API Setup")

# Load current config for sidebar
sidebar_config = load_config()
sidebar_has_key = bool(sidebar_config.get("groq_api_key"))

if sidebar_has_key:
    st.sidebar.success("✅ AI Features Active")
    if st.sidebar.button("🧪 Test Connection"):
        try:
            from src.analysis.ai_analyzer import GroqAnalyzer

            analyzer = GroqAnalyzer(sidebar_config["groq_api_key"])
            st.sidebar.success("✅ Connection successful!")
        except Exception as e:
            st.sidebar.error(f"❌ Connection failed: {str(e)}")
else:
    st.sidebar.warning("⚠️ AI Features Disabled")

    with st.sidebar.expander("⚡ Enable AI Features", expanded=True):
        st.markdown("**Quick Setup:**")

        # API key input in sidebar
        sidebar_api_key = st.text_input(
            "Groq API Key",
            type="password",
            placeholder="gsk_...",
            help="Get free key from console.groq.com",
            key="sidebar_api_key",
        )

        if st.button("💾 Save & Enable", key="sidebar_save"):
            if sidebar_api_key and len(sidebar_api_key) > 10:
                try:
                    # Update .env file
                    env_path = "/home/gourav/ai/.env"
                    with open(env_path, "r") as f:
                        lines = f.readlines()

                    with open(env_path, "w") as f:
                        for line in lines:
                            if line.startswith("GROQ_API_KEY="):
                                f.write(f"GROQ_API_KEY={sidebar_api_key}\n")
                            else:
                                f.write(line)

                    st.sidebar.success("✅ Saved! Refresh to activate.")
                    st.sidebar.balloons()
                except Exception as e:
                    st.sidebar.error(f"❌ Error: {str(e)}")
            else:
                st.sidebar.warning("⚠️ Enter a valid API key")

        st.markdown("[🔗 Get Free API Key](https://console.groq.com)")


# ---------- helpers ----------
def hr():
    st.markdown("---")


# ---------- sections ----------
if section == "Quick Start":
    hr()
    st.markdown('<p class="big-font">🚀 Quick Start Guide</p>', unsafe_allow_html=True)

    # Check API key status for Quick Start
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
<li>Select "Momentum" strategy type for trending markets</li>
<li>Use default parameters to start (you can optimize later)</li>
<li>Run a historical backtest to see initial performance</li>
</ol>

<strong>Step 4: Optimize with AI 🤖</strong>
<ol>
<li>Click <strong>🤖 AI Optimize</strong> to improve your strategy</li>
<li>Set reasonable improvement targets (start with 5-10%)</li>
<li>Review AI suggestions and apply the best optimizations</li>
<li>Compare before/after performance metrics</li>
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

    # Show current status
    if check_api_key_status():
        st.success(
            "✅ **You're All Set!** AI features are active. You can jump straight to Step 2."
        )
    else:
        st.info(
            "💡 **Next Step**: Complete the API key setup in Step 1 to unlock all features."
        )

    st.markdown(
        """
    <div class="info-box">
    <h4>💡 Pro Tips for Beginners</h4>
    <ul>
    <li><strong>Start Simple</strong>: Begin with basic momentum or mean reversion strategies</li>
    <li><strong>Learn the Metrics</strong>: Understand Sharpe ratio, maximum drawdown, and win rate</li>
    <li><strong>Use AI Guidance</strong>: The AI Assistant can explain complex concepts in simple terms</li>
    <li><strong>Practice First</strong>: Always use virtual money before considering real trading</li>
    <li><strong>Stay Curious</strong>: Explore different sections of this tutorial to deepen your knowledge</li>
    </ul>
    </div>
    """,
        unsafe_allow_html=True,
    )

    # Feature availability reminder
    st.markdown("---")
    st.markdown("**🎯 What You Can Do Right Now:**")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(
            """
        **✅ Available Without API Key:**
        - Browse market data and news
        - Create basic trading strategies
        - Run simple backtests
        - Practice virtual trading
        - Learn from tutorials
        """
        )

    with col2:
        if check_api_key_status():
            st.markdown(
                """
            **🚀 AI Features Active:**
            - AI strategy optimization
            - Intelligent trading assistant
            - Advanced performance analysis
            - Smart risk assessment
            - Personalized guidance
            """
            )
        else:
            st.markdown(
                """
            **🔒 Requires API Key Setup:**
            - AI strategy optimization
            - Intelligent trading assistant  
            - Advanced performance analysis
            - Smart risk assessment
            - Personalized guidance
            """
            )

elif section == "Setup & Configuration":
    hr()
    st.markdown(
        '<p class="big-font">⚙️ Setup & Configuration</p>', unsafe_allow_html=True
    )

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

    st.markdown(
        """
    <div class="info-box">
    <h3>🔧 Platform Configuration</h3>
    <p>Set up your AI-powered trading platform to access all features including strategy optimization and intelligent analysis.</p>
    </div>
    """,
        unsafe_allow_html=True,
    )

    with st.expander(
        "🔑 API Key Configuration (Required for AI Features)", expanded=not has_api_key
    ):
        st.markdown(
            f"""
        <div class="{'warning-box' if not has_api_key else 'success-box'}">
        <h4>{'🚨 Setup Required' if not has_api_key else '✅ Configured Successfully'}</h4>
        <p>The Groq API key powers all AI features in this platform. Without it, you'll have limited functionality.</p>
        </div>
        """,
            unsafe_allow_html=True,
        )

        st.markdown(
            """
        **🤖 Features Requiring API Key:**
        - **AI Strategy Optimization**: Automatically improve your trading strategies
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
        
        **💰 Cost**: The free tier provides generous usage limits perfect for individual traders
        """
        )

        # API key configuration interface
        st.markdown("---")
        st.markdown("**🔧 Configure Your API Key:**")

        # API key input with better UX
        current_key_display = "●●●●●●●●●●●●●●●●●●●●" if has_api_key else ""

        groq_key = st.text_input(
            "Groq API Key",
            value=current_key_display,
            type="password",
            help="Enter your Groq API key to enable AI features",
            placeholder="gsk_... (paste your API key here)",
        )

        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button(
                "💾 Save & Activate", type="primary", use_container_width=True
            ):
                if groq_key and not groq_key.startswith("●") and len(groq_key) > 10:
                    from src.utils.config import save_api_key

                    success, message = save_api_key(groq_key)

                    if success:
                        st.success(f"✅ {message}")
                        st.info("🔄 Reloading application...")
                        st.balloons()
                        # Clear cache and rerun to reload config
                        st.cache_data.clear()
                        st.cache_resource.clear()
                        st.rerun()
                    else:
                        st.error(f"❌ {message}")
                else:
                    st.warning(
                        "⚠️ Please enter a valid API key (should start with 'gsk_')"
                    )

        with col2:
            if st.button("🧪 Test Connection", use_container_width=True):
                if has_api_key:
                    try:
                        from src.analysis.ai_analyzer import GroqAnalyzer

                        analyzer = GroqAnalyzer(config["groq_api_key"])
                        st.success(
                            "✅ AI connection successful! All features are ready."
                        )
                    except Exception as e:
                        st.error(f"❌ Connection failed: {str(e)}")
                        st.warning("Please check your API key and internet connection.")
                else:
                    st.warning("⚠️ No API key configured yet. Please save a key first.")

        with col3:
            if st.button("🔗 Get Free API Key", use_container_width=True):
                st.markdown("**Quick Setup Guide:**")
                st.markdown(
                    """
                1. 🌐 Visit [console.groq.com](https://console.groq.com)
                2. 📧 Sign up with your email
                3. 🔑 Go to "API Keys" tab  
                4. ➕ Click "Create API Key"
                5. 📋 Copy the key and paste above
                """
                )

    with st.expander("🎛️ Trading Settings", expanded=False):
        st.markdown(
            """
        **📊 Customize Your Trading Environment:**
        """
        )

        col1, col2 = st.columns(2)
        with col1:
            virtual_money = st.number_input(
                "Virtual Portfolio Value ($)",
                min_value=10000.0,
                max_value=10000000.0,
                value=float(config.get("virtual_money", 100000)),
                step=10000.0,
                help="Amount of virtual money for paper trading practice",
            )

        with col2:
            commission = st.number_input(
                "Commission Rate (%)",
                min_value=0.0,
                max_value=1.0,
                value=float(config.get("commission", 0.1)),
                step=0.01,
                format="%.2f",
                help="Transaction cost per trade (typical range: 0.1-0.5%)",
            )

        if st.button("💾 Update Trading Settings", use_container_width=True):
            try:
                # Update .env file with trading settings
                env_path = "/home/gourav/ai/.env"
                with open(env_path, "r") as f:
                    lines = f.readlines()

                with open(env_path, "w") as f:
                    for line in lines:
                        if line.startswith("VIRTUAL_MONEY_AMOUNT="):
                            f.write(f"VIRTUAL_MONEY_AMOUNT={virtual_money}\n")
                        elif line.startswith("DEFAULT_COMMISSION="):
                            f.write(f"DEFAULT_COMMISSION={commission}\n")
                        else:
                            f.write(line)

                st.success("✅ Trading settings updated successfully!")
            except Exception as e:
                st.error(f"❌ Error updating settings: {str(e)}")

    with st.expander("ℹ️ Optional Enhancements", expanded=False):
        st.markdown(
            """
        **🔧 Additional API Keys (Optional):**
        
        These are completely optional and only provide enhanced data sources:
        
        **Alpha Vantage (Free)**
        - **Purpose**: Additional market data feeds
        - **Get Your Key**: Visit [alphavantage.co](https://www.alphavantage.co/support/#api-key)
        - **Benefit**: More comprehensive market analysis capabilities
        
        **News API (Free Tier Available)**
        - **Purpose**: Real-time news sentiment analysis
        - **Get Your Key**: Visit [newsapi.org](https://newsapi.org)
        - **Benefit**: Incorporate news sentiment into trading decisions
        
        **📝 Note**: The platform works completely with just the Groq API key. These additional services only enhance the experience.
        """
        )

    # Show feature comparison
    st.markdown("---")
    st.markdown("**🔍 Feature Availability Comparison:**")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(
            """
        **✅ Without API Key (Limited):**
        - 📊 Basic market data viewing
        - 📈 Simple strategy backtesting  
        - 📰 News feed browsing
        - 💰 Virtual trading interface
        - 📚 Educational content
        """
        )

    with col2:
        st.markdown(
            """
        **🚀 With API Key (Full Power):**
        - 🤖 AI strategy optimization
        - 🧠 AI assistant & analysis
        - 📊 Advanced performance insights  
        - 💬 Intelligent trading guidance
        - 🔍 AI-powered market analysis
        """
        )

    st.markdown(
        """
    <div class="warning-box">
    <h4>🔒 Security & Privacy</h4>
    <ul>
    <li><strong>Local Storage</strong>: Your API key is stored only on your device</li>
    <li><strong>No Sharing</strong>: We never transmit your key to third parties</li>
    <li><strong>You Control</strong>: You can update or remove your key anytime</li>
    <li><strong>Secure Transmission</strong>: All API calls use encrypted HTTPS</li>
    </ul>
    </div>
    """,
        unsafe_allow_html=True,
    )

elif section == "Platform Overview":
    hr()
    st.markdown('<p class="big-font">🔍 Platform Overview</p>', unsafe_allow_html=True)
    st.markdown(
        """
<div class="highlight">
<h4>🏗️ Architecture</h4>
<ol>
<li><strong>📊 Dashboard</strong> — real-time market overview</li>
<li><strong>🔧 Strategy Builder</strong> — create & AI-optimize strategies</li>
<li><strong>📈 Live Trading</strong> — paper trading with risk controls</li>
<li><strong>📰 News Analysis</strong> — sentiment & themes</li>
<li><strong>🔄 Backtesting</strong> — historical validation</li>
<li><strong>🤖 AI Assistant</strong> — explanations & guidance</li>
</ol>

<strong>🎯 Realism</strong>
<ul>
<li>Business day calendar</li>
<li>Close-to-close returns, next-bar execution (no look-ahead)</li>
<li>Costs modeled externally in backtests</li>
</ul>
</div>
""",
        unsafe_allow_html=True,
    )

elif section == "Trading Basics":
    hr()
    st.markdown(
        '<p class="big-font">📖 Trading Basics for Beginners</p>',
        unsafe_allow_html=True,
    )

    st.markdown(
        """
    <div class="info-box">
    <h4>🎯 What is Algorithmic Trading?</h4>
    <p>Algorithmic trading is the use of computer programs to execute trading decisions based on predefined rules, mathematical models, and statistical analysis. Unlike human traders who rely on emotions and intuition, algorithms make decisions based purely on data and logic.</p>
    </div>
    """,
        unsafe_allow_html=True,
    )

    st.subheader("🔍 Core Components of Algorithmic Trading")

    with st.expander("📊 Market Data & Analysis", expanded=True):
        st.markdown(
            """
        **Price Data (OHLCV)**
        - **Open**: First price when market opens
        - **High**: Highest price during the period  
        - **Low**: Lowest price during the period
        - **Close**: Last price when market closes
        - **Volume**: Number of shares traded
        
        **Why Each Matters:**
        - **Open**: Shows overnight sentiment and gaps
        - **High/Low**: Indicates intraday volatility and support/resistance
        - **Close**: Most important for trend analysis (closing price = market consensus)
        - **Volume**: Confirms price movements (high volume = strong conviction)
        
        **Time Frames:**
        - **1-minute**: High-frequency, noise-heavy
        - **1-hour**: Intraday patterns
        - **1-day**: Most common for swing trading
        - **Weekly/Monthly**: Long-term trends
        """
        )

    with st.expander("🧮 Technical Indicators Explained", expanded=True):
        st.markdown(
            """
        **Moving Averages (MA)**
        - **Simple Moving Average (SMA)**: Average price over N periods
        - **Exponential Moving Average (EMA)**: Gives more weight to recent prices
        - **Usage**: Trend direction, support/resistance, crossover signals
        - **Example**: If price > 50-day MA, uptrend likely
        
        **Relative Strength Index (RSI)**
        - **Range**: 0-100
        - **Overbought**: >70 (sell signal)
        - **Oversold**: <30 (buy signal)
        - **Purpose**: Identify potential reversal points
        - **Formula**: RSI = 100 - (100 / (1 + RS)), where RS = Avg Gain / Avg Loss
        
        **Bollinger Bands**
        - **Middle**: 20-period SMA
        - **Upper**: SMA + (2 × Standard Deviation)
        - **Lower**: SMA - (2 × Standard Deviation)
        - **Usage**: Volatility measure, mean reversion signals
        - **Squeeze**: Bands narrow → volatility breakout coming
        
        **Volume Indicators**
        - **Average Volume**: Confirms trend strength
        - **Volume Spike**: Unusual activity, potential breakout
        - **Volume Divergence**: Price up but volume down = weak trend
        """
        )

    with st.expander("📈 Strategy Types Deep Dive", expanded=True):
        st.markdown(
            """
        **1. Momentum Strategies**
        - **Concept**: "Trend is your friend" - assets moving up continue up
        - **Logic**: Markets trend due to herding behavior, information flow
        - **Signals**: Price breakouts, moving average crossovers
        - **Best Markets**: Trending markets with clear direction
        - **Risk**: Whipsaws in choppy markets
        - **Example**: Buy when 10-day MA crosses above 50-day MA
        
        **2. Mean Reversion Strategies**  
        - **Concept**: "What goes up must come down" - prices return to average
        - **Logic**: Markets overreact, creating temporary mispricings
        - **Signals**: RSI extremes, Bollinger Band touches
        - **Best Markets**: Range-bound, sideways markets
        - **Risk**: Catching a falling knife in strong trends
        - **Example**: Buy when RSI < 30 and price hits lower Bollinger Band
        
        **3. Breakout Strategies**
        - **Concept**: Trade when price breaks key levels with conviction
        - **Logic**: Breakouts often lead to sustained moves
        - **Signals**: Price above resistance + high volume
        - **Best Markets**: After consolidation periods
        - **Risk**: False breakouts (many fail)
        - **Example**: Buy when price breaks 20-day high with 2x average volume
        
        **4. Arbitrage Strategies**
        - **Concept**: Exploit price differences between related instruments
        - **Logic**: Same asset shouldn't trade at different prices
        - **Types**: Statistical arbitrage, pairs trading
        - **Best Markets**: Highly correlated instruments
        - **Risk**: Correlation breakdown
        - **Example**: Long underperformer, short overperformer in pair
        """
        )

    with st.expander("⚠️ Risk Management Fundamentals", expanded=True):
        st.markdown(
            """
        **Position Sizing**
        - **Fixed Dollar**: Same $ amount per trade
        - **Fixed Percentage**: Same % of portfolio per trade  
        - **Volatility-Based**: Larger positions in less volatile assets
        - **Kelly Criterion**: Mathematically optimal sizing based on edge
        - **Rule of Thumb**: Never risk more than 1-2% per trade
        
        **Stop Losses**
        - **Purpose**: Limit maximum loss per trade
        - **Types**: 
          - Fixed % (e.g., 5% below entry)
          - ATR-based (e.g., 2× Average True Range)
          - Technical (below support level)
          - Time-based (exit after N days)
        - **Trailing Stops**: Move stop up as price moves favorably
        
        **Take Profits**
        - **Fixed Ratio**: e.g., 2:1 reward-to-risk ratio
        - **Technical Levels**: Resistance, round numbers
        - **Partial Profits**: Take some profits, let rest run
        - **Momentum-Based**: Exit when momentum weakens
        
        **Portfolio Management**
        - **Diversification**: Don't put all eggs in one basket
        - **Correlation**: Avoid too many similar positions
        - **Concentration Limits**: Max % per position
        - **Drawdown Limits**: Stop trading if losses exceed threshold
        """
        )

    with st.expander("💡 Psychology & Discipline", expanded=True):
        st.markdown(
            """
        **Common Psychological Biases**
        - **Loss Aversion**: Feeling losses 2x more than gains
        - **Confirmation Bias**: Only seeing information that confirms beliefs
        - **Overconfidence**: Thinking you're better than you are
        - **Anchoring**: Stuck on first piece of information
        - **FOMO**: Fear of missing out leads to poor entries
        
        **Benefits of Algorithmic Trading**
        - **Removes Emotion**: Computer doesn't feel fear or greed
        - **Consistent Execution**: Same rules applied every time
        - **Speed**: Faster than human reaction time
        - **Backtesting**: Can test strategies on historical data
        - **Scalability**: Can monitor many instruments simultaneously
        
        **Developing Discipline**
        - **Write Trading Plan**: Document your strategy rules
        - **Paper Trade First**: Practice with virtual money
        - **Keep Trading Journal**: Record all trades and emotions
        - **Review Regularly**: Analyze what's working and what isn't
        - **Stay Small**: Start with small position sizes
        """
        )

    st.markdown(
        """
    <div class="warning-box">
    <h4>⚠️ Important Disclaimers</h4>
    <ul>
    <li><strong>Educational Purpose Only</strong>: This platform is for learning, not investment advice</li>
    <li><strong>Past Performance ≠ Future Results</strong>: Historical success doesn't guarantee future profits</li>
    <li><strong>Risk of Loss</strong>: All trading involves risk of substantial loss</li>
    <li><strong>Start Virtual</strong>: Always practice with virtual money before risking real capital</li>
    <li><strong>Understand Before Trading</strong>: Never trade strategies you don't fully understand</li>
    <li><strong>Market Conditions Change</strong>: Strategies that work in one environment may fail in another</li>
    </ul>
    </div>
    """,
        unsafe_allow_html=True,
    )

elif section == "Key Concepts":
    hr()
    st.markdown(
        '<p class="big-font">🔑 Key Trading & Performance Concepts</p>',
        unsafe_allow_html=True,
    )

    with st.expander("📊 Return Metrics - Deep Dive", expanded=True):
        st.markdown(
            """
        **Total Return**
        - **Definition**: Overall percentage gained or lost from start to finish
        - **Formula**: (Final Value - Initial Value) / Initial Value × 100
        - **Example**: $10,000 → $12,000 = (12,000 - 10,000) / 10,000 = 20%
        - **Use Case**: Quick overall performance snapshot
        - **Limitation**: Doesn't account for time or risk taken
        
        **Annualized Return**
        - **Definition**: Return scaled to a 1-year period for comparison
        - **Formula**: (1 + Total Return)^(1/Years) - 1
        - **Example**: 44% return over 2 years = (1.44)^(1/2) - 1 = 19.4% annualized
        - **Use Case**: Compare strategies with different time periods
        - **Standard**: Most common way to report performance
        
        **CAGR (Compound Annual Growth Rate)**
        - **Definition**: Rate at which investment would grow if compounded annually
        - **Formula**: (Ending Value / Beginning Value)^(1/n) - 1
        - **Example**: $10,000 → $15,000 over 3 years = (15,000/10,000)^(1/3) - 1 = 14.5%
        - **Use Case**: Long-term investment comparison
        - **Benefit**: Smooths out volatility, shows steady growth rate
        """
        )

        # Interactive calculator for returns
        st.markdown("**📱 Interactive Return Calculator**")
        col1, col2, col3 = st.columns(3)
        with col1:
            initial_val = st.number_input("Initial Value ($)", value=10000, min_value=1)
        with col2:
            final_val = st.number_input("Final Value ($)", value=12000, min_value=1)
        with col3:
            years = st.number_input(
                "Time Period (Years)", value=1.0, min_value=0.1, step=0.1
            )

        if initial_val > 0 and final_val > 0 and years > 0:
            total_return = (final_val - initial_val) / initial_val * 100
            annual_return = ((final_val / initial_val) ** (1 / years) - 1) * 100
            st.success(
                f"**Total Return**: {total_return:.2f}% | **Annualized**: {annual_return:.2f}%"
            )

    with st.expander("📈 Risk Metrics - Complete Guide", expanded=True):
        st.markdown(
            """
        **Volatility (Standard Deviation)**
        - **Definition**: Measure of how much returns vary from the average
        - **Formula**: √(Σ(Return - Avg Return)² / (n-1))
        - **Interpretation**: 
          - Low volatility (0-10%): Stable, boring investments (bonds)
          - Medium volatility (10-20%): Typical stock market
          - High volatility (>20%): Crypto, penny stocks, leveraged products
        - **Use Case**: Understanding investment "bumpiness"
        - **Time Period**: Usually annualized (multiply daily by √252)
        
        **Maximum Drawdown**
        - **Definition**: Worst peak-to-trough decline experienced
        - **Formula**: (Trough Value - Peak Value) / Peak Value × 100
        - **Example**: Portfolio goes $100k → $70k → $110k. Max DD = -30%
        - **Importance**: Shows worst-case scenario you must mentally prepare for
        - **Recovery Time**: How long to get back to peak (also important)
        - **Typical Ranges**: 
          - Conservative: 5-15%
          - Moderate: 15-30%
          - Aggressive: 30%+
        
        **Value at Risk (VaR)**
        - **Definition**: Maximum expected loss over specific time period at given confidence level
        - **Example**: "95% VaR of $1,000" = 95% chance you won't lose more than $1,000 tomorrow
        - **Methods**: 
          - Historical: Based on past data
          - Parametric: Assumes normal distribution
          - Monte Carlo: Computer simulation
        - **Limitation**: Doesn't predict extreme events (black swans)
        - **Regulatory Use**: Banks must maintain capital based on VaR
        
        **Beta**
        - **Definition**: How much an investment moves relative to market
        - **Values**: 
          - Beta = 1: Moves exactly with market
          - Beta > 1: More volatile than market
          - Beta < 1: Less volatile than market
          - Beta < 0: Moves opposite to market
        - **Example**: Beta = 1.5 means if market goes up 10%, stock typically goes up 15%
        - **Use Case**: Portfolio risk management, diversification
        """
        )

    with st.expander("⚖️ Risk-Adjusted Metrics - Professional Analysis", expanded=True):
        st.markdown(
            """
        **Sharpe Ratio**
        - **Definition**: Excess return per unit of risk
        - **Formula**: (Portfolio Return - Risk-Free Rate) / Portfolio Volatility
        - **Interpretation**:
          - <0: Negative excess return (bad)
          - 0-1: Acceptable performance
          - 1-2: Good performance
          - 2+: Excellent performance (very rare)
        - **Example**: 15% return, 3% risk-free rate, 12% volatility = (15%-3%)/12% = 1.0
        - **Use Case**: Compare strategies with different risk levels
        - **Limitation**: Assumes normal distribution of returns
        
        **Sortino Ratio**
        - **Definition**: Like Sharpe, but only considers downside volatility
        - **Formula**: (Portfolio Return - Target Return) / Downside Deviation
        - **Logic**: Upside volatility is good, only penalize downside risk
        - **Better For**: Asymmetric return distributions
        - **Interpretation**: Same scale as Sharpe, but usually higher values
        - **Advantage**: More intuitive (investors don't mind upside surprises)
        
        **Calmar Ratio**
        - **Definition**: Annualized return divided by maximum drawdown
        - **Formula**: Annualized Return / |Maximum Drawdown|
        - **Example**: 20% annual return, 10% max drawdown = 20/10 = 2.0
        - **Interpretation**:
          - <1: Poor (return doesn't justify the pain)
          - 1-3: Acceptable
          - 3+: Excellent
        - **Use Case**: Focus on worst-case scenarios
        - **Advantage**: Easy to understand (return per unit of worst loss)
        
        **Information Ratio**
        - **Definition**: Excess return relative to benchmark per unit of tracking error
        - **Formula**: (Portfolio Return - Benchmark Return) / Tracking Error
        - **Use Case**: Evaluate active management vs passive indexing
        - **Good Value**: Above 0.5-0.75 considered skilled management
        - **Limitation**: Requires appropriate benchmark selection
        
        **Treynor Ratio**
        - **Definition**: Excess return per unit of systematic risk (beta)
        - **Formula**: (Portfolio Return - Risk-Free Rate) / Beta
        - **Use Case**: Compare portfolios with different market exposures
        - **Difference from Sharpe**: Uses beta instead of total volatility
        - **Better When**: Analyzing well-diversified portfolios
        """
        )

        # Interactive risk-adjusted metrics calculator
        st.markdown("**📱 Interactive Risk-Adjusted Metrics Calculator**")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            port_return = st.number_input("Portfolio Return (%)", value=15.0, step=0.1)
        with col2:
            risk_free = st.number_input("Risk-Free Rate (%)", value=3.0, step=0.1)
        with col3:
            volatility = st.number_input(
                "Volatility (%)", value=12.0, min_value=0.1, step=0.1
            )
        with col4:
            max_dd = st.number_input(
                "Max Drawdown (%)", value=8.0, min_value=0.1, step=0.1
            )

        if volatility > 0 and max_dd > 0:
            sharpe = (port_return - risk_free) / volatility
            calmar = port_return / max_dd

            col1, col2 = st.columns(2)
            with col1:
                st.metric("Sharpe Ratio", f"{sharpe:.2f}")
                if sharpe < 0:
                    st.error("Negative excess return")
                elif sharpe < 1:
                    st.warning("Acceptable performance")
                elif sharpe < 2:
                    st.success("Good performance")
                else:
                    st.success("Excellent performance!")

            with col2:
                st.metric("Calmar Ratio", f"{calmar:.2f}")
                if calmar < 1:
                    st.error("Return doesn't justify risk")
                elif calmar < 3:
                    st.warning("Acceptable risk-adjusted return")
                else:
                    st.success("Excellent risk-adjusted return!")

    with st.expander("🔄 Market Regime Analysis", expanded=True):
        st.markdown(
            """
        **What are Market Regimes?**
        - **Definition**: Persistent market conditions with similar characteristics
        - **Importance**: Different strategies work in different regimes
        - **Examples**: Bull market, bear market, sideways market, high/low volatility
        
        **Regime Types:**
        
        **1. Trending Markets**
        - **Characteristics**: Clear directional movement, momentum persists
        - **Best Strategies**: Momentum, trend-following, breakout
        - **Indicators**: Moving averages sloping, ADX > 25
        - **Duration**: Can last months to years
        
        **2. Mean-Reverting Markets**
        - **Characteristics**: Prices oscillate around average, ranges persist
        - **Best Strategies**: Mean reversion, pairs trading, range-bound
        - **Indicators**: Flat moving averages, low ADX
        - **Duration**: Can last weeks to months
        
        **3. High Volatility Regimes**
        - **Characteristics**: Large price swings, uncertainty, fear
        - **Best Strategies**: Short volatility (carefully), defensive
        - **Indicators**: VIX > 20, large daily moves
        - **Triggers**: Economic uncertainty, geopolitical events
        
        **4. Low Volatility Regimes**
        - **Characteristics**: Calm markets, complacency, steady trends
        - **Best Strategies**: Long volatility, momentum
        - **Indicators**: VIX < 15, small daily moves
        - **Risk**: Can end suddenly with volatility explosion
        
        **Regime Detection Methods:**
        - **Technical**: Moving averages, volatility measures, momentum indicators
        - **Statistical**: Hidden Markov Models, regime-switching models
        - **Fundamental**: Economic indicators, market sentiment, correlations
        - **Quantitative**: Machine learning, pattern recognition
        """
        )

    st.markdown(
        """
    <div class="info-box">
    <h4>💡 Key Takeaways</h4>
    <ul>
    <li><strong>Returns without risk context are meaningless</strong> - Always consider volatility and drawdowns</li>
    <li><strong>Risk-adjusted metrics are crucial</strong> - Sharpe ratio is the gold standard</li>
    <li><strong>Past performance doesn't guarantee future results</strong> - Market regimes change</li>
    <li><strong>Diversification is the only free lunch</strong> - Don't put all eggs in one basket</li>
    <li><strong>Understand what you're measuring</strong> - Each metric tells a different story</li>
    </ul>
    </div>
    """,
        unsafe_allow_html=True,
    )

elif section == "Dashboard":
    hr()
    st.markdown(
        '<p class="big-font">📊 Dashboard — Market Overview</p>', unsafe_allow_html=True
    )
    with st.expander("Dashboard Tutorial", expanded=True):
        st.markdown(
            """
<div class="info-box">
<h4>🎯 Purpose</h4>
Context on trend, breadth, volume & sentiment.
</div>

**What you'll find**
- Real-time charts (multiple timeframes)
- Technical overlays (MA, BB, RSI)
- Sector/breadth snapshots

**How to use**
1. Start here daily  
2. Identify regime (trend vs chop)  
3. Match strategy to regime
""",
            unsafe_allow_html=True,
        )

elif section == "Strategy Builder":
    hr()
    st.markdown(
        '<p class="big-font">🔧 Strategy Builder — AI-Powered Strategy Development</p>',
        unsafe_allow_html=True,
    )

    st.markdown(
        """
    <div class="info-box">
    <h4>🎯 Purpose</h4>
    <p>The Strategy Builder helps you create, test, and optimize trading strategies using AI assistance. It combines human intuition with machine learning to find optimal parameter combinations.</p>
    </div>
    """,
        unsafe_allow_html=True,
    )

    with st.expander("🔄 Complete Strategy Development Workflow", expanded=True):
        st.markdown(
            """
        **Phase 1: Strategy Selection & Setup**
        1. **Choose Strategy Type**:
           - **Momentum**: Buy when price is rising (RSI > 70, MA crossover)
           - **Mean Reversion**: Buy when oversold (RSI < 30, Bollinger Band bounce)
           - **Breakout**: Buy when price breaks resistance with volume
           - **Custom**: Build your own combination of indicators
        
        2. **Configure Parameters**:
           - **Entry Conditions**: When to buy (multiple indicators can be combined)
           - **Exit Conditions**: When to sell (profit targets, stop losses, time-based)
           - **Position Sizing**: How much to buy (fixed $, % of portfolio, volatility-based)
           - **Risk Controls**: Maximum loss per trade, portfolio heat limits
        
        **Phase 2: Initial Backtesting**
        3. **Historical Testing**:
           - Select date range (longer = more reliable, but watch for regime changes)
           - Choose assets (start with liquid, well-known stocks)
           - Set transaction costs (typically 0.1-0.5% per trade)
           - Review initial results: returns, drawdowns, win rate, Sharpe ratio
        
        **Phase 3: AI Optimization**
        4. **Optimization Objective**:
           - **Sortino Ratio** (default): Focuses on downside risk
           - **Sharpe Ratio**: Classic risk-adjusted return
           - **Calmar Ratio**: Return per unit of max drawdown
           - **Custom**: Total return, win rate, profit factor
        
        5. **Parameter Optimization Process**:
           - AI suggests ≤3 parameter changes per iteration
           - Tests thousands of combinations efficiently
           - Avoids overfitting by respecting constraints
           - Validates on out-of-sample data
        
        **Phase 4: Validation & Refinement**
        6. **Robustness Testing**:
           - Test on different time periods
           - Test on different assets
           - Sensitivity analysis (how much do results change with small parameter tweaks?)
           - Walk-forward analysis (does it work on future data?)
        
        7. **Final Strategy Deployment**:
           - Document final parameters
           - Set up monitoring and alerts
           - Plan position sizing for live trading
           - Define stop conditions (when to turn off strategy)
        """
        )

    with st.expander("⚙️ Strategy Parameters - Detailed Guide", expanded=True):
        st.markdown(
            """
        **Technical Indicator Parameters**
        
        **RSI (Relative Strength Index)**
        - **Period**: Number of days to calculate (typical: 14)
          - Shorter (5-10): More sensitive, more signals, more false positives
          - Longer (20-30): Less sensitive, fewer signals, higher quality
        - **Overbought/Oversold Levels**: When to trigger signals
          - Conservative: 80/20 (fewer but stronger signals)
          - Aggressive: 70/30 (more signals but more noise)
          - Market dependent: Bull markets need higher levels
        
        **Moving Averages**
        - **Fast MA Period**: Shorter average (typical: 10-20 days)
        - **Slow MA Period**: Longer average (typical: 50-200 days)
        - **Type**: Simple (SMA) vs Exponential (EMA)
          - SMA: Equal weight to all periods
          - EMA: More weight to recent prices (faster response)
        - **Signal**: Fast crosses above slow = bullish
        
        **Bollinger Bands**
        - **Period**: Moving average length (typical: 20)
        - **Standard Deviations**: Band width (typical: 2.0)
          - Wider bands (2.5): Fewer false breakouts
          - Narrower bands (1.5): More sensitive to price moves
        - **Signals**: Price touches bands, band squeeze/expansion
        
        **Volume Indicators**
        - **Average Volume Period**: Days to calculate normal volume (typical: 20)
        - **Volume Multiplier**: How much above average to trigger (typical: 1.5-2.0x)
        - **Use Case**: Confirm breakouts, identify accumulation/distribution
        """
        )

        # Parameter sensitivity demonstration
        st.markdown("**📊 Parameter Sensitivity Example**")
        rsi_period = st.slider("RSI Period", 5, 30, 14)
        rsi_oversold = st.slider("RSI Oversold Level", 10, 40, 30)

        sensitivity_score = abs(14 - rsi_period) * 2 + abs(30 - rsi_oversold)

        if sensitivity_score < 10:
            st.success(
                f"✅ Conservative parameters - expect fewer, higher-quality signals"
            )
        elif sensitivity_score < 20:
            st.warning(f"⚠️ Moderate parameters - balanced approach")
        else:
            st.error(f"❌ Aggressive parameters - expect many signals, higher noise")

    with st.expander("🎯 AI Optimization Deep Dive", expanded=True):
        st.markdown(
            """
        **How AI Optimization Works**
        
        **1. Search Strategy**
        - **Grid Search**: Tests every combination (exhaustive but slow)
        - **Random Search**: Tests random combinations (faster, often effective)
        - **Bayesian Optimization**: Uses past results to guide future tests (smart)
        - **Genetic Algorithm**: Evolves parameters like natural selection
        
        **2. Objective Function Details**
        
        **Sortino Ratio (Recommended)**
        - **Formula**: (Return - Risk-Free Rate) / Downside Deviation
        - **Why Better**: Only penalizes downside volatility (upside good!)
        - **Interpretation**: Higher = better risk-adjusted returns
        - **Target**: Aim for >1.5 (excellent >2.0)
        
        **Sharpe Ratio (Classic)**
        - **Formula**: (Return - Risk-Free Rate) / Total Volatility
        - **Pro**: Widely understood, easy to compare
        - **Con**: Penalizes upside volatility unfairly
        - **Target**: Aim for >1.0 (excellent >2.0)
        
        **Calmar Ratio (Conservative)**
        - **Formula**: Annual Return / Maximum Drawdown
        - **Focus**: Emphasizes downside protection
        - **Best For**: Risk-averse investors
        - **Target**: Aim for >2.0 (excellent >3.0)
        
        **3. Overfitting Prevention**
        - **Cross-Validation**: Test on multiple time periods
        - **Parameter Constraints**: Limit how extreme parameters can be
        - **Sample Size**: Ensure enough trades for statistical significance
        - **Walk-Forward**: Optimize on past, test on future
        - **Simplicity Bias**: Prefer simpler strategies when performance similar
        
        **4. Optimization Constraints**
        - **Maximum Drawdown Limit**: Strategy killed if exceeds threshold
        - **Minimum Trade Count**: Need enough trades for confidence
        - **Parameter Bounds**: Prevent extreme values
        - **Transaction Cost Inclusion**: Realistic performance expectations
        """
        )

    with st.expander("🧮 Advanced Position Sizing Methods", expanded=True):
        st.markdown(
            """
        **1. Fixed Dollar Amount**
        - **Method**: Same dollar amount per trade
        - **Pros**: Simple, predictable exposure
        - **Cons**: Doesn't account for volatility or opportunity size
        - **Example**: Always trade $10,000 worth
        
        **2. Fixed Percentage of Portfolio**
        - **Method**: Same percentage of total equity per trade
        - **Pros**: Scales with portfolio size
        - **Cons**: Doesn't account for individual trade risk
        - **Example**: Always risk 2% of portfolio per trade
        
        **3. Volatility-Based Sizing**
        - **Method**: Larger positions in less volatile assets
        - **Formula**: Position Size = Target Risk / (Price × Historical Volatility)
        - **Pros**: Consistent risk across different assets
        - **Cons**: Complex, relies on past volatility
        
        **4. Kelly Criterion (Advanced)**
        - **Method**: Mathematically optimal sizing based on edge
        - **Formula**: f = (bp - q) / b, where b=odds, p=win rate, q=loss rate
        - **Pros**: Maximizes long-term growth
        - **Cons**: Can suggest very large positions, sensitive to estimates
        - **Practical**: Use 25-50% of Kelly suggestion for safety
        """
        )

        # Position sizing calculator
        st.markdown("**📱 Advanced Position Sizing Calculator**")
        col1, col2, col3 = st.columns(3)
        with col1:
            portfolio_value = st.number_input(
                "Portfolio Value ($)", value=100000, min_value=1000
            )
            risk_per_trade = st.slider("Risk per Trade (%)", 0.5, 5.0, 2.0, 0.1)
        with col2:
            entry_price = st.number_input("Entry Price ($)", value=50.0, min_value=0.01)
            stop_distance = st.slider("Stop Loss Distance (%)", 1.0, 15.0, 5.0, 0.5)
        with col3:
            win_rate = st.slider("Estimated Win Rate (%)", 30, 80, 55)
            avg_win_loss = st.slider("Avg Win/Loss Ratio", 1.0, 4.0, 2.0, 0.1)

        # Calculate different sizing methods
        risk_dollar = portfolio_value * (risk_per_trade / 100)
        stop_dollar = entry_price * (stop_distance / 100)
        shares_risk_based = int(risk_dollar / stop_dollar) if stop_dollar > 0 else 0

        # Kelly Criterion calculation
        p = win_rate / 100
        q = 1 - p
        b = avg_win_loss
        kelly_f = (b * p - q) / b if b > 0 else 0
        kelly_position = portfolio_value * max(0, min(kelly_f, 0.25))  # Cap at 25%
        kelly_shares = int(kelly_position / entry_price) if entry_price > 0 else 0

        # Display results
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Risk-Based Position", f"{shares_risk_based:,} shares")
            st.caption(f"${shares_risk_based * entry_price:,.0f} invested")
        with col2:
            st.metric("Kelly Criterion (25%)", f"{kelly_shares:,} shares")
            st.caption(f"${kelly_shares * entry_price:,.0f} invested")

    st.markdown(
        """
    <div class="warning-box">
    <h4>🚨 Optimization Warnings</h4>
    <ul>
    <li><strong>Curve Fitting</strong>: Don't over-optimize on limited data</li>
    <li><strong>Regime Changes</strong>: Market conditions change - strategies may stop working</li>
    <li><strong>Transaction Costs</strong>: High-frequency strategies killed by costs</li>
    <li><strong>Liquidity</strong>: Backtest assumes you can always trade - not true for all stocks</li>
    <li><strong>Look-Ahead Bias</strong>: Don't use future information in backtests</li>
    <li><strong>Survivorship Bias</strong>: Delisted stocks disappear from datasets</li>
    </ul>
    </div>
    """,
        unsafe_allow_html=True,
    )

elif section == "Live Trading":
    hr()
    st.markdown(
        '<p class="big-font">📈 Live Trading — Virtual Money Practice</p>',
        unsafe_allow_html=True,
    )
    with st.expander("Live Trading Tutorial", expanded=True):
        st.markdown(
            """
<div class="warning-box">
<h4>🎯 Safe environment</h4>
Practice execution & discipline with virtual cash.
</div>

**Includes**
- Portfolio, P/L & allocation
- Signal execution (market/limit/stop)
- Rolling metrics & benchmark compare

**Habits**
- Start small, log decisions, weekly reviews
""",
            unsafe_allow_html=True,
        )

elif section == "News Analysis":
    hr()
    st.markdown(
        '<p class="big-font">📰 News Analysis — Sentiment Integration</p>',
        unsafe_allow_html=True,
    )
    with st.expander("News Tutorial", expanded=True):
        st.markdown(
            """
<div class="info-box">
<h4>📈 Sentiment + context</h4>
Classify news flow, measure intensity, align with price.
</div>

**Use cases**
- Sentiment filters with technicals
- Regime tagging (risk-on/off)
- Event windows awareness
""",
            unsafe_allow_html=True,
        )

elif section == "Backtesting":
    hr()
    st.markdown(
        '<p class="big-font">🔄 Backtesting — Historical Strategy Validation</p>',
        unsafe_allow_html=True,
    )

    st.markdown(
        """
    <div class="info-box">
    <h4>🎯 What is Backtesting?</h4>
    <p>Backtesting is the process of testing a trading strategy using historical data to see how it would have performed in the past. It's like a time machine for trading strategies - but with important limitations.</p>
    </div>
    """,
        unsafe_allow_html=True,
    )

    with st.expander("🏗️ Backtesting Fundamentals", expanded=True):
        st.markdown(
            """
        **Core Principles of Valid Backtesting**
        
        **1. No Look-Ahead Bias**
        - **Rule**: Only use information available at the time of each trade
        - **Violation Example**: Using next day's close to make today's decision
        - **Correct Approach**: Make decision with today's data, execute at tomorrow's open
        - **Why Critical**: Look-ahead bias creates impossibly good results
        
        **2. Point-in-Time Data**
        - **Problem**: Company fundamentals get restated, stocks get delisted
        - **Solution**: Use data as it was known at the time
        - **Survivorship Bias**: Don't only test on stocks that survived
        - **Example**: Testing only on current S&P 500 ignores companies that failed
        
        **3. Realistic Execution**
        - **Market Orders**: Execute at next available price (usually next open)
        - **Slippage**: Price moves between decision and execution
        - **Bid-Ask Spread**: Cost of immediate execution
        - **Market Impact**: Large orders move prices against you
        
        **4. Transaction Costs**
        - **Brokerage Fees**: Direct cost per trade
        - **Spread Costs**: Difference between bid and ask
        - **Market Impact**: Your order affects the price
        - **Typical Range**: 0.1% to 0.5% per trade (round-trip)
        """
        )

    with st.expander("📊 Backtesting Methodology Deep Dive", expanded=True):
        st.markdown(
            """
        **Step-by-Step Backtesting Process**
        
        **1. Data Preparation**
        - **Clean Data**: Remove errors, adjust for splits/dividends
        - **Survivorship Filtering**: Include delisted companies
        - **Volume Filtering**: Ensure sufficient liquidity
        - **Price Filtering**: Remove penny stocks (< $5) for reliability
        
        **2. Signal Generation**
        - **Entry Signals**: When to buy (indicator crossovers, breakouts)
        - **Exit Signals**: When to sell (stop loss, take profit, time-based)
        - **Position Sizing**: How much to buy (fixed $, % portfolio, volatility-adjusted)
        - **Multiple Signals**: How to handle simultaneous opportunities
        
        **3. Trade Simulation**
        - **Order Timing**: When exactly orders are placed and filled
        - **Execution Price**: What price you actually get
        - **Partial Fills**: What if not all shares can be bought?
        - **Corporate Actions**: Handle splits, mergers, dividends
        
        **4. Performance Calculation**
        - **Mark-to-Market**: Daily portfolio value calculation
        - **Realized P&L**: Profit/loss from closed positions
        - **Unrealized P&L**: Current value of open positions
        - **Cash Management**: Dividends, interest on cash
        
        **Advanced Backtesting Techniques**
        
        **Walk-Forward Analysis**
        - **Method**: Optimize parameters on past data, test on future data
        - **Example**: Optimize on 2020-2021, test on 2022, then roll forward
        - **Benefit**: More realistic view of parameter stability
        - **Frequency**: Monthly, quarterly, or annual reoptimization
        
        **Out-of-Sample Testing**
        - **Method**: Hold back portion of data for final validation
        - **Split**: 70% optimization, 20% validation, 10% final test
        - **Purpose**: Ensure strategy wasn't overfit to the data
        - **Critical**: Never touch the final test set until the very end
        
        **Monte Carlo Analysis**
        - **Method**: Randomly resample historical trades to create alternative scenarios
        - **Purpose**: Understand range of possible outcomes
        - **Metrics**: Probability of drawdown, worst-case scenarios
        - **Limitation**: Assumes past trade distribution continues
        """
        )

    with st.expander("⚠️ Backtesting Pitfalls & Limitations", expanded=True):
        st.markdown(
            """
        **Common Backtesting Errors**
        
        **1. Overfitting (Curve Fitting)**
        - **Problem**: Strategy works perfectly on past data but fails live
        - **Cause**: Too many parameters, insufficient data
        - **Detection**: Dramatically different in-sample vs out-sample results
        - **Prevention**: Limit parameters, use cross-validation, keep it simple
        
        **2. Data Snooping**
        - **Problem**: Testing many strategies on same data finds false patterns
        - **Example**: Test 100 random strategies, 5 will look great by chance
        - **Solution**: Adjust for multiple testing, use fresh data for validation
        - **Rule**: If you've tested it, it's contaminated for future testing
        
        **3. Regime Change**
        - **Problem**: Market conditions change, strategies stop working
        - **Examples**: Low volatility → high volatility, growth → value rotation
        - **Detection**: Performance clustering in certain time periods
        - **Mitigation**: Test across multiple market regimes, stress testing
        
        **4. Liquidity Assumptions**
        - **Problem**: Backtests assume infinite liquidity
        - **Reality**: Small stocks, large positions, volatile markets = hard to trade
        - **Solution**: Volume filters, market impact models, position limits
        - **Rule**: If you can't easily buy/sell it, don't backtest it
        
        **5. Corporate Actions**
        - **Problem**: Splits, mergers, spin-offs complicate backtests
        - **Example**: Stock splits in half, price drops 50%, looks like crash
        - **Solution**: Adjust all historical prices for corporate actions
        - **Vendor Solutions**: Most data providers handle this automatically
        
        **Real-World vs Backtest Differences**
        
        **Execution Challenges**
        - **Gap Openings**: Stocks can gap past your stop loss
        - **Flash Crashes**: Extreme moves that don't appear in daily data
        - **Holiday Effects**: Reduced liquidity around holidays
        - **Economic Announcements**: Sudden volatility spikes
        
        **Psychological Factors**
        - **Backtest**: No emotions, perfect discipline
        - **Reality**: Fear, greed, second-guessing decisions
        - **Solution**: Paper trade first, start small, use automation
        - **Preparation**: Mentally rehearse drawdown scenarios
        
        **Technology Issues**
        - **System Failures**: Computers crash, internet goes down
        - **Data Delays**: Real-time feeds can lag or fail
        - **Order Routing**: Not all brokers execute at same price/speed
        - **Backup Plans**: Multiple brokers, offline contingencies
        """
        )

    with st.expander("📈 Interpreting Backtest Results", expanded=True):
        st.markdown(
            """
        **Performance Metrics Interpretation**
        
        **Returns Analysis**
        - **Total Return**: Overall gain/loss (but meaningless without time context)
        - **Annualized Return**: Scaled to yearly basis for comparison
        - **CAGR**: Smoothed growth rate (geometric mean)
        - **Benchmarking**: Always compare to buy-and-hold benchmark
        
        **Risk Assessment**
        - **Maximum Drawdown**: Worst peak-to-trough decline
          - <10%: Conservative strategy
          - 10-20%: Moderate risk
          - >20%: High risk, better have good returns to justify
        - **Volatility**: Annual standard deviation of returns
          - Compare to benchmark volatility
          - Higher vol needs higher returns to be worthwhile
        
        **Consistency Metrics**
        - **Win Rate**: % of profitable trades
          - High win rate (>60%) often means small wins, big losses
          - Low win rate (<40%) often means big wins, small losses
          - Sweet spot: 45-55% with good risk/reward ratio
        
        - **Profit Factor**: Gross profit / Gross loss
          - >1.5: Good strategy
          - >2.0: Excellent strategy
          - <1.2: Probably not worth the effort
        
        **Trade Analysis**
        - **Average Trade**: Mean profit per trade
        - **Best/Worst Trade**: Extreme outcomes
        - **Consecutive Losses**: Longest losing streak (psychological test)
        - **Trade Frequency**: How often you're trading (affects costs)
        
        **Red Flags in Backtest Results**
        - **Too Good to Be True**: >50% annual returns with low volatility
        - **Perfect Equity Curve**: Smooth upward slope (likely overfit)
        - **No Losing Periods**: Real strategies have bad months/years
        - **Sudden Performance Changes**: Strategy might be regime-dependent
        - **Very Few Trades**: Results not statistically significant
        """
        )

    # Interactive backtest analysis tool
    with st.expander("📱 Interactive Backtest Analyzer", expanded=True):
        st.markdown("**Analyze Your Backtest Results**")

        col1, col2, col3 = st.columns(3)
        with col1:
            total_return = st.number_input("Total Return (%)", value=45.0)
            years = st.number_input("Years Tested", value=3.0, min_value=0.1)
            max_dd = st.number_input("Max Drawdown (%)", value=12.0, min_value=0.1)

        with col2:
            total_trades = st.number_input("Total Trades", value=150, min_value=1)
            winning_trades = st.number_input("Winning Trades", value=85, min_value=0)
            avg_win = st.number_input("Avg Win (%)", value=3.2)

        with col3:
            avg_loss = st.number_input("Avg Loss (%)", value=-2.1)
            benchmark_return = st.number_input("Benchmark Return (%)", value=25.0)
            volatility = st.number_input(
                "Strategy Volatility (%)", value=18.0, min_value=0.1
            )

        # Calculations
        if total_trades > 0 and years > 0 and volatility > 0:
            cagr = ((1 + total_return / 100) ** (1 / years) - 1) * 100
            win_rate = (winning_trades / total_trades) * 100
            lose_rate = 100 - win_rate

            if avg_loss != 0:
                profit_factor = (winning_trades * avg_win) / (
                    abs(avg_loss) * (total_trades - winning_trades)
                )
            else:
                profit_factor = float("inf")

            sharpe = (cagr - 3) / volatility  # Assuming 3% risk-free rate
            calmar = cagr / max_dd if max_dd > 0 else 0

            # Display results
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("CAGR", f"{cagr:.1f}%")
                st.metric("Win Rate", f"{win_rate:.1f}%")
            with col2:
                st.metric("Sharpe Ratio", f"{sharpe:.2f}")
                st.metric("Calmar Ratio", f"{calmar:.2f}")
            with col3:
                st.metric("Profit Factor", f"{profit_factor:.2f}")
                st.metric("Excess Return", f"{cagr - benchmark_return/years:.1f}%")

            # Assessment
            with col4:
                st.markdown("**Assessment**")
                if sharpe > 1.5 and calmar > 2.0 and profit_factor > 1.5:
                    st.success("🟢 Excellent Strategy!")
                elif sharpe > 1.0 and calmar > 1.5 and profit_factor > 1.2:
                    st.success("🟡 Good Strategy")
                elif sharpe > 0.5 and profit_factor > 1.0:
                    st.warning("🟠 Marginal Strategy")
                else:
                    st.error("🔴 Poor Strategy")

    st.markdown(
        """
    <div class="warning-box">
    <h4>🚨 Critical Reminders</h4>
    <ul>
    <li><strong>Past Performance ≠ Future Results</strong> - The most important rule in finance</li>
    <li><strong>Start Small</strong> - Even great backtests can fail live</li>
    <li><strong>Paper Trade First</strong> - Practice with virtual money</li>
    <li><strong>Monitor Closely</strong> - Watch for performance degradation</li>
    <li><strong>Have Exit Rules</strong> - Know when to stop a strategy</li>
    <li><strong>Diversify</strong> - Don't rely on a single strategy</li>
    </ul>
    </div>
    """,
        unsafe_allow_html=True,
    )

elif section == "AI Assistant":
    hr()
    st.markdown(
        '<p class="big-font">🤖 AI Assistant — Intelligent Guidance</p>',
        unsafe_allow_html=True,
    )
    with st.expander("AI Assistant Tutorial", expanded=True):
        st.markdown(
            """
<div class="highlight">
<h4>🧠 Your mentor</h4>
Explain metrics, diagnose drawdowns, propose ranges, sanity-check ideas.
</div>

**Great prompts**
- "Sharpe vs Sortino — which for my strategy?"  
- "Why did drawdown spike in 2022?"  
- "Reasonable RSI bands for choppy regimes?"
""",
            unsafe_allow_html=True,
        )

elif section == "Best Practices":
    hr()
    st.markdown(
        '<p class="big-font">🎯 Best Practices & Advanced Tips</p>',
        unsafe_allow_html=True,
    )
    with st.expander("Optimization Best Practices", expanded=True):
        st.markdown(
            """
<div class="success-box">
<h4>Science of optimization</h4>
Hypothesis → test → validate OOS → implement → monitor.
</div>

- Use conservative thresholds; prefer Sortino over raw return
- Seek robust ranges, not single "peaks"
- Always hold back OOS data; rotate windows (walk-forward)
""",
            unsafe_allow_html=True,
        )

elif section == "Risk Management":
    hr()
    st.markdown(
        '<p class="big-font">⚠️ Risk Management Essentials</p>', unsafe_allow_html=True
    )
    with st.expander("Protect Your Capital", expanded=True):
        st.markdown(
            """
- Risk ≤ 1–2% per trade; scale down during drawdowns  
- Stops based on volatility (ATR) or structure; trail winners  
- Portfolio limits (exposure, correlation, MaxDD)  
- Monitor rolling Sharpe/Sortino, drawdown depth & duration
"""
        )

elif section == "Common Pitfalls":
    hr()
    st.markdown(
        '<p class="big-font">🚫 Common Pitfalls to Avoid</p>', unsafe_allow_html=True
    )
    with st.expander("Learn from others' mistakes", expanded=True):
        st.markdown(
            """
- **Overfitting**: unrealistic equity curves, too many knobs  
- **Look-ahead**: using future info; fix with next-bar execution  
- **Ignoring costs**: turnover × costs can erase edge  
- **Correlation neglect**: many positions, one risk
"""
        )

elif section == "Troubleshooting":
    hr()
    st.markdown(
        '<p class="big-font">🛠️ Troubleshooting & Support</p>', unsafe_allow_html=True
    )

    with st.expander("🔧 Common Issues & Solutions", expanded=True):
        st.markdown(
            """
        **Application Performance Issues**
        - **Slow Loading**: Clear browser cache and refresh the page
        - **Interface Not Responding**: Refresh the browser page
        - **Charts Not Displaying**: Check your internet connection and refresh
        
        **AI Assistant Problems**
        - **AI Features Not Working**: Ensure your Groq API key is properly configured
        - **Analysis Taking Too Long**: The free tier has rate limits - try again in a few minutes
        - **Connection Errors**: Check your internet connection and API key validity
        
        **Data & Trading Issues**
        - **No Data Available**: Market data may be temporarily unavailable - try again later
        - **Unusual Results**: Very high returns may indicate data issues - verify with multiple timeframes
        - **Virtual Trading Issues**: Reset your virtual portfolio in the Live Trading section
        """
        )

    with st.expander("📊 Performance Optimization Tips", expanded=True):
        st.markdown(
            """
        **For Better Performance:**
        - Use shorter date ranges for faster backtesting
        - Start with fewer assets and gradually increase
        - Close unused browser tabs to free up memory
        - Use the latest version of your web browser
        
        **Best Practices:**
        - Save your strategy configurations before major changes
        - Test strategies on different time periods
        - Start with small virtual positions before scaling up
        - Regular monitoring of strategy performance
        """
        )

    with st.expander("❓ Getting Help", expanded=True):
        st.markdown(
            """
        **Self-Help Resources:**
        - Review this tutorial for step-by-step guidance
        - Use the AI Assistant for strategy-specific questions
        - Check the "Best Practices" section for optimization tips
        
        **Platform Features:**
        - Interactive calculators in various sections
        - Built-in help text for all major features
        - Example strategies to learn from
        - Risk management guidelines
        """
        )

elif section == "Privacy & Safety":
    hr()
    st.markdown(
        '<p class="big-font">🔒 Privacy & Data Security</p>', unsafe_allow_html=True
    )

    with st.expander("🛡️ Data Protection", expanded=True):
        st.markdown(
            """
        **Your Data Security:**
        - **API Keys**: Stored locally on your device, never transmitted to third parties
        - **Trading Data**: All virtual trading data stays on your local system
        - **Personal Information**: We don't collect or store personal data
        - **Session Data**: Cleared when you close the application
        
        **Best Security Practices:**
        - Keep your API keys private and secure
        - Don't share your configuration with others
        - Use unique, strong API keys
        - Regularly monitor your API usage on provider dashboards
        """
        )

    with st.expander("⚠️ Important Disclaimers", expanded=True):
        st.markdown(
            """
        **Educational Purpose Only:**
        - This platform is designed for learning and educational purposes
        - Not intended as investment advice or financial recommendations
        - Past performance does not guarantee future results
        - Always consult with qualified financial advisors for investment decisions
        
        **Risk Warnings:**
        - All trading involves risk of financial loss
        - Start with virtual money before considering real trading
        - Never trade with money you cannot afford to lose
        - Market conditions can change rapidly and unpredictably
        
        **Virtual Trading Limitations:**
        - Virtual results may not reflect real trading conditions
        - Real trading involves additional costs, slippage, and execution delays
        - Emotional factors in real trading differ significantly from virtual trading
        """
        )

    with st.expander("🔐 API Key Security", expanded=True):
        st.markdown(
            """
        **Protecting Your API Keys:**
        - **Never Share**: Keep your API keys private and confidential
        - **Rotate Regularly**: Update your keys periodically for security
        - **Monitor Usage**: Check your API provider dashboards for unusual activity
        - **Limit Permissions**: Use API keys with minimal required permissions
        
        **If Your Key is Compromised:**
        1. Immediately revoke the compromised key from your API provider
        2. Generate a new API key
        3. Update your configuration with the new key
        4. Monitor your accounts for any unauthorized usage
        """
        )

    st.markdown(
        """
    <div class="warning-box">
    <h4>🚨 Legal Disclaimer</h4>
    <p>This trading platform is provided for educational and simulation purposes only. It is not intended to provide investment advice, and any trading decisions made using this platform are at your own risk. The developers and providers of this platform are not responsible for any financial losses incurred through the use of this software.</p>
    </div>
    """,
        unsafe_allow_html=True,
    )

# ---------- footer ----------
hr()
st.markdown(
    """
<div class="small-dim" style="text-align:center;">
<p>📚 <strong>Complete Trading Platform Tutorial</strong></p>
<p>🎯 Educational use only. Practice with virtual capital first.</p>
<p>🚀 Happy trading & continuous learning!</p>
</div>
""",
    unsafe_allow_html=True,
)
