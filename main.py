import streamlit as st
import sys
import os
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent / "src"))

from utils.config import load_config

# Page configuration
st.set_page_config(
    page_title="Algo Trading App",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

def main():
    """Main application entry point"""
    
    # Legal disclaimer in sidebar
    st.sidebar.warning("""
    ⚠️ **Educational Purpose Only**
    
    This platform is for educational and demonstration purposes only. 
    - Not financial advice
    - Paper trading simulation only  
    - No real money involved
    - Trade at your own risk
    """)
    st.sidebar.markdown("---")
    
    # Load configuration
    config = load_config()
    
    # Check API key status and show prompt
    has_api_key = bool(config.get('groq_api_key'))
    
    if not has_api_key:
        st.error("""
        🚨 **AI Features Disabled**: To unlock the full power of this platform, you need to configure your free Groq API key.
        
        **Missing Features:** AI Strategy Optimization • AI Assistant • Advanced Analysis • Intelligent Guidance
        
        ➡️ **Quick Setup**: Go to **📚 How to Use** → **Setup & Configuration** to add your free API key (takes 2 minutes)
        """)
    else:
        st.success("✅ **AI Features Active**: All platform capabilities are ready to use!")
    
    st.markdown("---")
    
    # App header
    st.title("🚀 Algorithmic Trading Platform")
    st.markdown("*AI-Powered Strategy Development & Optimization*")
    
    # Sidebar navigation
    with st.sidebar:
        st.header("🔧 Navigation")
        st.markdown("Use the pages above to navigate through the app")
        
        # API Key Configuration in Sidebar
        st.markdown("---")
        st.subheader("🔑 AI Setup")
        
        if has_api_key:
            st.success("✅ AI Features Active")
            if st.button("🧪 Test AI Connection"):
                try:
                    from src.analysis.ai_analyzer import GroqAnalyzer
                    analyzer = GroqAnalyzer(config['groq_api_key'])
                    st.success("✅ AI connection successful!")
                except Exception as e:
                    st.error(f"❌ Connection failed: {str(e)}")
        else:
            st.warning("⚠️ Setup Required")
            
            with st.expander("⚡ Quick Setup"):
                st.markdown("**Enable AI Features:**")
                
                # API key input
                main_api_key = st.text_input(
                    "Groq API Key",
                    type="password",
                    placeholder="gsk_...",
                    help="Get free key from console.groq.com",
                    key="main_api_key"
                )
                
                if st.button("💾 Save & Activate"):
                    if main_api_key and len(main_api_key) > 10:
                        from src.utils.config import save_api_key
                        success, message = save_api_key(main_api_key)
                        
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
                        st.warning("⚠️ Enter a valid API key")
                
                st.markdown("[🔗 Get Free Key](https://console.groq.com)")
        
        st.header("📊 Quick Stats")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Portfolio Value", "$100,000")
        with col2:
            st.metric("Today's P&L", "+$2,450", "2.45%")
    
    # Main content
    st.header("🎯 Welcome to Your Trading Platform")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.info("""
        **📊 Dashboard**
        
        View real-time market data, portfolio performance, and key metrics.
        
        ✅ *Available without API key*
        """)
    
    with col2:
        if has_api_key:
            st.success("""
            **🔧 Strategy Builder**
            
            Create and test your algorithmic trading strategies with AI optimization.
            
            🤖 *AI optimization active*
            """)
        else:
            st.warning("""
            **🔧 Strategy Builder**
            
            Create and test basic trading strategies. AI optimization requires setup.
            
            ⚠️ *Limited without API key*
            """)
    
    with col3:
        st.info("""
        **📈 Live Trading**
        
        Execute trades with virtual money and monitor real-time performance.
        
        ✅ *Available without API key*
        """)
    
    # Additional features section
    st.markdown("---")
    st.header("🚀 Advanced Features")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if has_api_key:
            st.success("""
            **🤖 AI Assistant**
            
            Get intelligent trading advice and strategy analysis.
            
            ✅ *Active and ready*
            """)
        else:
            st.error("""
            **🤖 AI Assistant**
            
            Get intelligent trading advice and strategy analysis.
            
            🔒 *Requires API key setup*
            """)
    
    with col2:
        if has_api_key:
            st.success("""
            **🔄 Advanced Backtesting**
            
            AI-powered performance analysis and optimization insights.
            
            ✅ *AI analysis available*
            """)
        else:
            st.warning("""
            **🔄 Basic Backtesting**
            
            Historical testing available. AI analysis requires setup.
            
            ⚠️ *Limited without API key*
            """)
    
    with col3:
        st.info("""
        **📰 News Analysis**
        
        Stay updated with market news and sentiment analysis.
        
        ✅ *Available without API key*
        """)
    
    # Setup reminder
    if not has_api_key:
        st.markdown("---")
        st.markdown("""
        <div style="background: linear-gradient(90deg, #ff6b6b, #4ecdc4); padding: 20px; border-radius: 10px; text-align: center; color: white; margin: 20px 0;">
        <h3>🚀 Unlock Full Platform Power</h3>
        <p>Get your free Groq API key in 2 minutes to enable AI optimization, intelligent assistance, and advanced analysis!</p>
        <p><strong>📚 Go to "How to Use" → "Setup & Configuration" to get started</strong></p>
        </div>
        """, unsafe_allow_html=True)
    
    # Recent activity
    st.header("📋 Recent Activity")
    
    # Placeholder for recent trades/signals
    activity_data = [
        {"Time": "10:30:15", "Action": "BUY", "Symbol": "RELIANCE", "Qty": 10, "Price": "₹2,456.75", "Strategy": "Momentum"},
        {"Time": "10:25:32", "Action": "SELL", "Symbol": "TCS", "Qty": 5, "Price": "₹3,234.50", "Strategy": "Mean Reversion"},
        {"Time": "10:20:18", "Action": "BUY", "Symbol": "INFY", "Qty": 15, "Price": "₹1,567.25", "Strategy": "Regime Switch"},
    ]
    
    st.dataframe(activity_data, use_container_width=True)
    
    # Footer
    st.markdown("---")
    st.markdown("💡 **Tip**: Use the navigation pages above to explore different features of the platform.")

if __name__ == "__main__":
    main()
