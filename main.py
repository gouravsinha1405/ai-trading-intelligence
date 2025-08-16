#!/usr/bin/env python3
"""
AI Trading Intelligence Platform - Main Application
A comprehensive algorithmic trading platform with AI-powered analysis
"""

import sys
from pathlib import Path

import streamlit as st

# --- Path setup ---
ROOT = Path(__file__).resolve().parent
sys.path.append(str(ROOT / "src"))

def main():
    """Main application entry point"""
    
    # Page configuration
    st.set_page_config(
        page_title="AI Trading Intelligence",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded",
        menu_items={
            'Get Help': 'https://github.com/gouravsinha1405/ai-trading-intelligence',
            'Report a bug': 'https://github.com/gouravsinha1405/ai-trading-intelligence/issues',
            'About': """
            # AI Trading Intelligence Platform
            
            A comprehensive algorithmic trading platform featuring:
            - Real-time market data analysis
            - AI-powered trading strategies
            - Risk management tools
            - Mobile-responsive design
            
            Built with Streamlit, powered by AI.
            """
        }
    )
    
    # Import and inject mobile CSS
    try:
        from utils.mobile_ui import inject_mobile_css
        inject_mobile_css()
    except ImportError:
        pass  # Mobile UI is optional
    
    # Import authentication
    try:
        from auth.auth_ui import require_auth
        require_auth()
    except ImportError:
        st.error("Authentication module not found. Please check your installation.")
        return
    
    # Main content
    st.title("🤖 AI Trading Intelligence Platform")
    st.markdown("---")
    
    # Welcome message
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown("""
        ### Welcome to Your AI Trading Platform! 🚀
        
        Navigate through the different sections using the sidebar:
        
        📊 **Dashboard** - Overview of your portfolio and market data  
        🔧 **Strategy Builder** - Create and test trading strategies  
        📈 **Live Trading** - Execute trades and monitor positions  
        📰 **News Analysis** - Market sentiment and news analysis  
        🔄 **Backtesting** - Test strategies on historical data  
        🤖 **AI Assistant** - Get AI-powered market insights  
        📚 **How to Use** - Learn how to use the platform  
        
        ---
        
        ### Quick Start Guide:
        
        1. **Set up your API keys** in the sidebar (Groq for AI features)
        2. **Visit the Dashboard** to see market overview
        3. **Build strategies** in the Strategy Builder
        4. **Test them** with the Backtesting tool
        5. **Get AI insights** from the AI Assistant
        
        """)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; padding: 20px;'>
        Built with ❤️ using Streamlit | AI-Powered Trading Platform | 
        <a href='https://github.com/gouravsinha1405/ai-trading-intelligence' target='_blank'>GitHub</a>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
