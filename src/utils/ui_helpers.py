"""
UI Helper Functions for the Trading Platform
"""
import streamlit as st

from .config import load_config


def check_api_key_status():
    """Check if Groq API key is configured"""
    config = load_config()
    return bool(config.get("groq_api_key"))


def show_api_key_warning(feature_name="AI features"):
    """Show a warning when API key is required but not configured"""
    st.warning(
        f"""
    ⚠️ **{feature_name.title()} Require Setup**: This feature needs a free Groq API key to work.

    📋 **Quick Setup** (2 minutes):
    1. Go to **📚 How to Use** → **Setup & Configuration**
    2. Get your free key from [console.groq.com](https://console.groq.com)
    3. Enter it in the configuration section

    💡 **Why needed?** This powers the AI optimization, analysis, and assistance features.
    """
    )


def show_feature_status(feature_name, required_for_ai=True):
    """Show the status of a feature based on API key availability"""
    has_api_key = check_api_key_status()

    if required_for_ai and not has_api_key:
        st.error(
            f"""
        🔒 **{feature_name} - Limited Mode**: AI features are disabled without API key setup.

        **Available**: Basic functionality
        **Missing**: AI optimization, intelligent analysis, advanced insights

        ➡️ Configure your free API key to unlock full capabilities.
        """
        )
        return False
    elif required_for_ai and has_api_key:
        st.success(f"✅ **{feature_name} - Full Power**: All AI features are active!")
        return True
    else:
        # Feature doesn't require API key
        return True


def show_api_setup_button():
    """Show a button to navigate to API setup"""
    if st.button("🔧 Setup API Key", type="primary"):
        st.info(
            "Navigate to **📚 How to Use** → **Setup & Configuration** to set up your API key."
        )


def get_feature_availability():
    """Get a dictionary of feature availability based on API key status"""
    has_api_key = check_api_key_status()

    return {
        "ai_optimization": has_api_key,
        "ai_assistant": has_api_key,
        "advanced_analysis": has_api_key,
        "basic_backtesting": True,
        "virtual_trading": True,
        "market_data": True,
        "news_feeds": True,
        "tutorials": True,
    }


def show_feature_comparison():
    """Show a comparison of available vs premium features"""
    has_api_key = check_api_key_status()

    st.markdown("**🎯 Platform Feature Status:**")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**✅ Always Available:**")
        st.markdown(
            """
        - 📊 Market data viewing
        - 📈 Basic strategy creation
        - 🔄 Simple backtesting
        - 💰 Virtual trading
        - 📰 News feeds
        - 📚 Educational content
        """
        )

    with col2:
        if has_api_key:
            st.markdown("**🚀 AI Features (Active):**")
            st.success(
                """
            - 🤖 AI strategy optimization
            - 🧠 Intelligent trading assistant
            - 📊 Advanced performance analysis
            - 💡 Smart risk assessment
            - 🎯 Personalized guidance
            - 🔍 AI-powered insights
            """
            )
        else:
            st.markdown("**🔒 AI Features (Setup Required):**")
            st.error(
                """
            - 🤖 AI strategy optimization
            - 🧠 Intelligent trading assistant
            - 📊 Advanced performance analysis
            - 💡 Smart risk assessment
            - 🎯 Personalized guidance
            - 🔍 AI-powered insights
            """
            )
