#!/usr/bin/env python3
"""
Streamlit Cloud startup script for AI Trading Intelligence Platform
Handles imports gracefully and provides helpful error messages
"""

import sys
import os
from pathlib import Path

# Add src directory to Python path
current_dir = Path(__file__).parent
src_path = current_dir / "src"
sys.path.insert(0, str(src_path))

def main():
    try:
        import streamlit as st
        st.set_page_config(
            page_title="AI Trading Intelligence", 
            page_icon="🤖",
            layout="wide"
        )
        
        # Test core dependencies
        try:
            import pandas as pd
            import numpy as np
            import plotly.graph_objects as go
            st.success("✅ Core dependencies loaded successfully!")
        except ImportError as e:
            st.error(f"❌ Core dependency missing: {e}")
            return
        
        # Test data sources
        data_sources = []
        try:
            import yfinance as yf
            data_sources.append("yfinance")
        except ImportError:
            pass
            
        try:
            from jugaad_data.nse import stock_df
            data_sources.append("jugaad-data")
        except ImportError:
            pass
        
        if data_sources:
            st.info(f"📊 Available data sources: {', '.join(data_sources)}")
        else:
            st.warning("⚠️ No data sources available")
        
        # Test AI integration
        try:
            from utils.config import load_config
            config = load_config()
            if config.get('groq_api_key'):
                st.success("✅ AI integration ready!")
            else:
                st.warning("⚠️ Configure Groq API key in sidebar for AI features")
        except ImportError as e:
            st.warning(f"⚠️ AI features not available: {e}")
        
        # Import and run main app
        from main import main as main_app
        main_app()
        
    except ImportError as e:
        import streamlit as st
        st.error(f"""
        🚨 **Import Error**: {str(e)}
        
        **Common Solutions**:
        1. Check requirements.txt for missing packages
        2. Ensure Python version compatibility (using Python 3.11)
        3. Try redeploying the app
        4. Check Streamlit Cloud logs for details
        
        **Error**: `{str(e)}`
        """)
        
    except Exception as e:
        import streamlit as st
        st.error(f"""
        🚨 **Application Error**: {str(e)}
        
        **Error Details**: `{str(e)}`
        
        Please check the Streamlit Cloud logs for more information.
        """)

if __name__ == "__main__":
    main()
