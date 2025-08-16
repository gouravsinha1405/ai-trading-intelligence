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

try:
    import streamlit as st
    import pandas as pd
    import numpy as np
    import plotly.graph_objects as go
    
    # Test critical imports
    from utils.config import load_config
    
    # If we get here, core imports work
    st.success("✅ All core dependencies loaded successfully!")
    
    # Import and run main app
    from main import main
    main()
    
except ImportError as e:
    import streamlit as st
    st.error(f"""
    🚨 **Import Error**: {str(e)}
    
    This usually means a dependency failed to install on Streamlit Cloud.
    
    **Common Solutions**:
    1. Check that all packages in requirements.txt are available on Linux
    2. Use `>=` instead of `==` for version pinning
    3. Remove packages that require system-level dependencies
    4. Try deploying again (sometimes it's a temporary issue)
    
    **Current Error**: `{str(e)}`
    """)
    
except Exception as e:
    import streamlit as st
    st.error(f"""
    🚨 **Application Error**: {str(e)}
    
    The dependencies loaded but there's an issue with the application code.
    
    **Error Details**: `{str(e)}`
    """)
