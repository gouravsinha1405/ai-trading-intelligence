import streamlit as st
import sys
from pathlib import Path

# Add src to path for imports  
sys.path.append(str(Path(__file__).parent.parent / "src"))

from auth.auth_ui import require_auth, show_admin_panel

# Page config
st.set_page_config(
    page_title="Admin Panel - AI Trading Platform",
    page_icon="👨‍💼",
    layout="wide"
)

# Require admin authentication
require_auth(show_admin_only=True)

# Show admin panel
show_admin_panel()
