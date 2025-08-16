"""
Mobile-Friendly UI Enhancements for the Trading Platform
"""

import streamlit as st


def inject_mobile_css():
    """Inject mobile-friendly CSS without affecting desktop experience"""
    mobile_css = """
    <style>
    /* Mobile-specific styles using media queries */
    @media only screen and (max-width: 768px) {
        /* Main content area improvements */
        .main .block-container {
            padding-top: 1rem !important;
            padding-left: 1rem !important;
            padding-right: 1rem !important;
            max-width: 100% !important;
        }

        /* Header and title adjustments */
        .main h1 {
            font-size: 1.8rem !important;
            line-height: 1.2 !important;
            margin-bottom: 0.5rem !important;
        }

        .main h2 {
            font-size: 1.4rem !important;
            line-height: 1.2 !important;
        }

        .main h3 {
            font-size: 1.2rem !important;
            line-height: 1.2 !important;
        }

        /* Sidebar adjustments */
        .css-1d391kg {
            width: 280px !important;
        }

        /* Metric cards stack vertically */
        [data-testid="metric-container"] {
            margin-bottom: 0.5rem !important;
        }

        /* Button improvements */
        .stButton > button {
            width: 100% !important;
            margin-bottom: 0.5rem !important;
            font-size: 0.9rem !important;
            padding: 0.5rem 1rem !important;
        }

        /* Form elements */
        .stTextInput > div > div > input,
        .stSelectbox > div > div > select,
        .stTextArea > div > div > textarea {
            font-size: 16px !important; /* Prevents zoom on iOS */
        }

        /* Data tables */
        .dataframe {
            font-size: 0.8rem !important;
        }

        .dataframe th,
        .dataframe td {
            padding: 0.25rem !important;
            text-align: left !important;
        }

        /* Alert boxes */
        .stAlert {
            padding: 0.75rem !important;
            margin-bottom: 0.5rem !important;
        }

        .stAlert > div {
            font-size: 0.9rem !important;
            line-height: 1.3 !important;
        }

        /* Expander adjustments */
        .streamlit-expanderHeader {
            font-size: 1rem !important;
            padding: 0.5rem !important;
        }

        /* Plotly charts responsive */
        .js-plotly-plot {
            width: 100% !important;
        }

        /* Column layout improvements for mobile */
        [data-testid="column"] {
            padding: 0.25rem !important;
        }

        /* Navigation improvements */
        .css-1kyxreq {
            margin-bottom: 0.5rem !important;
        }

        /* Progress indicators */
        .stProgress > div > div {
            height: 0.5rem !important;
        }

        /* Number inputs */
        .stNumberInput > div > div > input {
            font-size: 16px !important;
        }

        /* Date inputs */
        .stDateInput > div > div > input {
            font-size: 16px !important;
        }

        /* File uploader */
        .stFileUploader > div {
            padding: 0.5rem !important;
        }

        /* Markdown content */
        .markdown-text-container {
            font-size: 0.9rem !important;
            line-height: 1.4 !important;
        }

        /* Code blocks */
        .stCodeBlock {
            font-size: 0.8rem !important;
        }

        /* JSON display */
        .stJson {
            font-size: 0.8rem !important;
        }

        /* Tabs */
        .stTabs [data-baseweb="tab-list"] {
            gap: 0.25rem !important;
        }

        .stTabs [data-baseweb="tab"] {
            padding: 0.5rem 0.75rem !important;
            font-size: 0.9rem !important;
        }
    }

    /* Tablet-specific adjustments */
    @media only screen and (min-width: 769px) and (max-width: 1024px) {
        .main .block-container {
            padding-left: 2rem !important;
            padding-right: 2rem !important;
        }

        .css-1d391kg {
            width: 300px !important;
        }
    }

    /* Touch-friendly enhancements for all devices */
    @media (hover: none) and (pointer: coarse) {
        .stButton > button {
            min-height: 44px !important; /* iOS recommended touch target */
            padding: 0.75rem 1rem !important;
        }

        .stSelectbox > div {
            min-height: 44px !important;
        }

        .stRadio > div {
            gap: 0.75rem !important;
        }

        .stCheckbox > div {
            min-height: 44px !important;
        }
    }

    /* Dark mode mobile adjustments */
    @media (prefers-color-scheme: dark) and (max-width: 768px) {
        .main .block-container {
            background: #0e1117 !important;
        }

        .stAlert {
            border-radius: 0.5rem !important;
        }
    }

    /* Improve readability on small screens */
    @media only screen and (max-width: 480px) {
        .main h1 {
            font-size: 1.5rem !important;
        }

        .main h2 {
            font-size: 1.3rem !important;
        }

        /* Stack columns on very small screens */
        [data-testid="column"] {
            min-width: 100% !important;
            margin-bottom: 1rem !important;
        }

        /* Smaller metric cards */
        [data-testid="metric-container"] {
            padding: 0.5rem !important;
        }

        [data-testid="metric-container"] > div {
            font-size: 0.8rem !important;
        }

        /* Compact tables */
        .dataframe {
            font-size: 0.7rem !important;
        }

        .dataframe th,
        .dataframe td {
            padding: 0.1rem !important;
        }
    }
    </style>
    """
    st.markdown(mobile_css, unsafe_allow_html=True)


def mobile_friendly_columns(ratios, gap="small"):
    """Create mobile-friendly columns that stack on small screens"""
    return st.columns(ratios, gap=gap)


def mobile_friendly_metrics(metrics_data, cols_desktop=3, cols_mobile=2):
    """Display metrics in a mobile-friendly layout"""
    # Check if we're on mobile (simplified check)
    cols = st.columns(cols_desktop)

    for i, (label, value, delta) in enumerate(metrics_data):
        with cols[i % len(cols)]:
            if delta:
                st.metric(label, value, delta)
            else:
                st.metric(label, value)


def mobile_friendly_dataframe(df, height=400):
    """Display dataframe with mobile-friendly settings"""
    st.dataframe(df, use_container_width=True, height=height, hide_index=True)


def mobile_navigation_hint():
    """Show mobile navigation hints"""
    with st.container():
        st.markdown(
            """
        <div style="background: rgba(255, 193, 7, 0.1); padding: 0.5rem; border-radius: 0.25rem; border-left: 3px solid #ffc107; margin-bottom: 1rem;">
        📱 <strong>Mobile Tip:</strong> Use the sidebar (← arrow) to navigate between pages and access quick settings.
        </div>
        """,
            unsafe_allow_html=True,
        )


def responsive_chart_config():
    """Get responsive configuration for Plotly charts"""
    return {
        "displayModeBar": False,
        "responsive": True,
        "toImageButtonOptions": {
            "format": "png",
            "filename": "chart",
            "height": 500,
            "width": 700,
            "scale": 1,
        },
    }


def mobile_friendly_form(form_key):
    """Create a mobile-friendly form container"""
    return st.form(form_key, clear_on_submit=False)


def is_mobile():
    """Simple check to determine if user might be on mobile"""
    # This is a basic implementation - in a real app you might use JS
    # For now, we'll make responsive design universal
    return False  # Always return False to make design universal


def mobile_sidebar_content():
    """Optimize sidebar content for mobile"""
    with st.sidebar:
        # Collapsible sections for mobile
        with st.expander("📊 Quick Stats", expanded=False):
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Portfolio", "$100K")
            with col2:
                st.metric("P&L", "+2.45%")

        with st.expander("🔧 Settings", expanded=False):
            st.selectbox("Theme", ["Auto", "Light", "Dark"], key="theme_select")
            st.selectbox("Currency", ["USD", "INR", "EUR"], key="currency_select")

        with st.expander("📱 Mobile Options", expanded=False):
            st.checkbox("Compact View", key="compact_view")
            st.checkbox("Auto-refresh", key="auto_refresh")


def show_mobile_welcome():
    """Show a mobile-optimized welcome message"""
    st.markdown(
        """
    <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                padding: 1rem; border-radius: 0.5rem; color: white; text-align: center; margin-bottom: 1rem;">
    <h3 style="margin: 0; color: white;">📱 Mobile Trading</h3>
    <p style="margin: 0.5rem 0 0 0; opacity: 0.9;">Optimized for your mobile device</p>
    </div>
    """,
        unsafe_allow_html=True,
    )


def mobile_action_buttons(buttons_config):
    """Create mobile-friendly action buttons"""
    cols = st.columns(len(buttons_config))

    for i, (label, key, type_) in enumerate(buttons_config):
        with cols[i]:
            st.button(label, key=key, type=type_, use_container_width=True)


def mobile_info_cards(cards_data):
    """Display information cards optimized for mobile"""
    for title, content, type_ in cards_data:
        if type_ == "success":
            st.success(f"**{title}**\n\n{content}")
        elif type_ == "warning":
            st.warning(f"**{title}**\n\n{content}")
        elif type_ == "error":
            st.error(f"**{title}**\n\n{content}")
        else:
            st.info(f"**{title}**\n\n{content}")


def setup_mobile_viewport():
    """Setup mobile viewport meta tag"""
    st.markdown(
        """
    <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no">
    """,
        unsafe_allow_html=True,
    )
