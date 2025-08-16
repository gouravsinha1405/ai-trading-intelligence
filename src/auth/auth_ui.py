"""
Authentication UI Components
Login forms, admin panel, and user management interface
"""

import streamlit as st
import pandas as pd
from datetime import datetime
from src.auth.auth_manager import auth_manager

def show_login_page():
    """Display login page"""
    st.set_page_config(
        page_title="AI Trading Platform - Login",
        page_icon="🔐",
        layout="centered"
    )
    
    st.title("🔐 AI Trading Platform")
    st.subheader("Secure Login")
    
    # Login form
    with st.form("login_form"):
        st.markdown("### Sign In")
        username = st.text_input("Username", placeholder="Enter your username")
        password = st.text_input("Password", type="password", placeholder="Enter your password")
        
        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:
            login_button = st.form_submit_button("🔑 Login", use_container_width=True)
        
        if login_button:
            if username and password:
                success, role, message = auth_manager.authenticate(username, password)
                
                if success:
                    auth_manager.login_user(username, role)
                    st.success(f"Welcome back, {username}!")
                    st.balloons()
                    st.rerun()
                else:
                    st.error(message)
            else:
                st.warning("Please enter both username and password")
    
    # Info section
    st.markdown("---")
    # Demo credentials removed for production security
    
    # Features preview
    st.markdown("### 🌟 Platform Features")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **📊 Trading Features:**
        - Real-time market data
        - AI-powered strategy optimization
        - Backtesting engine
        - Virtual trading
        """)
    
    with col2:
        st.markdown("""
        **🤖 AI Assistant:**
        - Groq-powered analysis
        - Market sentiment analysis
        - Strategy recommendations
        - Risk management
        """)

def show_logout_button():
    """Show logout button in sidebar"""
    with st.sidebar:
        st.markdown("---")
        current_user = auth_manager.get_current_user()
        user_role = st.session_state.get("user_role", "user")
        
        st.markdown(f"**👤 Logged in as:** {current_user}")
        st.markdown(f"**🔰 Role:** {user_role.title()}")
        
        if st.button("🚪 Logout", use_container_width=True):
            auth_manager.logout_user()
            st.success("Logged out successfully!")
            st.rerun()

def show_admin_panel():
    """Display admin panel for user management"""
    if not auth_manager.is_admin():
        st.error("🚫 Access denied. Admin privileges required.")
        return
    
    st.title("👨‍💼 Admin Panel")
    st.subheader("User Management")
    
    tab1, tab2, tab3, tab4 = st.tabs(["👥 Users", "➕ Add User", "🔐 Change Password", "📊 System Info"])
    
    with tab1:
        st.markdown("### Current Users")
        users = auth_manager.get_all_users()
        
        if users:
            # Create DataFrame for better display
            df = pd.DataFrame(users)
            df['created_at'] = pd.to_datetime(df['created_at']).dt.strftime('%Y-%m-%d %H:%M')
            df['last_login'] = pd.to_datetime(df['last_login'], errors='coerce').dt.strftime('%Y-%m-%d %H:%M')
            df['status'] = df['active'].apply(lambda x: '✅ Active' if x else '❌ Disabled')
            
            # Display users table
            st.dataframe(
                df[['username', 'role', 'status', 'created_at', 'last_login']],
                use_container_width=True,
                hide_index=True
            )
            
            # User management actions
            st.markdown("### User Actions")
            col1, col2 = st.columns(2)
            
            with col1:
                user_to_modify = st.selectbox(
                    "Select User to Modify",
                    [u['username'] for u in users if u['username'] != 'admin']
                )
                
                if user_to_modify:
                    col1a, col1b = st.columns(2)
                    
                    with col1a:
                        if st.button("🔄 Toggle Status"):
                            success, message = auth_manager.toggle_user_status(user_to_modify)
                            if success:
                                st.success(message)
                                st.rerun()
                            else:
                                st.error(message)
                    
                    with col1b:
                        if st.button("🗑️ Delete User", type="secondary"):
                            success, message = auth_manager.delete_user(user_to_modify)
                            if success:
                                st.success(message)
                                st.rerun()
                            else:
                                st.error(message)
        else:
            st.info("No users found.")
    
    with tab2:
        st.markdown("### Add New User")
        
        with st.form("add_user_form"):
            new_username = st.text_input("Username", placeholder="Enter new username")
            new_password = st.text_input("Password", type="password", placeholder="Enter password (min 6 chars)")
            new_role = st.selectbox("Role", ["user", "admin"])
            
            if st.form_submit_button("➕ Create User"):
                if new_username and new_password:
                    success, message = auth_manager.create_user(new_username, new_password, new_role)
                    if success:
                        st.success(message)
                        st.rerun()
                    else:
                        st.error(message)
                else:
                    st.warning("Please fill in all fields")
    
    with tab3:
        st.markdown("### Change Password")
        
        with st.form("change_password_form"):
            target_user = st.selectbox("User", [u['username'] for u in users])
            old_password = st.text_input("Current Password", type="password")
            new_password = st.text_input("New Password", type="password", placeholder="Min 6 characters")
            confirm_password = st.text_input("Confirm New Password", type="password")
            
            if st.form_submit_button("🔐 Change Password"):
                if new_password != confirm_password:
                    st.error("New passwords don't match")
                elif target_user and old_password and new_password:
                    success, message = auth_manager.change_password(target_user, old_password, new_password)
                    if success:
                        st.success(message)
                    else:
                        st.error(message)
                else:
                    st.warning("Please fill in all fields")
    
    with tab4:
        st.markdown("### System Information")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Total Users", len(users))
            active_users = len([u for u in users if u['active']])
            st.metric("Active Users", active_users)
        
        with col2:
            admin_users = len([u for u in users if u['role'] == 'admin'])
            st.metric("Admin Users", admin_users)
            regular_users = len([u for u in users if u['role'] == 'user'])
            st.metric("Regular Users", regular_users)
        
        # Recent activity
        st.markdown("### Recent Activity")
        recent_logins = [u for u in users if u.get('last_login')]
        recent_logins.sort(key=lambda x: x.get('last_login', ''), reverse=True)
        
        if recent_logins:
            for user in recent_logins[:5]:
                last_login = pd.to_datetime(user['last_login']).strftime('%Y-%m-%d %H:%M')
                st.write(f"👤 **{user['username']}** - Last login: {last_login}")
        else:
            st.info("No recent activity")

def require_auth(show_admin_only: bool = False):
    """
    Decorator function to require authentication
    Use this to protect pages
    """
    # Check session timeout
    if auth_manager.check_session_timeout():
        st.warning("⏰ Session expired. Please log in again.")
        show_login_page()
        st.stop()
    
    # Check authentication
    if not auth_manager.is_authenticated():
        show_login_page()
        st.stop()
    
    # Check admin access if required
    if show_admin_only and not auth_manager.is_admin():
        st.error("🚫 Access denied. Admin privileges required.")
        st.stop()
    
    # Show logout button in sidebar
    show_logout_button()

def show_user_profile():
    """Show user profile and settings"""
    st.subheader("👤 User Profile")
    
    current_user = auth_manager.get_current_user()
    user_role = st.session_state.get("user_role", "user")
    login_time = st.session_state.get("login_time")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.info(f"""
        **Username:** {current_user}
        **Role:** {user_role.title()}
        **Login Time:** {login_time.strftime('%Y-%m-%d %H:%M:%S') if login_time else 'Unknown'}
        """)
    
    with col2:
        if st.button("🔐 Change Password"):
            st.session_state["show_password_change"] = True
    
    # Password change form
    if st.session_state.get("show_password_change", False):
        st.markdown("### Change Password")
        
        with st.form("user_password_change"):
            old_password = st.text_input("Current Password", type="password")
            new_password = st.text_input("New Password", type="password")
            confirm_password = st.text_input("Confirm New Password", type="password")
            
            col1, col2 = st.columns(2)
            with col1:
                if st.form_submit_button("Update Password"):
                    if new_password != confirm_password:
                        st.error("New passwords don't match")
                    elif old_password and new_password:
                        success, message = auth_manager.change_password(
                            current_user, old_password, new_password
                        )
                        if success:
                            st.success(message)
                            st.session_state["show_password_change"] = False
                            st.rerun()
                        else:
                            st.error(message)
                    else:
                        st.warning("Please fill in all fields")
            
            with col2:
                if st.form_submit_button("Cancel"):
                    st.session_state["show_password_change"] = False
                    st.rerun()
