"""
Authentication Manager for AI Trading Platform
Provides secure login, session management, and admin controls
"""

import hashlib
import hmac
import json
import os
import secrets
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import streamlit as st


class AuthManager:
    """Secure authentication and session management"""

    def __init__(self):
        self.users_file = "users.json"
        self.session_timeout = 24 * 60 * 60  # 24 hours in seconds
        self.secret_key = os.getenv("AUTH_SECRET_KEY", secrets.token_hex(32))
        self.admin_username = os.getenv("ADMIN_USERNAME", "admin")
        self.admin_password = os.getenv(
            "ADMIN_PASSWORD", "admin123"
        )  # Change in production

        # Initialize users database
        self._init_users_db()

    def _init_users_db(self):
        """Initialize users database with admin user"""
        if not os.path.exists(self.users_file):
            admin_data = {
                "users": {
                    self.admin_username: {
                        "password_hash": self._hash_password(self.admin_password),
                        "role": "admin",
                        "created_at": datetime.now().isoformat(),
                        "last_login": None,
                        "active": True,
                    }
                },
                "created_at": datetime.now().isoformat(),
            }
            self._save_users(admin_data)

    def _hash_password(self, password: str) -> str:
        """Hash password with salt"""
        salt = self.secret_key.encode()
        key = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, 100000)
        return key.hex()

    def _verify_password(self, password: str, password_hash: str) -> bool:
        """Verify password against hash"""
        return hmac.compare_digest(self._hash_password(password), password_hash)

    def _load_users(self) -> Dict:
        """Load users from file"""
        try:
            with open(self.users_file, "r") as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            self._init_users_db()
            return self._load_users()

    def _save_users(self, users_data: Dict):
        """Save users to file"""
        with open(self.users_file, "w") as f:
            json.dump(users_data, f, indent=2)

    def authenticate(self, username: str, password: str) -> Tuple[bool, str, str]:
        """
        Authenticate user
        Returns: (success, role, message)
        """
        users_data = self._load_users()
        users = users_data.get("users", {})

        if username not in users:
            return False, "", "Invalid username or password"

        user = users[username]
        if not user.get("active", True):
            return False, "", "Account is disabled"

        if not self._verify_password(password, user["password_hash"]):
            return False, "", "Invalid username or password"

        # Update last login
        user["last_login"] = datetime.now().isoformat()
        users_data["users"][username] = user
        self._save_users(users_data)

        return True, user["role"], "Login successful"

    def create_user(
        self, username: str, password: str, role: str = "user"
    ) -> Tuple[bool, str]:
        """
        Create new user (admin only)
        Returns: (success, message)
        """
        users_data = self._load_users()
        users = users_data["users"]

        if username in users:
            return False, "Username already exists"

        if len(password) < 6:
            return False, "Password must be at least 6 characters"

        users[username] = {
            "password_hash": self._hash_password(password),
            "role": role,
            "created_at": datetime.now().isoformat(),
            "last_login": None,
            "active": True,
        }

        users_data["users"] = users
        self._save_users(users_data)

        return True, f"User '{username}' created successfully"

    def delete_user(self, username: str) -> Tuple[bool, str]:
        """
        Delete user (admin only)
        Returns: (success, message)
        """
        if username == self.admin_username:
            return False, "Cannot delete admin user"

        users_data = self._load_users()
        users = users_data["users"]

        if username not in users:
            return False, "User not found"

        del users[username]
        users_data["users"] = users
        self._save_users(users_data)

        return True, f"User '{username}' deleted successfully"

    def toggle_user_status(self, username: str) -> Tuple[bool, str]:
        """
        Enable/disable user (admin only)
        Returns: (success, message)
        """
        if username == self.admin_username:
            return False, "Cannot modify admin user status"

        users_data = self._load_users()
        users = users_data["users"]

        if username not in users:
            return False, "User not found"

        users[username]["active"] = not users[username]["active"]
        status = "enabled" if users[username]["active"] else "disabled"

        users_data["users"] = users
        self._save_users(users_data)

        return True, f"User '{username}' {status} successfully"

    def change_password(
        self, username: str, old_password: str, new_password: str
    ) -> Tuple[bool, str]:
        """
        Change user password
        Returns: (success, message)
        """
        users_data = self._load_users()
        users = users_data["users"]

        if username not in users:
            return False, "User not found"

        if not self._verify_password(old_password, users[username]["password_hash"]):
            return False, "Current password is incorrect"

        if len(new_password) < 6:
            return False, "New password must be at least 6 characters"

        users[username]["password_hash"] = self._hash_password(new_password)
        users_data["users"] = users
        self._save_users(users_data)

        return True, "Password changed successfully"

    def get_all_users(self) -> List[Dict]:
        """Get all users for admin panel"""
        users_data = self._load_users()
        users = users_data["users"]

        user_list = []
        for username, user_info in users.items():
            user_list.append(
                {
                    "username": username,
                    "role": user_info["role"],
                    "created_at": user_info["created_at"],
                    "last_login": user_info.get("last_login"),
                    "active": user_info.get("active", True),
                }
            )

        return sorted(user_list, key=lambda x: x["created_at"])

    def is_authenticated(self) -> bool:
        """Check if current user is authenticated"""
        return st.session_state.get("authenticated", False)

    def is_admin(self) -> bool:
        """Check if current user is admin"""
        return st.session_state.get("user_role") == "admin"

    def get_current_user(self) -> Optional[str]:
        """Get current username"""
        return st.session_state.get("username")

    def login_user(self, username: str, role: str):
        """Set user session"""
        st.session_state["authenticated"] = True
        st.session_state["username"] = username
        st.session_state["user_role"] = role
        st.session_state["login_time"] = datetime.now()

    def logout_user(self):
        """Clear user session"""
        for key in ["authenticated", "username", "user_role", "login_time"]:
            if key in st.session_state:
                del st.session_state[key]

    def check_session_timeout(self) -> bool:
        """Check if session has timed out"""
        if not self.is_authenticated():
            return True

        login_time = st.session_state.get("login_time")
        if login_time and (datetime.now() - login_time).seconds > self.session_timeout:
            self.logout_user()
            return True

        return False


# Global auth manager instance
auth_manager = AuthManager()
