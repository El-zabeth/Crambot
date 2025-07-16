import streamlit as st
from firebase_config import *
#from authentication.auth_component import firebase_auth_component
import firebase_admin
from firebase_admin import credentials,auth as admin_auth
from datetime import datetime


# Ensure session state variables exist
if "action_done" not in st.session_state:
    st.session_state["action_done"] = False  # Track if an action was performed
if "fetched_users" not in st.session_state:
    st.session_state["fetched_users"] = []  # Store fetched users

# Function to fetch all users
def fetch_all_users():
    users = []
    try:
        for user in admin_auth.list_users().iterate_all():
            users.append({
                "UID": user.uid,
                "Email": user.email,
                "Email Verified": user.email_verified,
                "Display Name": user.display_name if user.display_name else "N/A",
                "Phone Number": user.phone_number if user.phone_number else "N/A",
                "Photo URL": user.photo_url if user.photo_url else "N/A",
                "Last Login": datetime.fromtimestamp(user.user_metadata.last_sign_in_timestamp / 1000).strftime('%B %d, %Y at %I:%M %p') if user.user_metadata and user.user_metadata.last_sign_in_timestamp else "N/A",
                "Account Created": datetime.fromtimestamp(user.user_metadata.creation_timestamp / 1000).strftime('%B %d, %Y at %I:%M %p') if user.user_metadata and user.user_metadata.creation_timestamp else "N/A",
            })
    except Exception as e:
        st.error(f"Error fetching users: {e}")

    return users



# Function to delete user account and refresh user list
def delete_user(uid, email=None):
    try:
        admin_auth.delete_user(uid)  # Attempt to delete the user
        st.session_state["action_done"] = True

        # Check if user still exists
        try:
            admin_auth.get_user(uid)
            st.error(f"User {email or uid} was not deleted and still exists.")
        except firebase_admin.auth.UserNotFoundError:
            st.success(f"✅ User {email or uid} successfully deleted.")

        # Refresh the user list
        st.session_state["fetched_users"] = fetch_all_users()

    except firebase_admin.exceptions.FirebaseError as firebase_error:
        st.error(f"Firebase error occurred: {firebase_error}")

    except Exception as e:
        st.error(f"Error deleting user: {e}")

# Admin Page Function
def admin_page():
    st.subheader("Admin Panel - User Management")

    if st.button("Fetch All Users") or st.session_state.get("action_done", False):
        st.session_state["fetched_users"] = fetch_all_users()  # Fetch users and store in session
        st.session_state["action_done"] = False  # Reset after fetching users

    users = st.session_state.get("fetched_users", [])  # Load users from session state

    if users:
        for user in users:
            with st.expander(f"User: {user['Email']}"):
                st.write(f"*UID:* {user['UID']}")
                #st.write(f"*Email Verified:* {user['Email Verified']}")
                #st.write(f"*Display Name:* {user['Display Name']}")
                #st.write(f"*Phone Number:* {user['Phone Number']}")
                st.write(f"*Account Created:* {user['Account Created']}")
                st.write(f"*Last Login:* {user['Last Login']}")

                if st.button(f"Delete User - {user['UID']}", key=f"delete_{user['UID']}"):
                        delete_user(user["UID"])

    if st.button("Logout"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        
        # Rerun the app to go back to sign-in page
        st.rerun()