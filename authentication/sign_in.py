import streamlit as st
from firebase_config import *
from firebase_admin import credentials,auth as admin_auth
import pyrebase


firebase_config = {
    "apiKey": st.secrets.firebase_client.apiKey,
    "authDomain": st.secrets.firebase_client.authDomain,
    "databaseURL":st.secrets.firebase_client.databaseURL,
    "projectId": st.secrets.firebase_client.projectId,
    "storageBucket": st.secrets.firebase_client.storageBucket,
    "messagingSenderId": st.secrets.firebase_client.messagingSenderId,
    "appId": st.secrets.firebase_client.appId,
    "measurementId": st.secrets.firebase_client.measurementId,
}


firebase = pyrebase.initialize_app(firebase_config)
auth = firebase.auth()

if not firebase_admin._apps:
    cred = credentials.Certificate(os.getenv("PATH"))  # Replace with your actual key file
    firebase_admin.initialize_app(cred)
    
def sign_in_ui():
    st.subheader("User Login")

    email = st.text_input("Enter your email")
    password = st.text_input("Enter your password", type="password")

    if st.button("Sign In"):
        if email and password:
            try:
                # Get user details using Firebase Admin SDK

                # Authenticate user using Pyrebase
                if email == "admin@gmail.com" and password == "adminpassword":
                    st.session_state.is_admin = True
                    st.session_state.logged_in = True
                else:
                    user = auth.sign_in_with_email_and_password(email, password)
                    #st.success(f"Successfully logged in as {user['email']}")

                    # Set session state for logged-in user
                    st.session_state.logged_in = True
                    st.session_state.is_admin = False
                    st.session_state.user_email = user['email']
                    st.rerun()
            except admin_auth.UserNotFoundError:
                st.error("Invalid credentials: The email address is not registered.")
            except Exception as e:
                error_message = str(e)
                if "EMAIL_NOT_FOUND" in error_message:
                    st.error("Invalid credentials: The email address is not registered.")
                elif "INVALID_LOGIN_CREDENTIALS" in error_message:
                    st.error("Invalid credentials: Incorrect password.")
                else:
                    st.error(f"Error: {e}")
        else:
            st.error("Please fill in both fields.")