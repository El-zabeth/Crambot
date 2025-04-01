
import streamlit as st
from firebase_config import *
import pyrebase

firebase_config = {
    "apiKey": os.getenv("FIREBASE_API_KEY"),
    "authDomain": os.getenv("FIREBASE_AUTH_DOMAIN"),
    "databaseURL": os.getenv("FIREBASE_DATABASE_URL"),
    "projectId": os.getenv("FIREBASE_PROJECT_ID"),
    "storageBucket": os.getenv("FIREBASE_STORAGE_BUCKET"),
    "messagingSenderId": os.getenv("FIREBASE_MESSAGING_SENDER_ID"),
    "appId": os.getenv("FIREBASE_APP_ID"),
    "measurementId": os.getenv("FIREBASE_MEASUREMENT_ID")
}
firebase = pyrebase.initialize_app(firebase_config)
auth = firebase.auth()

if not firebase_admin._apps:
    cred = credentials.Certificate(os.getenv("PATH"))  # Replace with your actual key file
    firebase_admin.initialize_app(cred)

# Updated Sign-Up and Sign-In functions using Pyrebase
def sign_up_ui():
    st.subheader("User Registration")

    email = st.text_input("Enter your email")
    password = st.text_input("Enter your password", type="password")
    name = st.text_input("Enter your name")

    if st.button("Sign Up"):
        if email and password and name:
            try:
                # Create a new user using Pyrebase
                user = auth.create_user_with_email_and_password(email, password)
                st.success(f"Successfully created user: {user['email']}")

                # Set session state to indicate successful sign-up
                st.session_state.logged_in = True
                st.session_state.user_email = user['email']
                st.rerun()
            except Exception as e:
                error_message = str(e)
                if "EMAIL_EXISTS" in error_message:
                    st.error("The email address is already registered. Please try with a different email.")
                elif "WEAK_PASSWORD"  in error_message:
                    st.error("Password should be at least 6 characters long.")
                else:
                    st.error("An error occurred during sign-up. Please try again.")
        else:
            st.error("Please fill in all fields.")

