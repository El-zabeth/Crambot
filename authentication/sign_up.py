
import streamlit as st
from firebase_config import *
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

