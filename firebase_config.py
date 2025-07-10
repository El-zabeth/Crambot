import firebase_admin
import streamlit as st
from firebase_admin import credentials, firestore, auth
import os
import json

# Path to your Firebase service account key JSON file from environment variable

cred_path ={
    "type": os.getenv("FIREBASE_TYPE"),
    "project_id": os.getenv("FIREBASE_PROJECT_ID"),
    "private_key_id": os.getenv("FIREBASE_PRIVATE_KEY_ID"),
    "private_key": os.getenv("FIREBASE_PRIVATE_KEY"),  # Ensure newlines
    "client_email": os.getenv("FIREBASE_CLIENT_EMAIL"),
    "client_id": os.getenv("FIREBASE_CLIENT_ID"),
    "auth_uri": os.getenv("FIREBASE_AUTH_URI"),
    "token_uri": os.getenv("FIREBASE_TOKEN_URI"),
    "auth_provider_x509_cert_url": os.getenv("FIREBASE_AUTH_PROVIDER_CERT_URL"),
    "client_x509_cert_url": os.getenv("FIREBASE_CLIENT_CERT_URL"),
    "universe_domain": os.getenv("FIREBASE_UNIVERSE_DOMAIN"),
}

# Initialize Firebase Admin SDK (only if not initialized)
if len(firebase_admin._apps) == 0:
    # Initialize Firebase if not already initialized
    cred = credentials.Certificate(json.loads(st.secrets["firebase_service_account"]))
    firebase_admin.initialize_app(cred)
else:
    print("Firebase is already initialized.")

# Firestore client
db = firestore.client()

# Save user profile to Firestore
def save_user_profile(user, name):
    try:
        # Use user.uid as the document ID to ensure unique profiles
        db.collection('users').document(user.uid).set({
            'name': name,
            'email': user.email
        })
        st.success(f"User profile for {user.email} saved.")
    except Exception as e:
        st.error(f"Error saving user profile: {e}")