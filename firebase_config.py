import firebase_admin
import streamlit as st
from firebase_admin import credentials, firestore, auth
import os

# Path to your Firebase service account key JSON file from environment variable

cred_path = os.getenv("FIREBASE_CRED_PATH", "C:\\Users\\lizbe\\Desktop\\pr1\\config\\crambot-b6b8d-firebase-adminsdk-fbsvc-2727c2d548.json")

# Initialize Firebase Admin SDK (only if not initialized)
if len(firebase_admin._apps) == 0:
    # Initialize Firebase if not already initialized
    cred = credentials.Certificate(cred_path)
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