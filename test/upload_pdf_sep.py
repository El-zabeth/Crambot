import streamlit as st
st.set_page_config(layout="wide")


from authentication.sign_in import sign_in_ui
from authentication.sign_up import sign_up_ui

from pyrebase import pyrebase
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.embeddings.spacy_embeddings import SpacyEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.tools.retriever import create_retriever_tool
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_tool_calling_agent
from firebase_config import *
from authentication.auth_component import firebase_auth_component
import firebase_admin
from firebase_admin import credentials,auth as admin_auth
import os
from serpapi import GoogleSearch


# Load environment variables
load_dotenv()
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

embeddings = SpacyEmbeddings(model_name="en_core_web_sm")
serpapi_key = os.getenv("SERPAPI_KEY")

# Define helper functions for PDF processing and chatbot (same as your current ones)
# Helper functions for PDF processing and chatbot
def pdf_read(pdf_doc):
    text = ""
    for pdf in pdf_doc:
        pdf_reader = PdfReader(pdf) 
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_text(text)
    return chunks 

def load_formulas():
    return [
        {"name": "Quadratic Formula", "formula": r"x = \frac{-b \pm \sqrt{b^2 - 4ac}}{2a}", "description": "Solves ax² + bx + c = 0."},
        {"name": "Logarithm Rule", "formula": r"\log_b(xy) = \log_b x + \log_b y", "description": "Log multiplication property."},
        {"name": "Pythagorean Theorem", "formula": r"a^2 + b^2 = c^2", "description": "Right triangle relation."},
        {"name": "Area of Circle", "formula": r"A = \pi r^2", "description": "Formula for the area of a circle."},
        {"name": "Circumference of Circle", "formula": r"C = 2\pi r", "description": "Perimeter of a circle."},
        {"name": "Volume of Sphere", "formula": r"V = \frac{4}{3} \pi r^3", "description": "Volume of a sphere."},
        {"name": "Derivative Rule", "formula": r"\frac{d}{dx} x^n = n x^{n-1}", "description": "Differentiation rule for power functions."},
        {"name": "Chain Rule", "formula": r"\frac{d}{dx} f(g(x)) = f'(g(x)) g'(x)", "description": "Used for nested functions."},
        {"name": "Integral of Power", "formula": r"\int x^n dx = \frac{x^{n+1}}{n+1} + C", "description": "Basic integral formula."},
        {"name": "Newton's Second Law", "formula": r"F = ma", "description": "Force is mass times acceleration."},
        {"name": "Kinetic Energy", "formula": r"KE = \frac{1}{2}mv^2", "description": "Formula for kinetic energy."},
        {"name": "Ohm's Law", "formula": r"V = IR", "description": "Relation of voltage, current, and resistance."}
    ]

def display_formula_buttons():
    formulas = load_formulas()
    
    st.write("### Click on a Formula Button to Display its Information:")
    
    # Iterate through formulas and create buttons
    for formula in formulas:
        if st.button(formula['name']):
            st.write(f"*Formula:* {formula['formula']}")
            st.write(f"*Description:* {formula['description']}")

# Function to perform a Google search using SerpAPI
def google_search(query: str, num_results: int = 5):
    try:
        params = {
            "q": query,
            "num": num_results,
            "api_key": serpapi_key
        }
        search = GoogleSearch(params)
        results = search.get_dict()

        if "organic_results" in results:
            sources = []
            for res in results["organic_results"]:
                title = res.get("title", "No Title")
                link = res.get("link", "#")
                sources.append(f"- [{title}]({link})")
            return "\n".join(sources)
    except Exception as e:
        return f"Error fetching search results: {str(e)}"

    return "No sources found."


def vector_store(text_chunks):
    if not os.path.exists("faiss_db"):
        os.makedirs("faiss_db")
    try:
        formulas = load_formulas()
        for formula in formulas:
            if st.button(formula["name"]):
                st.latex(formula["formula"])
                st.write(formula["description"])
        #formula_texts = [f"{f['name']}: {f['formula']} - {f['description']}" for f in formulas]
        #combined_data = text_chunks + formula_texts

        vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
        vector_store.save_local("faiss_db")
        print("FAISS index created and saved successfully.")
    except Exception as e:
        print(f"Error creating FAISS index: {e}")

def get_conversational_chain(tools, ques):
    openai_api_key = os.getenv("API_KEY")
    if openai_api_key is None:
        st.error("API Key not defined in .env file")
        return  # Early exit if API key is missing

    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0, api_key=openai_api_key)

    # Initialize chat_history as a list of messages (it can start as empty or contain an initial system message)
    chat_history = [("system", "You are a helpful assistant.")]

    # Initialize agent_scratchpad as an empty string or any default value
    agent_scratchpad = ""  # This could be modified if you need to store intermediate steps

    # Define the prompt for generating both answers and revision questions
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a helpful assistant. Answer the user's question as detailed as possible using the provided context. 
                      If the answer is not in the provided context, simply say, "The answer is not available in the context." 
                      After answering the user's query, generate revision questions directly based on the provided paragraph. 
                      These questions should relate specifically to the content and be answerable using the paragraph itself."""),
        
        ("placeholder", "{chat_history}"),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),

        ("assistant", """Based on the paragraph you provided, here are some specific revision questions:
                        1. What are the main concepts or key ideas discussed in the paragraph?
                        2. How does [specific process/idea] mentioned in the paragraph work?
                        3. What is the role of [a particular element or concept] as described in the paragraph?
                        4. Can you summarize how [an event or fact] is explained in the paragraph?
                        5. What conclusion or outcome is drawn from the information provided in the paragraph?""")
    ])

    # Set up the tools for the agent
    tool = [tools]

    # Create the agent and execute
    agent = create_tool_calling_agent(llm, tool, prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tool, verbose=True)

    try:
        # Pass input and placeholders when invoking the agent
        response = agent_executor.invoke({
            "input": ques,
            "chat_history": chat_history,         # Now chat_history is a list of message tuples
            "agent_scratchpad": agent_scratchpad  # Initialize agent scratchpad
        })

        output = response['output']
        
        # If no relevant answer, perform a Google search
        if "not available" in output:
            search_results = google_search(ques)
            output += "\n\n*Check these external sources:*\n" + search_results
        
        st.write(output)

    except Exception as e:
        st.error(f"Error during query execution: {e}")



def user_input(user_question):
    index_file_path = "faiss_db/index.faiss"
    if os.path.exists(index_file_path):
        try:
            new_db = FAISS.load_local("faiss_db", embeddings, allow_dangerous_deserialization=True)
            retriever = new_db.as_retriever()
            retrieval_chain = create_retriever_tool(retriever, "pdf_extractor", "This tool is to give answers to queries from the pdf.")
            get_conversational_chain(retrieval_chain, user_question)
        except Exception as e:
            st.error(f"Error loading FAISS index: {e}")
    else:
        st.error(f"No FAISS index found at {index_file_path}. Please upload and process a PDF file first.")


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
                "Last Login": user.user_metadata.last_sign_in_timestamp if user.user_metadata else "N/A",
                "Account Created": user.user_metadata.creation_timestamp if user.user_metadata else "N/A",
                "Disabled": user.disabled  # Whether the account is disabled
            })
    except Exception as e:
        st.error(f"Error fetching users: {e}")

    return users


# Function to disable/enable user account and refresh user list
def toggle_user_status(uid, current_status):
    if st.button(f"Confirm {'Disable' if not current_status else 'Enable'}"):
        try:
            # Toggle account status
            admin_auth.update_user(uid, disabled=not current_status)
            st.session_state["action_done"] = True  # Mark an action as completed
            st.success(f"User {uid} has been {'disabled' if not current_status else 'enabled'}.")

            # Refresh the list of users after status change
            st.session_state["fetched_users"] = fetch_all_users()

        except Exception as e:
            st.error(f"Error updating user status: {e}")


# Function to delete user account and refresh user list
def delete_user(uid):
    try:
        admin_auth.delete_user(uid)  # Attempt to delete the user
        st.session_state["action_done"] = True
        
        # Verify if the user still exists after deletion attempt
        try:
            admin_auth.get_user(uid)  # Attempt to fetch the user
            st.error(f"User {uid} was not deleted and still exists.")
        except firebase_admin.auth.UserNotFoundError:
            st.success(f"User {uid} no longer exists.")

        # Refresh the list of users
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
                st.write(f"*Email Verified:* {user['Email Verified']}")
                st.write(f"*Display Name:* {user['Display Name']}")
                st.write(f"*Phone Number:* {user['Phone Number']}")
                st.write(f"*Account Created:* {user['Account Created']}")
                st.write(f"*Last Login:* {user['Last Login']}")
                st.write(f"*Account Disabled:* {user['Disabled']}")

                # Columns for buttons
                col2, col3 = st.columns(2)

                # Reset Password Button with confirmation prompt

                # Disable/Enable Account Button
                with col2:
                    status_text = "Disable" if not user["Disabled"] else "Enable"
                    if st.button(f"{status_text} User - {user['UID']}", key=f"toggle_{user['UID']}"):
                        toggle_user_status(user["UID"], user["Disabled"])

                # Delete User Button
                with col3:
                    if st.button(f"Delete User - {user['UID']}", key=f"delete_{user['UID']}"):
                        delete_user(user["UID"])

    if st.button("Logout"):
        st.session_state.logged_in = False
        st.session_state.is_admin = False


# Main app logic
def chatbot_page():
    
    user_question = st.text_input("Ask a Question from the PDF Files")
    if user_question:
        user_input(user_question)

    with st.sidebar:
        st.title("Menu:")
        pdf_doc = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button", accept_multiple_files=True)
        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                raw_text = pdf_read(pdf_doc)
                text_chunks = get_chunks(raw_text)
                vector_store(text_chunks)
                st.success("Done")
                
        if "show_formulas" not in st.session_state:
            st.session_state.show_formulas = False

        if st.button("📘 Maths Formulas"):
            st.session_state.show_formulas = not st.session_state.show_formulas  # Toggle state

        # Show formulas if button is clicked
        if st.session_state.show_formulas:
            st.subheader("📘 Formula Library")
            formulas = load_formulas()
            for formula in formulas:
                if st.button(formula["name"]):
                    st.latex(formula["formula"])
                    st.write(formula["description"])


def main():
    st.header("RAG-Based Chat with PDF")

    # Ensure session state is properly initialized
    if "logged_in" not in st.session_state:
        st.session_state.logged_in = False
    if "is_admin" not in st.session_state:
        st.session_state.is_admin = False

    # Redirect based on login type
    if st.session_state.logged_in:
        if st.session_state.is_admin:
            admin_page()  # Redirect to Admin Panel
        else:
            chatbot_page()  # Redirect to Chatbot Page
    else:
        # User is not logged in, show authentication options
        menu = ["Sign In", "Sign Up"]
        choice = st.sidebar.selectbox("Menu", menu)

        if choice == "Sign Up":
            sign_up_ui()  # Show Sign-up form
        elif choice == "Sign In":
            sign_in_ui()  # Show Sign-in form


if __name__ == "__main__":
    main()