import streamlit as st
st.set_page_config(layout="wide")


from authentication.sign_in import sign_in_ui
from authentication.sign_up import sign_up_ui
from authentication.admin import admin_page


from pyrebase import pyrebase
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
# from langchain_community.embeddings.spacy_embeddings import SpacyEmbeddings
from langchain_community.embeddings.openai import OpenAIEmbeddings
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
import logging
from openai import OpenAI
from serpapi import GoogleSearch
from querying import query_pinecone  # Import query_pinecone from querying.py
import textwrap

# Suppress unnecessary logs
logging.getLogger("langchain").setLevel(logging.ERROR)

client = OpenAI()
# Load environment variables
load_dotenv()
serpapi_key = os.getenv("SERPAPI_KEY")

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
    cred = credentials.Certificate(st.secrets["firebase_service_account"])  # Replace with your actual key file
    firebase_admin.initialize_app(cred)

# embeddings = SpacyEmbeddings(model_name="en_core_web_sm")
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-large"
)
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
    # if openai_api_key is None:
    #     st.error("API Key not defined in .env file")
    #     return  # Early exit if API key is missing
    st.session_state.history.append({"role": "user", "content": ques})
    # with st.chat_message("user"):
    #     st.markdown(ques)

    llm = ChatOpenAI(model_name="gpt-4o", temperature=0)


    # Initialize agent_scratchpad as an empty string or any default value
    agent_scratchpad = ""  # This could be modified if you need to store intermediate steps

    # Define the prompt for generating both answers and revision questions
    _instruction = textwrap.dedent(
        """\
        You are a helpful assistant. Answer the user's question as detailed as possible using ONLY the provided context.  
                      
        IMPORTANT: 
        1. If the answer is not in the provided context, YOU MUST REPOND WITH: "The answer is not available in the textbook."
        2. If provided context is relevant and can be used to answer the user's query: After answering the user's query, generate revision questions based on the topic and context collected. 
        These questions should relate specifically to the content and be answerable based on the collected context itself.
        """
    )
    prompt = ChatPromptTemplate.from_messages([
        ("system", _instruction),
        ("placeholder", "{chat_history}"),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),

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
            "chat_history": [],         # Now chat_history is a list of message tuples
            "agent_scratchpad": agent_scratchpad  # Initialize agent scratchpad
        })

        output = response['output']
        
        # If no relevant answer, perform a Google search
        if "not available" in output:
            search_results = google_search(ques)
            output += "\n\n*Check these external sources:*\n" + search_results
        
        _response = output
    except Exception as e:
        _response = f"Error during query execution: {e}"

    st.session_state.history.append({"role": "assistant", "content": output})
    # with st.chat_message("assistant"):
    #     st.markdown(response)


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


def generate_gpt_response(query: str, context: str = ""):
    try:
        if not context.strip():  # Ensure GPT is not called when no context is available
            return None

        # Format messages for chat-based models
        messages = [
            {"role": "system", "content": "You are a helpful AI tutor. Answer the user's question and generate 2-3 related revision questions for self-study."},
            {"role": "user", "content": f"Query: {query}\n\nContext: {context}\n\n1. Provide a clear and concise answer.\n2. Generate 2-3 revision questions based on the answer."}
        ]

        # Make the API call to GPT-4
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            max_tokens=300,  # Increased token limit to fit answer + questions
            temperature=0.7,
            n=1
        )

        return response.choices[0].message.content.strip()

    except Exception as e:
        logging.error(f"Error generating GPT response: {e}")
        return "Sorry, I encountered an error while generating the response."

def upload_pdf():
    st.session_state.uploadpdf = True
    st.session_state.directquery = False

    # User input box
    user_query = st.chat_input("Ask me anything...")

    if st.session_state.active_page == "Upload PDF" and user_query:  # Only process if it's the active page
        user_input(user_query)

    with st.sidebar:
        st.title("Menu:")
        pdf_doc = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button", accept_multiple_files=True)
        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                raw_text = pdf_read(pdf_doc)
                text_chunks = get_chunks(raw_text)
                vector_store(text_chunks)
                st.success("Done")

    # Display previous chat messages
    for message in st.session_state.history:
        if isinstance(message, dict) and "role" in message and "content" in message:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

def direct_query():
    st.session_state.uploadpdf = False
    st.session_state.directquery = True

    if "history" not in st.session_state:
        st.session_state.history = []

    # Display previous chat messages
    for message in st.session_state.history:
        if isinstance(message, dict) and "role" in message and "content" in message:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

    # User input box
    user_query = st.chat_input("Ask me anything...")

    if st.session_state.active_page == "Direct query" and user_query:  # Only process if it's the active page
        # Display user query immediately in chat history
        st.session_state.history.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.markdown(user_query)

        response = ""

        # Step 1: Retrieve relevant chunks from Pinecone using the external query_pinecone function
        results = query_pinecone(user_query)

        response = None  # Initialize response as None

        if results:
            # Extract full_text from metadata and filter out empty/irrelevant results
            valid_texts = [
                match.get("metadata", {}).get("full_text", "").strip()
                for match in results
                if match.get("metadata", {}).get("full_text", "").strip()  # Ensure it's non-empty
            ]

            if valid_texts:  # Only proceed if there is meaningful content
                context = "\n\n---\n\n".join(valid_texts)
                response = generate_gpt_response(user_query, context)

        # If there are no relevant Pinecone results, fall back to Google search
        if not response:
            search_results = google_search(user_query)

            if search_results:
                response = "No relevant information found in the dataset.\n\n*Check these external sources:*\n"
                response += search_results
            else:
                response = "Sorry, no relevant data found."

        # Display assistant response and update chat history
        st.session_state.history.append({"role": "assistant", "content": response})
        with st.chat_message("assistant"):
            st.markdown(response)


# Main app logic
def chatbot_page():
    menu = ["Direct query", "Upload PDF"]
    choice = st.sidebar.selectbox("Menu", menu)

    if choice == "Direct query":
        st.session_state.active_page = "Direct query"
        direct_query()
    elif choice == "Upload PDF":
        st.session_state.active_page = "Upload PDF"
        upload_pdf()



    with st.sidebar:
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
        st.markdown(
        """
        <style>
        .logout-button {
            position: fixed;
            top: 10px;
            right: 20px;
            z-index: 100;
        }
        </style>
        """,
        unsafe_allow_html=True
        )

        # Logout Button
        st.markdown('<div class="logout-button">', unsafe_allow_html=True)
        if st.button("Logout"):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            
            # Rerun the app to go back to sign-in page
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

def main():
    st.markdown("""
        <style>
        .big-font {
            font-size:48px !important;
            font-family:'Lucida Handwriting', sans-serif;
            font-weight: bold;
            color: rgba(0, 0, 0, 0.5)
            text-align: center;
        }
        </style>
        """, unsafe_allow_html=True)

    # Apply the custom class to the title
    st.markdown('<p class="big-font">CRAMBOT</p>', unsafe_allow_html=True)

    # Ensure session state is properly initialized
    if "logged_in" not in st.session_state:
        st.session_state.logged_in = False
    if "is_admin" not in st.session_state:
        st.session_state.is_admin = False
    if "active_page" not in st.session_state:  # Track the active page
        st.session_state.active_page = "Direct query"

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
