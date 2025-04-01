import os
import logging
import streamlit as st
from dotenv import load_dotenv
import openai
from serpapi import GoogleSearch
from querying import query_pinecone  # Import query_pinecone from querying.py

# Suppress unnecessary logs
logging.getLogger("langchain").setLevel(logging.ERROR)

# Load environment variables
load_dotenv()
openai_api_key = os.getenv("API_KEY")
serpapi_key = os.getenv("SERPAPI_KEY")

# Set up OpenAI API key
openai.api_key = openai_api_key

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

# Function to generate response using GPT-3.5
def generate_gpt_response(query: str, context: str = ""):
    try:
        if not context.strip():  # Ensure GPT is not called when no context is available
            return None

        # Format messages for chat-based models
        messages = [
            {"role": "system", "content": "You are a helpful AI tutor. Answer the user's question and generate 2-3 related revision questions for self-study."},
            {"role": "user", "content": f"Query: {query}\n\nContext: {context}\n\n1. Provide a clear and concise answer.\n2. Generate 2-3 revision questions based on the answer."}
        ]

        # Make the API call to GPT-3.5 Turbo
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=messages,
            max_tokens=300,  # Increased token limit to fit answer + questions
            temperature=0.7,
            n=1
        )

        return response["choices"][0]["message"]["content"].strip()

    except Exception as e:
        logging.error(f"Error generating GPT response: {e}")
        return "Sorry, I encountered an error while generating the response."

# Main Streamlit app
def main():
    # Maintain chat history in session state
    if "history" not in st.session_state:
        st.session_state.history = []

    st.title("CRAMBOT")

    # Display previous chat messages
    for message in st.session_state.history:
        if isinstance(message, dict) and "role" in message and "content" in message:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

    # User input box
    user_query = st.chat_input("Ask me anything...")

    if user_query:
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

if __name__ == "__main__":
    main()