from sentence_transformers import SentenceTransformer
import pinecone
from pinecone import Pinecone as pc
from dotenv import load_dotenv
import os
import logging
import streamlit as st

# Set up logging
logging.basicConfig(level=logging.INFO)

# Load environment variables
load_dotenv()
pinecone_api_key = st.secrets["api_keys"]["PINECONE_API_KEY"]
pinecone_environment = st.secrets["api_keys"]["PINECONE_ENVIRONMENT"]


# Initialize Pinecone
try:
    pc = pinecone.Pinecone(api_key=pinecone_api_key, environment=pinecone_environment)
except Exception as e:
    logging.error(f"Error initializing Pinecone: {e}")
    raise

# Define index name
index_name = "textbook-dataset"

# Check if the index exists
if index_name in pc.list_indexes().names():
    index = pc.Index(index_name)
    logging.info(f"Connected to Pinecone index: {index_name}")
else:
    raise ValueError(f"Index '{index_name}' not found in Pinecone.")

# Initialize Sentence Transformer
try:
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    logging.info("Sentence Transformer model loaded successfully.")
except Exception as e:
    logging.error(f"Error loading Sentence Transformer model: {e}")
    raise

# Query function
def query_pinecone(query: str, top_k: int = 5,  min_score: float = 0.5):
    namespaces = ["textbook_viiisci", "textbook_ixscience", "textbook_xeng1", "textbook_xhistory", "textbook_viiieng1", "textbook_viiihistory", "textbook_ixeconomics",
                  "textbook_ixeng2", "textbook_xeng3", "textbook_ixeng1", "textbook_ixpolitics", "textbook_xeng2", "textbook_viiieng2", "textbook_xpolitics", "textbook_xgeog",
                  "textbook_viiipolitics", "textbook_viiigeog", "textbook_ixcontemporary", "textbook_xeconomics"]
    query_embedding = embedder.encode([query]).tolist()[0]
    best_results = []

    for namespace in namespaces:
        try:
            response = index.query(vector=query_embedding, top_k=top_k, include_metadata=True, namespace=namespace)
            matches = response.get("matches", [])
            # Apply score filtering
            filtered_matches = [match for match in matches if match["score"] >= min_score]
            best_results.extend(filtered_matches)
            
        except Exception as e:
            logging.error(f"Error querying namespace '{namespace}': {e}")
    
    if not best_results:
        logging.info("No results found across all namespaces.")
        return []

    # Sort results by score (highest first)
    best_results.sort(key=lambda x: x["score"], reverse=True)

    logging.info(best_results)
    
    return best_results[:top_k]