import os
import re
import spacy
import PyPDF2
import numpy as np
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from gensim import corpora
from gensim.models import LdaModel
import pinecone
from pinecone import ServerlessSpec
import streamlit as st

# Load environment variables
load_dotenv()
pinecone_api_key = st.secrets["api_keys"]["PINECONE_API_KEY"]
pinecone_environment = st.secrets["api_keys"]["PINECONE_ENVIRONMENT"]

# Initialize Pinecone
pc = pinecone.Pinecone(api_key=pinecone_api_key, environment=pinecone_environment)
index_name = "textbook-dataset"
dimension = 384

# Create the index if it doesn't exist
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=dimension,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region=pinecone_environment)
    )

index = pc.Index(index_name)
embedder = SentenceTransformer('all-MiniLM-L6-v2')

# Load spaCy's language model for sentence segmentation
nlp = spacy.load("en_core_web_sm")

def extract_text_from_pdf(pdf_path):
    """Extracts text from a PDF file."""
    text = []
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text.append(page_text)
    return " ".join(text)

def semantic_boundary_detection(text, embedding_model, similarity_threshold=0.7):
    """Detects semantic boundaries by comparing sentence embeddings and cosine similarity."""
    doc = nlp(text)
    sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]

    if len(sentences) < 2:
        return sentences  

    embeddings = embedding_model.encode(sentences)
    similarities = cosine_similarity(embeddings[:-1], embeddings[1:])

    chunks = []
    chunk = [sentences[0]]

    for i, sim in enumerate(similarities.flatten()):
        if i + 1 >= len(sentences):
            break
        if sim < similarity_threshold:
            chunks.append(" ".join(chunk))
            chunk = [sentences[i + 1]]
        else:
            chunk.append(sentences[i + 1])

    if chunk:
        chunks.append(" ".join(chunk))

    return chunks

def topic_refinement(semantic_chunks, num_topics=5):
    """Refines semantic chunks by merging them based on topic coherence."""
    tokenized_chunks = [re.findall(r'\b\w+\b', chunk.lower()) for chunk in semantic_chunks]
    
    if not tokenized_chunks:
        return semantic_chunks

    dictionary = corpora.Dictionary(tokenized_chunks)
    corpus = [dictionary.doc2bow(chunk) for chunk in tokenized_chunks]

    lda_model = LdaModel(corpus, num_topics=num_topics, id2word=dictionary, passes=10)

    topic_distribution = [lda_model.get_document_topics(doc) for doc in corpus]

    refined_chunks = []
    chunk = [semantic_chunks[0]]
    prev_topic = max(topic_distribution[0], key=lambda x: x[1])[0]

    for i in range(1, len(semantic_chunks)):
        current_topic = max(topic_distribution[i], key=lambda x: x[1])[0]

        if current_topic != prev_topic:
            refined_chunks.append(" ".join(chunk))
            chunk = [semantic_chunks[i]]
        else:
            chunk.append(semantic_chunks[i])

        prev_topic = current_topic

    if chunk:
        refined_chunks.append(" ".join(chunk))

    return refined_chunks

def batch_upsert(index, upserts, namespace, batch_size=50):
    """Splits upserts into smaller batches to avoid exceeding Pinecone's 2MB limit."""
    for i in range(0, len(upserts), batch_size):
        batch = upserts[i : i + batch_size]
        index.upsert(batch, namespace=namespace)

def process_pdfs_and_index(pdf_directory):
    """Processes PDFs and indexes meaningful chunks into Pinecone."""
    for file_name in os.listdir(pdf_directory):
        if file_name.endswith('.pdf'):
            pdf_path = os.path.join(pdf_directory, file_name)
            print(f"Processing {file_name}")

            text = extract_text_from_pdf(pdf_path)

            if not text.strip():
                print(f"Skipping {file_name} - No extractable text found.")
                continue

            # Stage 1: Semantic boundary detection
            semantic_chunks = semantic_boundary_detection(text, embedder)

            # Stage 2: Topic-based refinement
            refined_chunks = topic_refinement(semantic_chunks)

            namespace = f"textbook_{file_name.split('.')[0].lower()}"

            # Prepare data for batch upsert
            upserts = [
                (f"{file_name}chunk{idx}", embedder.encode(chunk).tolist(), {
                    "file_name": file_name,
                    "chunk_index": idx,
                    "chunk_preview": chunk[:100],
                    "full_text": chunk
                }) for idx, chunk in enumerate(refined_chunks)
            ]

            # *Use batch upsert instead of one large request*
            if upserts:
                batch_upsert(index, upserts, namespace)

    print("Indexing completed.")

if __name__ == "__main__":
    pdf_directory = "./dataset"  
    process_pdfs_and_index(pdf_directory)