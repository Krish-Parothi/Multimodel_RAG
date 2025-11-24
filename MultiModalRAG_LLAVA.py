import os
import io
import fitz #PyMuPDF ke andar ye use hota hai.
import base64
import torch 
import numpy as np
from PIL import Image
import streamlit as st
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity
from transformers import CLIPProcessor, CLIPModel
from langchain_core.prompts import PromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_classic.schema.messages import HumanMessage
from langchain.chat_models import init_chat_model
from langchain_core.documents import Document
from langchain_groq import ChatGroq
from ollama import Client
ollama = Client(host="http://localhost:11434")

### Loading Clip Model so we reuired processor and model
load_dotenv()

## set up the environment
# os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

## Initialize Clip Model

clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
# This model is reponsible for conversion of text and images into embeddings.

clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# CLIP processor isliye use hota hai kyunki CLIP model ko image aur text ek specific standardized format me chahiye hota hai — processor unhe convert karke ready-to-use banata hai.

clip_model.eval()


def embed_image(image_data):
    ''' Embbed image using clip'''
    if isinstance(image_data, str): # if path
        image = Image.open(image_data).convert("RGB")
    else: # If PIL Image
        image = image_data

    input = clip_processor(images=image, return_tensors="pt") # we need to return tensors in pytorch tensors.
    with torch.no_grad():
        features = clip_model.get_image_features(**input)
        # Normalize embeddings to unit vectors
        features = features/features.norm(dim=1, keepdim=True)
        return features.squeeze().numpy()

    
def embed_text(text):
    ''' Embed text using CLIP'''
    inputs = clip_processor(
        text=text,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=77 # Clip's Max token length
    )
    with torch.no_grad():
        features = clip_model.get_text_features(**inputs)
        # Normalize embeddings
        features = features/features.norm(dim=1, keepdim=True)
        return features.squeeze().numpy()




## Process PDF
pdf_path = "Data.pdf"
doc = fitz.open(pdf_path)

# we now create variables for Storage for all documents and embeddings
all_docs = []
all_embeddings = []
image_data_store = {}

# Text Splitter
splitter = RecursiveCharacterTextSplitter(chunk_size=500) 
# RecursiveCharacterTextSplitter large text ko intelligent, meaning-preserving chunks me todta hai taaki embeddings and RAG best perform karein.

doc

for i,page in enumerate(doc): #go inside my doc
    # Proess the text
    text = page.get_text()
    if text.strip():
        # create temporary document for splitting
        temp_doc = Document(page_content=text, metadata={"page": i, "type": "text"})
        # For all the text data, keep meta data type as text only.
        text_chunks = splitter.split_documents([temp_doc])


        for chunk in text_chunks:
            embedding = embed_text(chunk.page_content)
            all_embeddings.append(embedding)
            all_docs.append(chunk)

## process images
    ##Three Important Actions:

    ##Convert PDF image to PIL format
    ##Store as base64 for GPT-4V (which needs base64 images)
    ##Create CLIP embedding for retrieval

    for img_index, img in enumerate(page.get_images(full=True)):
        try:
            xref = img[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            
            # Convert to PIL Image
            pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            
            # Create unique identifier
            image_id = f"page_{i}_img_{img_index}"
            
            # Store image as base64 for later use with GPT-4V
            buffered = io.BytesIO()
            pil_image.save(buffered, format="PNG")
            img_base64 = base64.b64encode(buffered.getvalue()).decode()
            image_data_store[image_id] = img_base64
            
            # Embed image using CLIP
            embedding = embed_image(pil_image)
            all_embeddings.append(embedding)
            
            # Create document for image
            image_doc = Document(
                page_content=f"[Image: {image_id}]",
                metadata={"page": i, "type": "image", "image_id": image_id}
            )
            all_docs.append(image_doc)
            
        except Exception as e:
            print(f"Error processing image {img_index} on page {i}: {e}")
            continue

doc.close()


all_docs

# Create unified FAISS vector store with CLIP embeddings
embeddings_array = np.array(all_embeddings)
embeddings_array


(all_docs,embeddings_array)

# Create custom FAISS index since we have precomputed embeddings
vector_store = FAISS.from_embeddings(
    text_embeddings=[(doc.page_content, emb) for doc, emb in zip(all_docs, embeddings_array)],
    embedding=None,  # We're using precomputed embeddings
    metadatas=[doc.metadata for doc in all_docs]
)
vector_store

# Initialize llava Vision model

# llm = ChatGroq(
#     api_key=os.getenv("GROQ_API_KEY"),
#     model="meta-llama/llama-prompt-guard-2-86m",
#     streaming=True,
#     temperature=0.8
# )
# llm = Ollama(
#      model="llava:7b",
#      temperature=0.8
#  )

# llm

def retrieve_multimodal(query, k=5):
    """Unified retrieval using CLIP embeddings for both text and images."""
    # Embed query using CLIP
    query_embedding = embed_text(query)
    
    # Search in unified vector store
    results = vector_store.similarity_search_by_vector(
        embedding=query_embedding,
        k=k
    )
    
    return results


def create_multimodal_message(query, retrieved_docs):
    content = []

    content.append({
        "type": "text",
        "text": f"Question: {query}\n\nContext:\n"
    })

    text_docs = [doc for doc in retrieved_docs if doc.metadata.get("type") == "text"]
    image_docs = [doc for doc in retrieved_docs if doc.metadata.get("type") == "image"]

    if text_docs:
        text_context = "\n\n".join([
            f"[Page {doc.metadata['page']}]: {doc.page_content}"
            for doc in text_docs
        ])
        content.append({
            "type": "text",
            "text": f"Text excerpts:\n{text_context}\n"
        })

    for doc in image_docs:
        img_id = doc.metadata["image_id"]
        content.append({
            "type": "text",
            "text": f"\n[Image from page {doc.metadata['page']}]: {img_id}\n"
        })

    content.append({
        "type": "text",
        "text": "Please answer the question based on the provided text and attached images."
    })

    return content

def multimodal_pdf_rag_pipeline(query):
    """Main pipeline for multimodal RAG."""
    # Retrieve relevant documents
    context_docs = retrieve_multimodal(query, k=5)
    
    # Create multimodal message
    message = create_multimodal_message(query, context_docs)   # <-- message is LIST now
    
    # Collect images for LLaVA
    images_for_llava = []

    for docu in context_docs:
        if docu.metadata.get("type") == "image":
            img_id = docu.metadata["image_id"]
            b64 = image_data_store[img_id]
            img_bytes = base64.b64decode(b64)
            images_for_llava.append(img_bytes)

    # Build text prompt
    final_prompt = ""

    # FIX: message is a LIST, not message.content
    for part in message:
        if part["type"] == "text":
            final_prompt += part["text"] + "\n"
    
    # Print retrieved context info
    print(f"\nRetrieved {len(context_docs)} documents:")
    for doc in context_docs:
        doc_type = doc.metadata.get("type", "unknown")
        page = doc.metadata.get("page", "?")
        if doc_type == "text":
            preview = doc.page_content[:100] + "..." if len(doc.page_content) > 100 else doc.page_content
            print(f"  - Text from page {page}: {preview}")
        else:
            print(f"  - Image from page {page}")
    print("\n")
    
    # REAL LLaVA CALL
    stream = ollama.generate(
    model="llava:7b",
    prompt=final_prompt,
    images=images_for_llava,
    stream=True
    )

    full = ""
    for chunk in stream:
        if "response" in chunk:
            print(chunk["response"], end="", flush=True)
            full += chunk["response"]

    return full


if __name__ == "__main__":
    # Example queries
    queries = [
        
        "At Page 1 What Does the Graph Tells You About , and which ML Model does that graph is for? Explain that Graph and all its components in a systematic format. And there are three more graphs what does that tells?",
        
        "In Page 2, How many Graphs are there? , Tell Their respective functionlities and explain all labels on the graphs."
        
    ]
    
    for query in queries:
        print(f"\nQuery: {query}")
        print("-" * 50)
        answer = multimodal_pdf_rag_pipeline(query)
        print("Answer:", answer)
        print("=" * 70)
