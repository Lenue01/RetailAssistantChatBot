
#python3 -m uvicorn main:app --host 0.0.0.0 --port 5000 --reload


from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFacePipeline 
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
import os 

from fastapi import FastAPI, Request
from pydantic import BaseModel
from typing import Optional

# Initialize FastAPI app
app = FastAPI(
    title="Retail Assistant RAG API",
    description="API for a retail assistant providing answers based on process guides.",
    version="1.0.0",
)


# --- RAG SYSTEM INITIALIZATION ---
# This part runs once when the API starts up

print("Initializing RAG system...")
print(f"Current working directory: {os.getcwd()}") # Debug: Check current directory

if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"There are {torch.cuda.device_count()} GPU(s) available.")
    print(f"GPU device name: {torch.cuda.get_device_name(0)}")
else:
    device = torch.device("cpu")
    print("No GPU available, using CPU instead.")

print(f"Using device: {device}")

#Indicate start of embedding model loading
print("Loading embedding model...")
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
print("Embedding model loaded.")

#Indicate start of vectorstore loading
print("Loading FAISS vectorstore...")
# Ensure 'faiss_index' directory is in the correct location relative to where you run main.py
vectorstore = FAISS.load_local("faiss_index/", embedding_model, allow_dangerous_deserialization=True)
print("FAISS vectorstore loaded.")

#Replace "ADD ID TOKEN HERE" with your actual Hugging Face token if needed
model_name = "mistral_model"  # Point to the local directory
print(f"Loading tokenizer from {model_name}...")
tokenizer = AutoTokenizer.from_pretrained(model_name, token="ADD ID TOKEN HERE")
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
if tokenizer.eos_token is None:
    tokenizer.eos_token = tokenizer.unk_token
print("Tokenizer loaded and configured.")

print(f"Loading model from {model_name}...")
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto", token="ADD ID TOKEN HERE")
print("Model loaded.")


print("Creating HuggingFacePipeline...")
llm_pipeline = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=150, 
    temperature=0.7,
    top_p=0.9,
    device_map="auto",
    return_full_text=False
)
print("HuggingFacePipeline created.")

llm = HuggingFacePipeline(pipeline=llm_pipeline)

prompt_template = """You are a retail assistant with knowledge of company process guides. You are to follow the process guides and assist workers with any questions they may ask. Only use information found in the provided documents. You are to assume it is an in-store return unless otherwise told. Respond in 1-2 sentences.


Context:
{context}

Question: {question}
Answer:"""
prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

print("Creating RetrievalQA chain...")
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 2}),
    return_source_documents=True, # Keep this as True for debugging context
    chain_type_kwargs={"prompt": prompt}
)
print("RetrievalQA chain created.")

print("RAG system initialized successfully. API ready to receive requests.")
# --- END RAG SYSTEM INITIALIZATION ---


# Define model for the request body
class QueryRequest(BaseModel):
    query: str


# Define an API endpoint for RAG system
@app.post("/ask")
async def ask_question(request: QueryRequest):
    """
    API endpoint to receive a question and return an answer from the RAG system.
    Expects a JSON payload with a 'query' field.
    """
    query = request.query.strip()
    print(f"\n--- NEW REQUEST RECEIVED ---")
    print(f"Received query: '{query}'")

    if not query:
        print("Error: No query provided.")
        return {"error": "No query provided"}

    try:
        print("Starting RAG chain invocation...")

        
        response = qa_chain.invoke({"query": query})
        print(f"RAG chain invocation complete.")

        # Extract the results
        final_answer = response.get('result', 'No answer found.')
        source_documents = response.get('source_documents', [])

        print("\n--- RETRIEVED CONTEXT ---")
        if source_documents:
            for i, doc in enumerate(source_documents):
                print(f"Document {i+1} (Source: {doc.metadata.get('source', 'N/A')}):")
                # Truncate long content for readability, or print full if needed
                print(doc.page_content[:500] + "..." if len(doc.page_content) > 500 else doc.page_content)
                print("-" * 20)
        else:
            print("No source documents retrieved.")
        print("------------------------")
        
        print(f"Final answer generated: '{final_answer}'")

        return {"answer": final_answer}

    except Exception as e:
        print(f"!!! ERROR DURING REQUEST PROCESSING !!!")
        import traceback
        traceback.print_exc()
        print(f"Error details: {e}")
        return {"error": str(e)}
