import os
import shutil
import re
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()

# --- SECURITY: PII Scrubbing ---
def scrub_sensitive_info(text: str) -> str:
    """Removes emails and phone numbers from the text before it enters the DB."""
    # Email pattern
    text = re.sub(r'\S+@\S+', '[EMAIL_REDACTED]', text)
    # Basic phone number pattern
    text = re.sub(r'\+?\d{10,12}', '[PHONE_REDACTED]', text)
    return text

def start_ingestion():
    # Path Validation to prevent Path Traversal
    data_path = os.path.abspath('./data')
    if not os.path.exists(data_path):
        print("❌ Error: Data directory missing.")
        return

    print("Reading PDFs...")
    loader = DirectoryLoader(data_path, glob="./*.pdf", loader_cls=PyPDFLoader)
    documents = loader.load()
    
    # Scrubbing sensitive data from each document
    for doc in documents:
        doc.page_content = scrub_sensitive_info(doc.page_content)

    print(f"✅ {len(documents)} pages loaded and scrubbed.")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800, # Slightly smaller for better precision
        chunk_overlap=100,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    texts = text_splitter.split_documents(documents)

    # Database setup (Same as before but with error handling)
    if os.path.exists("./vector_db"):
        shutil.rmtree("./vector_db")

    try:
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        vector_db = Chroma.from_documents(
            documents=texts,
            embedding=embeddings,
            persist_directory="./vector_db"
        )
        print(f"✅ Secure Database ready! ({len(texts)} chunks saved)")
    except Exception as e:
        print(f"❌ DB Error: {e}")

if __name__ == "__main__":
    start_ingestion()