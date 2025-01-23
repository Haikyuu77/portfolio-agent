import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document

# Updated imports:
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_community.vectorstores import FAISS

# The new location of HuggingFaceEmbeddings:
from langchain_huggingface.embeddings import HuggingFaceEmbeddings

# Constants
DOCS_PATH = "docs"  # folder containing .pdf and/or .txt files
VECTORSTORE_PATH = "faiss_index"  # folder to save FAISS index
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"  # Hugging Face model for embeddings

def load_documents_from_folder(folder_path):
    """
    Loads .txt and .pdf files from a folder into a list of LangChain Document objects.
    """
    docs = []
    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)
        # Handle .txt
        if file_name.lower().endswith(".txt"):
            loader = TextLoader(file_path, encoding="utf-8")
            file_docs = loader.load()
            docs.extend(file_docs)
        # Handle .pdf
        elif file_name.lower().endswith(".pdf"):
            loader = PyPDFLoader(file_path)
            pdf_docs = loader.load_and_split()
            docs.extend(pdf_docs)
        else:
            print(f"Skipping file {file_name} (unsupported format).")
    return docs

def chunk_documents(documents, chunk_size=1000, chunk_overlap=100):
    """
    Splits the documents into smaller chunks for better retrieval performance.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    chunked_docs = []
    for doc in documents:
        # doc.page_content = text content
        # doc.metadata = metadata (file name, page number, etc.)
        chunks = text_splitter.split_text(doc.page_content)
        for chunk in chunks:
            chunked_docs.append(
                Document(
                    page_content=chunk,
                    metadata=doc.metadata
                )
            )
    return chunked_docs

def create_faiss_index():
    """
    Loads documents from DOCS_PATH, creates embeddings, stores them in FAISS, and saves the index.
    """
    # 1. Load documents
    print(f"Loading documents from '{DOCS_PATH}'...")
    all_docs = load_documents_from_folder(DOCS_PATH)
    print(f"Loaded {len(all_docs)} document(s).")

    # 2. Chunk documents
    print("Splitting documents into smaller chunks...")
    chunked_docs = chunk_documents(all_docs)
    print(f"Created {len(chunked_docs)} chunks from all documents.")

    # 3. Initialize the embedding model from langchain_huggingface
    print(f"Using Hugging Face embedding model: {MODEL_NAME}")
    hf_embeddings = HuggingFaceEmbeddings(model_name=MODEL_NAME)
    print(f"Created embeddings object: {hf_embeddings}")

    # 4. Build the FAISS vector store
    print("Creating FAISS index from documents...")
    vectorstore = FAISS.from_documents(chunked_docs, hf_embeddings)

    # 5. Save the FAISS index to disk
    print(f"Saving FAISS index to '{VECTORSTORE_PATH}'...")
    vectorstore.save_local(VECTORSTORE_PATH)
    print("Index creation completed.")

if __name__ == "__main__":
    create_faiss_index()