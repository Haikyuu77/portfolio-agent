# test_retrieval.py

import os
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


def retrieve_chunks(query : str):
    vectorstore_path = "faiss_index"
    model_name = "sentence-transformers/all-MiniLM-L6-v2"

    print(f"Loading FAISS index from '{vectorstore_path}'...")
    hf_embeddings = HuggingFaceEmbeddings(model_name=model_name)
    vectorstore = FAISS.load_local(vectorstore_path, hf_embeddings, allow_dangerous_deserialization=True)

    print(f"\nQuery: {query}")

    # similarity_search returns a list of Document objects with the best matches
    docs = vectorstore.similarity_search(query, k=4)
    for i, doc in enumerate(docs):
        print(f"\n--- Result {i+1} ---")
        print("Content:", doc.page_content)
        print("Metadata:", doc.metadata)

    return docs

def generate_prompt(docs, user_query):
    promt = f"""
    SYSTEM: You are a helpful assistant that answers questions about Arjun’s portfolio.
Use the following text as reference. If you don't find relevant info, say "I don't know."

CONTEXT:
    {docs[0].page_content}
    {docs[1].page_content}
    {docs[2].page_content}
    {docs[3].page_content}

USER QUESTION: {user_query} ? 

ASSISTANT:
"""
    return promt

if __name__ == "__main__":
    test_retrival()