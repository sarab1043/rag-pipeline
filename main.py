# startup.py
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import TextLoader
from langchain_ollama import OllamaEmbeddings

def build_vectorstore():
    loader = TextLoader("data/data.txt")
    docs = loader.load()
    embedding = OllamaEmbeddings(model="gemma:2b", base_url="http://ollama:11434")
    Chroma.from_documents(docs, embedding, persist_directory="./chroma-store")

if __name__ == "__main__":
    build_vectorstore()
