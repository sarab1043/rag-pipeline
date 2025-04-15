from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain.chains import RetrievalQA
import streamlit as st

@st.cache_resource
def load_qa_pipeline():
    embedding = OllamaEmbeddings(model="gemma:2b", base_url="http://ollama:11434")
    vectorstore = Chroma(persist_directory="./chroma-store", embedding_function=embedding)
    llm = OllamaLLM(model="gemma:2b", base_url="http://ollama:11434")
    return RetrievalQA.from_chain_type(llm=llm, retriever=vectorstore.as_retriever())

qa_pipeline = load_qa_pipeline()

def ask_question(query):
    response = qa_pipeline.run(query)
    return response
