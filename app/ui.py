# app/ui.py

import streamlit as st
from rag_pipeline import ask_question

st.title("🧠 RAG Chatbot UI")

query = st.text_input("Enter your question:")

if st.button("Ask"):
    if query.strip():
        with st.spinner("Thinking..."):
            answer = ask_question(query)
            st.success(answer)
