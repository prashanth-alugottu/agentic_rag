import streamlit as st
import os
import db.vector_store as vector_store
import agents.agent as agent
# from utils import query_rewriter

st.set_page_config(page_title="RAG Chatbot", layout="wide")
st.title("🧠 Agentic RAG Chatbot")

if "messages" not in st.session_state: st.session_state.messages = []

# ---------------- File Upload Section -----------------
uploaded_file = st.file_uploader("Upload your PDF", type=["pdf"])

if uploaded_file:
    os.makedirs("docs", exist_ok=True)
    file_path = f"docs/{uploaded_file.name}"

    # prevent duplicate uploading same file
    if not os.path.exists(file_path):
        with open(file_path,"wb") as f: f.write(uploaded_file.getbuffer())
        vector_store.upload_file(uploaded_file.name)
        st.success("📄 Document Added to Vector DB!")
    else:
        print("ℹ File already exists — not reprocessed ✔")

# ---------------- Chat UI -----------------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]): st.markdown(msg["content"])

query = st.chat_input("Ask something from documents...")

if query:
    st.session_state.messages.append({"role":"user","content":query})

    with st.chat_message("user"): st.markdown(query)
    with st.spinner("Thinking...⏳"):

        # improved_query = query_rewriter.rewrite_query(query)
        # print("🔄 Query improved with HF:", improved_query)
        result = agent.retrieve_agent(query)
        answer = result["messages"][0]["content"]

        st.session_state.messages.append({"role":"assistant","content":answer})
        with st.chat_message("assistant"): st.markdown(answer)