import streamlit as st
import os
import db.vector_store as vector_store
import agents.agent as agent



st.set_page_config(page_title="RAG Chatbot", layout="wide")
st.title("Agentic RAG Application")


def uplloadingDocs(file : str):
    return vector_store.upload_file(file)


if "messages" not in st.session_state:
    st.session_state.messages = []

if "loading" not in st.session_state:
    st.session_state.loading = False

uploaded_file = st.file_uploader("Upload your document", type=["pdf"])
if uploaded_file:
        # ensure data/ folder exists
        os.makedirs("docs/", exist_ok=True)
        file_path = f"docs/{uploaded_file.name}"
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
            
        uplloadingDocs(uploaded_file.name)
        st.success("File uploaded successfully!")
            
chat_container = st.container()

with chat_container:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            


def load_vector_store():
    return vector_store.get_vector_store()

user_query = st.chat_input("Ask a question about your documents")

if user_query:
    st.session_state.messages.append({
        "role": "user",
        "content": user_query
    })
    
    with chat_container:
        with st.chat_message("user"):
            st.markdown(user_query)
            
    with st.spinner("🔄 Agent is processing your request..."):
        try:
            result = agent.retrieve_agent(user_query)
            answer = result["messages"][-1].content
             # Add assistant response to chat history
            st.session_state.messages.append({
                "role": "assistant",
                "content": answer
            })
            with chat_container:
                with st.chat_message("assistant"):
                    st.markdown(answer)
        except Exception as e:
            st.error(f"An error occurred: {e}")
            if "No Chroma database found" in str(e):
                st.info("No documents have been uploaded yet. Please upload a document to proceed.")
            
    








