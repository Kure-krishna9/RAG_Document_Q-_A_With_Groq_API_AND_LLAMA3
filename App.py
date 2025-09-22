
import streamlit as st
import os
from dotenv import load_dotenv
import time

from langchain_groq import ChatGroq
from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain

# Load environment variables
load_dotenv()
groq_api_key = os.getenv("Groq_key")  # ✅ Make sure .env has: GROQ_API_KEY=your_key

# Initialize LLM
llm = ChatGroq(groq_api_key=groq_api_key, model_name="llama-3.1-8b-instant")

# Prompt template
prompt = ChatPromptTemplate.from_template("""
Answer the question based on the provided context only.
Please provide the most accurate response based on the question.

<context>
{context}
</context>

Question: {input}
""")

# Embedding and Vector Store creation
def create_vectors_embeddings():
    if "vectors" not in st.session_state:
        try:
            # Load PDFs
            data_path = os.path.join(os.getcwd(), "data")  # 👈 dynamic path
            loader = PyPDFDirectoryLoader(data_path)
            raw_docs = loader.load()
            if not raw_docs:
                st.warning("⚠️ No documents found in the 'data/' folder.")
                return

            # Split documents
            splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
            split_docs = splitter.split_documents(raw_docs[:10])  # limit to 10 for speed

            # Embed with HuggingFace
            embeddings = HuggingFaceEmbeddings()
            vectordb = FAISS.from_documents(split_docs, embeddings)

            # Save in session state
            st.session_state.embeddings = embeddings
            st.session_state.vectors = vectordb
            st.session_state.docs = split_docs

            st.success("✅ Vector database created from documents.")
        except Exception as e:
            st.error(f"❌ Error creating embeddings: {e}")

# Streamlit UI
st.set_page_config(page_title="📄 RAG Chatbot with Groq + LLaMA3")
st.title("📄 RAG Document Q&A with Groq + LLaMA3")
st.write("Ask a question based on the research papers in the `data/` folder.")

# Button to trigger embedding creation
if st.button("📁 Document Embeddings"):
    create_vectors_embeddings()

# User query
user_prompt = st.text_input("🔍 Enter your query:")

# RAG flow
if user_prompt:
    if "vectors" not in st.session_state:
        st.warning("⚠️ Please create the vector database first by clicking '📁 Document Embeddings'.")
    else:
        try:
            document_chain = create_stuff_documents_chain(llm, prompt)
            retriever = st.session_state.vectors.as_retriever()
            retrieval_chain = create_retrieval_chain(retriever, document_chain)

            start_time = time.process_time()
            response = retrieval_chain.invoke({"input": user_prompt})
            duration = time.process_time() - start_time

            st.markdown(f"⏱️ **Response time:** {duration:.2f} seconds")
            st.write("🤖 **Answer:**")
            st.success(response.get("answer", "No answer returned."))

            with st.expander("📚 Document Chunks Used for Answer"):
                for i, doc in enumerate(response.get("context", [])):
                    st.markdown(f"**Chunk {i+1}:**")
                    st.write(doc.page_content)

        except Exception as e:
            st.error(f"❌ Error during retrieval: {e}")
