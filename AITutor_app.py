import streamlit as st
import os
import time
import warnings
import logging

from dotenv import load_dotenv

# LangChain
from langchain_groq import ChatGroq
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import YoutubeLoader
# Loaders
from langchain_community.document_loaders import (
    PyPDFLoader,
    WebBaseLoader,
    YoutubeLoader
)

# Vector DB
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

# Google Search
from langchain_community.utilities import SerpAPIWrapper
from langchain_core.documents import Document

# ------------------ CLEAN LOGS ------------------
warnings.filterwarnings("ignore", category=FutureWarning)
logging.getLogger("transformers").setLevel(logging.ERROR)
os.environ["USER_AGENT"] = "student-rag-app"

# ------------------ LOAD ENV ------------------
load_dotenv()
groq_api_key = st.secrets["GROQ_API_KEY"]
serp_api_key = st.secrets["SERPAPI_API_KEY"]

# ------------------ UI ------------------
st.set_page_config(page_title="PragyanAI Student Tutor", layout="wide")
st.title("🎓 PragyanAI Multi-Source AI Tutor")
st.image("PragyanAI_Transperent.png")
# ------------------ SESSION ------------------
if "vector" not in st.session_state:
    st.session_state.vector = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ------------------ SIDEBAR ------------------
with st.sidebar:
    st.header("📚 Add Learning Sources")

    pdf_files = st.file_uploader("Upload PDFs", type="pdf", accept_multiple_files=True)
    youtube_url = st.text_input("YouTube URL")
    web_url = st.text_input("Website URL")
    google_query = st.text_input("Google Search Topic")

    if st.button("Process All Sources"):

        all_docs = []

        with st.spinner("Processing all sources..."):

            # -------- PDF --------
            if pdf_files:
                for file in pdf_files:
                    with open(file.name, "wb") as f:
                        f.write(file.getbuffer())
                    loader = PyPDFLoader(file.name)
                    docs = loader.load()
                    for d in docs:
                        d.metadata["source"] = "PDF"
                    all_docs.extend(docs)

            # -------- YouTube --------
            if youtube_url:
                yt_loader = YoutubeLoader.from_youtube_url(
                    youtube_url,
                    add_video_info=True
                )
                yt_docs = yt_loader.load()
                for d in yt_docs:
                    d.metadata["source"] = "YouTube"
                all_docs.extend(yt_docs)

            # -------- Web --------
            if web_url:
                web_loader = WebBaseLoader(web_url)
                web_docs = web_loader.load()
                for d in web_docs:
                    d.metadata["source"] = "Web"
                all_docs.extend(web_docs)

            # -------- Google --------
            if google_query:
                search = SerpAPIWrapper(serpapi_api_key=serp_api_key)
                results = search.results(google_query)

                if "organic_results" in results:
                    for r in results["organic_results"]:
                        all_docs.append(
                            Document(
                                page_content=r.get("snippet", ""),
                                metadata={
                                    "source": "Google",
                                    "link": r.get("link", "")
                                }
                            )
                        )

            if not all_docs:
                st.warning("No data sources provided!")
                st.stop()

            # -------- SPLIT --------
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200
            )
            documents = splitter.split_documents(all_docs)

            # -------- EMBEDDINGS --------
            embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            )

            # -------- VECTOR --------
            st.session_state.vector = FAISS.from_documents(documents, embeddings)

            st.success("✅ All sources processed successfully!")

# ------------------ MAIN ------------------
st.header("💬 Ask Anything")

llm = ChatGroq(
    groq_api_key=groq_api_key,
    model_name="llama-3.3-70b-versatile"
)

# ------------------ PROMPT ------------------
prompt = ChatPromptTemplate.from_template("""
You are an AI tutor helping students learn.

Use ONLY the provided context.

<context>
{context}
</context>

Question: {input}

Provide:
1. Simple Explanation
2. Detailed Explanation
3. Key Points
4. Summary

If not found, say: "I don't know based on the provided context."
""")

# ------------------ CHAT HISTORY ------------------
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ------------------ USER INPUT ------------------
if user_query := st.chat_input("Ask your question..."):

    if st.session_state.vector is None:
        st.warning("⚠️ Please process sources first.")
        st.stop()

    with st.chat_message("user"):
        st.markdown(user_query)

    st.session_state.chat_history.append({
        "role": "user",
        "content": user_query
    })

    with st.spinner("Thinking..."):
        start = time.process_time()

        retriever = st.session_state.vector.as_retriever()

        docs = retriever.invoke(user_query)

        if docs:
            context = "\n\n".join([doc.page_content for doc in docs[:5]])

            chain = prompt | llm | StrOutputParser()

            answer = chain.invoke({
                "context": context,
                "input": user_query
            })

            # Collect sources
            sources = set([d.metadata.get("source", "Unknown") for d in docs])
        else:
            answer = "No relevant information found."
            sources = []

        response_time = time.process_time() - start

    with st.chat_message("assistant"):
        st.markdown(answer)

        if sources:
            st.caption(f"📚 Sources used: {', '.join(sources)}")

        st.info(f"⏱ Response time: {response_time:.2f}s")

    st.session_state.chat_history.append({
        "role": "assistant",
        "content": answer
    })
