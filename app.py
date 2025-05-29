import subprocess
import streamlit as st
from dotenv import load_dotenv
import os

# Load .env file for environment variables (like API keys if needed)
load_dotenv('./../.env')

from langchain_community.document_loaders import PyMuPDFLoader
from langchain_ollama import ChatOllama
from langchain_core.prompts import (
    SystemMessagePromptTemplate, HumanMessagePromptTemplate, ChatPromptTemplate
)
from langchain_core.output_parsers import StrOutputParser

# ---- Load PDFs ----
PDF_ROOT = "D:/ML/Pdf_Chatbot/rag-dataset"
pdfs = []
for root, dirs, files in os.walk(PDF_ROOT):
    for file in files:
        if file.endswith(".pdf"):
            pdfs.append(os.path.join(root, file))

docs = []
for pdf in pdfs:
    try:
        loader = PyMuPDFLoader(pdf)
        temp = loader.load()
        docs.extend(temp)
    except Exception as e:
        st.warning(f"Failed to load {pdf}: {e}")

def format_docs(docs):
    return "\n\n".join([x.page_content for x in docs])

context = format_docs(docs)

# ---- Model Selection ----
BASE_URL = "http://localhost:11434"
def get_local_ollama_models():
    try:
        result = subprocess.run(["ollama", "list"], capture_output=True, text=True, check=True)
        models = []
        lines = result.stdout.strip().splitlines()[1:]  # Skip the header
        for line in lines:
            name = line.split()[0]
            models.append(name)
        return models
    except subprocess.CalledProcessError as e:
        st.error(f"Could not fetch model list: {e}")
        return []
AVAILABLE_MODELS = get_local_ollama_models()

# Set default model to "llama3.2:1b" if available
default_model = "llama3.2:1b"
default_index = AVAILABLE_MODELS.index(default_model) if default_model in AVAILABLE_MODELS else 0

selected_model = st.sidebar.selectbox("Select LLM Model:", AVAILABLE_MODELS, index=default_index)
llm = ChatOllama(base_url=BASE_URL, model=selected_model)


# ---- Streamlit UI ----


st.image(r"D:/ML/Pdf_Chatbot/HKA_LOGO.png", width=700)
st.title("PDF Chatbot: Potential Herbal and Dietary Supplement Toxicities")

project = st.sidebar.radio(
    "Choose a project:",
    [
        "Question Answering from PDF",
        "PDF Document Summarization",
        "Report Generation from PDF"
    ]
)
words = st.sidebar.slider(
    "Number of words (approx.)", 
    min_value=20, max_value=2000, value=100, step=10
)

# ---- Project Logic ----

if project == "Question Answering from PDF":
    st.header("Ask a Question about the PDF(s)")
    user_question = st.text_input("Enter your question:")
    if not context.strip():
        st.error("No text found in the loaded PDFs! Please check your PDF folder and try again.")
    elif user_question:
        system = SystemMessagePromptTemplate.from_template(
            "You are a helpful AI assistant who answers user questions based on the provided context. Do not answer in more than {words} words."
        )
        prompt = """Answer the user's question based on the provided context ONLY! If you do not know the answer, just say "I don't know".
        ### Context:
        {context}

        ### Question:
        {question}

        ### Answer:"""
        prompt = HumanMessagePromptTemplate.from_template(prompt)
        messages = [system, prompt]
        template = ChatPromptTemplate(messages)
        qna_chain = template | llm | StrOutputParser()
        if st.button("Get Answer"):
            with st.spinner("Thinking..."):
                response = qna_chain.invoke({'context': context, 'question': user_question, 'words': words})
            st.markdown("**Answer:**")
            st.write(response)

elif project == "PDF Document Summarization":
    st.header("Summarize the PDF(s)")
    if not context.strip():
        st.error("No text found in the loaded PDFs! Please check your PDF folder and try again.")
    else:
        system = SystemMessagePromptTemplate.from_template(
            "You are helpful AI assistant who works as document summarizer. You must not hallucinate or provide any false information."
        )
        prompt = """Summarize the given context in {words} words.
        ### Context:
        {context}

        ### Summary:"""
        prompt = HumanMessagePromptTemplate.from_template(prompt)
        messages = [system, prompt]
        template = ChatPromptTemplate(messages)
        summary_chain = template | llm | StrOutputParser()
        if st.button("Summarize PDF(s)"):
            with st.spinner("Summarizing..."):
                response = summary_chain.invoke({'context': context, 'words': words})
            st.markdown("**Summary:**")
            st.write(response)

elif project == "Report Generation from PDF":
    st.header("Generate a Detailed Report from PDF(s)")
    if not context.strip():
        st.error("No text found in the loaded PDFs! Please check your PDF folder and try again.")
    else:
        system = SystemMessagePromptTemplate.from_template(
            "You are helpful AI assistant who generates detailed reports in Markdown from the provided context. Be accurate and detailed."
        )
        prompt = """Provide a detailed report from the provided context. Write answer in Markdown (do not hallucinate).
        ### Context:
        {context}

        ### Report (max {words} words):"""
        prompt = HumanMessagePromptTemplate.from_template(prompt)
        messages = [system, prompt]
        template = ChatPromptTemplate(messages)
        report_chain = template | llm | StrOutputParser()
        if st.button("Generate Report"):
            with st.spinner("Generating report..."):
                response = report_chain.invoke({
                    'context': context,
                    'words': words,
                })
            st.markdown("**Report:**")
            st.markdown(response)

# Optional: Show loaded PDFs
with st.expander("Show loaded PDF files"):
    for pdf in pdfs:
        st.write(pdf)

# Optional: Show context stats for debugging
with st.expander("Show context debug info"):
    st.write(f"Number of doc chunks: {len(docs)}")
    st.write(f"Total context length: {len(context)} characters")
    if docs:
        st.write("Sample from first doc chunk:")
        st.write(docs[0].page_content[:300])
