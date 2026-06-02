import streamlit as st

from src.rag_pipeline import RAGPipeline


st.set_page_config(
    page_title="RAG V2",
    page_icon="🤖",
    layout="wide"
)


INDEX_PATH = "storage/index.faiss"
CHUNKS_PATH = "storage/chunks.pkl"


rag = RAGPipeline(
    INDEX_PATH,
    CHUNKS_PATH
)


st.title("🤖 RAG V2 AI Assistant")

query = st.text_input(
    "Ask a question"
)


if st.button("Ask"):

    if query.strip() == "":

        st.warning(
            "Please enter a question."
        )

    else:

        with st.spinner(
            "Generating response..."
        ):

            response = rag.ask(query)


        st.subheader(
            "📌 Final Answer"
        )

        st.write(response)