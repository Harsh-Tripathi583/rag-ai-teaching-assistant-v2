import os

from src.pdf_loader import PDFLoader
from src.chunker import TextChunker
from src.embedder import Embedder
from src.vector_store import VectorStore
from src.llm import LLM


PDF_FOLDER = "data/pdfs"

INDEX_PATH = "storage/index.faiss"

CHUNKS_PATH = "storage/chunks.pkl"


vector_store = VectorStore()

embedder = Embedder()

llm = LLM()


# Load existing index if already saved
if os.path.exists(INDEX_PATH):

    vector_store.load_index(
        INDEX_PATH,
        CHUNKS_PATH
    )

else:

    all_text = ""

    # Load all PDFs
    for file_name in os.listdir(PDF_FOLDER):

        if file_name.endswith(".pdf"):

            pdf_path = os.path.join(
                PDF_FOLDER,
                file_name
            )

            print(f"\nLoading: {file_name}")

            loader = PDFLoader(pdf_path)

            text = loader.load_pdf()

            all_text += text + "\n"


    print("\nAll PDFs Loaded Successfully.")


    # Chunking
    chunker = TextChunker()

    chunks = chunker.create_chunks(all_text)

    print(f"\nTotal Chunks: {len(chunks)}")


    # Embeddings
    embeddings = embedder.create_embeddings(chunks)

    print("\nEmbeddings Created Successfully.")

    print("\nEmbedding Shape:")

    print(embeddings.shape)


    # Create FAISS index
    vector_store.create_index(
        embeddings,
        chunks
    )


    # Save FAISS index + chunks
    vector_store.save_index(
        INDEX_PATH,
        CHUNKS_PATH
    )


# Chat loop
while True:

    query = input("\nAsk Question (type 'exit' to quit): ")

    if query.lower() == "exit":

        break


    query_embedding = embedder.create_embeddings(
        [query]
    )[0]


    retrieved_chunks = vector_store.search(
        query_embedding,
        top_k=10
    )


    print("\nRetrieved Chunks:\n")


    filtered_chunks = []


    for item in retrieved_chunks:

        print("\nScore:", item["score"])

        print(item["chunk"][:200])

        # Similarity threshold
        if item["score"] > 0.65:

            filtered_chunks.append(item)


    context = "\n\n".join(

        [
            item["chunk"]
            for item in filtered_chunks
        ]
    )


    response = llm.generate_response(
        query,
        context
    )


    print("\nFINAL ANSWER:\n")

    print(response)