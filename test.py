from src.pdf_loader import PDFLoader
from src.chunker import TextChunker
from src.embedder import Embedder
from src.vector_store import VectorStore
from src.retriever import Retriever
from src.llm import LLM


loader = PDFLoader("data/pdfs/sample.pdf")

text = loader.load_pdf()


chunker = TextChunker()

chunks = chunker.create_chunks(text)

print(f"\nTotal Chunks: {len(chunks)}")


embedder = Embedder()

embeddings = embedder.create_embeddings(chunks)
print(embeddings)
print("\nEmbeddings Created Successfully.")


vector_store = VectorStore()

vector_store.create_index(
    embeddings,
    chunks
)


# retriever = Retriever(
#     embedder,
#     vector_store
# )


llm = LLM()


query = "why did gisburn stopped painting?"

print(query)

query_embedding = embedder.create_embeddings([query])[0]

retrieved_chunks = vector_store.search(query_embedding, 10)

print("\nRetrieved Chunks:\n")

filtered_chunks = []

for item in retrieved_chunks:

    print(item['score'])
    print(item['chunk'][:100])

    if item['score'] > 0.65:
        filtered_chunks.append(item)

context = "\n\n".join(
    [f"Chunk {i+1}:\n{chunk}"
     for i, chunk in enumerate(retrieved_chunks)]
)

response = llm.generate_response(
    query,
    context
)

print("\nFINAL ANSWER:\n")

print(response)