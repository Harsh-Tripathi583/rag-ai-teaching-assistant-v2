from src.pdf_loader import PDFLoader
from src.chunker import TextChunker
from src.embedder import Embedder
from src.vector_store import VectorStore
from src.retriever import Retriever


loader = PDFLoader("data/pdfs/sample.pdf")

text = loader.load_pdf()


chunker = TextChunker()

chunks = chunker.create_chunks(text)

print(f"\nTotal Chunks: {len(chunks)}")
for i,chunk in enumerate(chunks):
    print(f"{i+1} :\\n")
    print(f"{chunk}")



""" embedder = Embedder()

embeddings = embedder.create_embeddings(chunks)

print("\nEmbeddings Created Successfully.")


vector_store = VectorStore()

vector_store.create_index(embeddings,chunks)


retriever = Retriever(embedder,vector_store)


query = "Why did Jack Gisburn stop painting?"

results = retriever.retrieve(query,top_k=3)


print("\nTop Retrieved Chunks:\n")


for i, chunk in enumerate(results):

    print(f"\nRESULT {i+1}:\n")

    print(chunk[:500]) """