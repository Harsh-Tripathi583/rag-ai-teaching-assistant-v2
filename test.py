from src.pdf_loader import PDFLoader
from src.chunker import TextChunker
from src.embedder import Embedder


loader = PDFLoader("data/pdfs/sample.pdf")

text = loader.load_pdf()


chunker = TextChunker()

chunks = chunker.create_chunks(text)

print(f"\nTotal Chunks: {len(chunks)}")


embedder = Embedder()

embeddings = embedder.create_embeddings(chunks)

print("\nEmbeddings Created Successfully.")

print("\nEmbedding Shape:")

print(embeddings.shape)


print("\nFirst Embedding Vector:")

print(embeddings[0])