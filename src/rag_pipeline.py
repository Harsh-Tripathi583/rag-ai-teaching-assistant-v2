from src.embedder import Embedder
from src.vector_store import VectorStore
from src.llm import LLM


class RAGPipeline:

    def __init__(self,
                 index_path,
                 chunks_path):

        print("Loading RAG Pipeline...")

        self.embedder = Embedder()

        self.vector_store = VectorStore()

        self.vector_store.load_index(
            index_path,
            chunks_path
        )

        self.llm = LLM()

        print("RAG Pipeline Loaded Successfully.")

    def ask(self,
            query,
            top_k=10,
            threshold=0.65):

        # Query embedding
        query_embedding = self.embedder.create_embeddings(
            [query]
        )[0]


        # Retrieve chunks
        retrieved_chunks = self.vector_store.search(
            query_embedding,
            top_k
        )


        # Filter chunks
        filtered_chunks = []

        for item in retrieved_chunks:

            if item["score"] > threshold:

                filtered_chunks.append(item)


        # Build context
        context = "\n\n".join(

            [
                item["chunk"]
                for item in filtered_chunks
            ]
        )


        # Generate response
        response = self.llm.generate_response(
            query,
            context
        )


        return response