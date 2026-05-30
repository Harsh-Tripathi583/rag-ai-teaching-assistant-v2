class Retriever:

    def __init__(self, embedder, vector_store):

        self.embedder = embedder
        self.vector_store = vector_store

    def retrieve(self, query, top_k=3):

        query_embedding = self.embedder.create_embeddings(
            [query]
        )[0]

        retrieved_chunks = self.vector_store.search(
            query_embedding,
            top_k
        )

        return retrieved_chunks