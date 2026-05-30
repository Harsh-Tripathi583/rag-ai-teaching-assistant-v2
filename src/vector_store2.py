import faiss
import numpy as np
import pickle


class VectorStore:

    def __init__(self):

        self.index = None
        self.chunks = None

    def create_index(self, embeddings, chunks):

        embeddings = np.array(embeddings).astype("float32")

        dimension = embeddings.shape[1]

        # Cosine similarity
        faiss.normalize_L2(embeddings)

        # HNSW Index
        self.index = faiss.IndexHNSWFlat(dimension, 32)

        # Add embeddings
        self.index.add(embeddings)

        self.chunks = chunks

        print(f"FAISS index created with {self.index.ntotal} vectors.")

    def search(self, query_embedding, top_k=3):

        query_embedding = np.array([query_embedding]).astype("float32")

        # Cosine similarity
        faiss.normalize_L2(query_embedding)

        scores, indices = self.index.search(
            query_embedding,
            top_k
        )

        results = []

        for score, idx in zip(scores[0], indices[0]):

            # Skip invalid neighbors
            if idx == -1:
                continue

            results.append({

                "chunk": self.chunks[idx],

                "score": float(score)

            })

        return results

    def save_index(self, index_path, chunks_path):

        faiss.write_index(
            self.index,
            index_path
        )

        with open(chunks_path, "wb") as f:

            pickle.dump(self.chunks, f)

        print("FAISS index saved successfully.")

    def load_index(self, index_path, chunks_path):

        self.index = faiss.read_index(index_path)

        with open(chunks_path, "rb") as f:

            self.chunks = pickle.load(f)

        print("FAISS index loaded successfully.")