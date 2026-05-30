import faiss
import numpy as np
import pickle


class VectorStore:

    def __init__(self):

        self.index = None
        self.chunks = None

    def create_index(self, embeddings, chunks):

        dimension = embeddings.shape[1]

        self.index = faiss.IndexFlatIP(dimension)

        embeddings = np.array(embeddings).astype("float32")

        self.index.add(embeddings)

        self.chunks = chunks

        print(f"FAISS index created with {self.index.ntotal} vectors.")

    def search(self, query_embedding, top_k=3):

        query_embedding = np.array([query_embedding]).astype("float32")

        scores, indices = self.index.search(query_embedding,top_k)

        retrieved_chunks = []

        for idx in indices[0]:

            retrieved_chunks.append(self.chunks[idx])

        return retrieved_chunks

    def save_index(self, index_path, chunks_path):

        faiss.write_index(self.index,index_path)

        with open(chunks_path, "wb") as f:

            pickle.dump(self.chunks,f)

        print("FAISS index saved successfully.")

    def load_index(self, index_path, chunks_path):

        self.index = faiss.read_index(index_path)

        with open(chunks_path, "rb") as f:

            self.chunks = pickle.load(f)

        print("FAISS index loaded successfully.")