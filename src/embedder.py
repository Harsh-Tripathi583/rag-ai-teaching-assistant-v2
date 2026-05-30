from sentence_transformers import SentenceTransformer
import numpy as np


class Embedder:

    def __init__(self,
                 model_name="BAAI/bge-base-en-v1.5"):

        self.model_name = model_name

        print("Loading embedding model...")

        self.model = SentenceTransformer(self.model_name)

        print("Embedding model loaded successfully.")

    def create_embeddings(self, chunks):

        embeddings = self.model.encode(
            chunks,
            convert_to_numpy=True,
            normalize_embeddings=True
        )

        return embeddings