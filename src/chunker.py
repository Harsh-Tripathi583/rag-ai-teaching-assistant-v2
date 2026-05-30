from langchain_text_splitters import RecursiveCharacterTextSplitter

class TextChunker:

    def __init__(self,
                 chunk_size=1000,
                 chunk_overlap=200):

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap
        )

    def create_chunks(self, text):

        chunks = self.text_splitter.split_text(text)

        return chunks