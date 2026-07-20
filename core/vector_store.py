from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings

from core.config import CHROMA_DIR, EMBEDDING_MODEL, TOP_K, FETCH_K


class VectorStoreManager:
    def __init__(self, persist_directory: str = CHROMA_DIR) -> None:
        self.persist_directory = persist_directory
        self.embeddings = OllamaEmbeddings(
            model=EMBEDDING_MODEL
        )

    def create_and_save(self, chunks: list[Document]) -> Chroma:
        return Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            persist_directory=self.persist_directory,
        )

    def load(self) -> Chroma:
        if not Path(self.persist_directory).exists():
            raise FileNotFoundError(
                f"Vector store not found: {self.persist_directory}"
            )

        return Chroma(
            persist_directory=self.persist_directory,
            embedding_function=self.embeddings,
        )

    def get_retriever(self, k: int = TOP_K):
        return self.load().as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": k,
                "fetch_k": FETCH_K,
            },
        )