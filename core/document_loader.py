# core/document_loader.py

import os
import tempfile
import uuid
from typing import List

from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader


class PDFLoader:
    def load_from_streamlit(self, uploaded_file) -> List[Document]:
        if uploaded_file is None:
            return []

        temp_filepath = None
        document_id = str(uuid.uuid4())

        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
                temp_file.write(uploaded_file.getvalue())
                temp_filepath = temp_file.name

            loader = PyPDFLoader(temp_filepath)
            documents = loader.load()

            for doc in documents:
                doc.metadata.update({
                    "filename": uploaded_file.name,
                    "source": uploaded_file.name,
                    "file_type": "pdf",
                    "page_number": doc.metadata.get("page", 0),
                    "document_id": document_id
                })

            return documents

        finally:
            if temp_filepath and os.path.exists(temp_filepath):
                os.remove(temp_filepath)

    def load_from_path(self, file_path: str) -> List[Document]:
        loader = PyPDFLoader(file_path)
        documents = loader.load()

        filename = os.path.basename(file_path)

        for doc in documents:
            doc.metadata["filename"] = filename
            doc.metadata["source"] = file_path
            doc.metadata["file_type"] = "pdf"
            doc.metadata["page_number"] = doc.metadata.get("page", 0)

        return documents
