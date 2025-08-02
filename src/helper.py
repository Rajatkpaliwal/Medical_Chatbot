import os
from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain.text_splitter import RecursiveCharacterTextSplitter
from typing import List
from langchain.schema import Document
from langchain.embeddings import HuggingFaceEmbeddings

# Extract text from PDF files
def load_pdf_files(data):
    loader = DirectoryLoader(
            path = data,
            glob = "*.pdf",
            loader_cls = PyPDFLoader
        )
    documents = loader.load()
    return documents

def filter_to_minimal_docs(docs: List[Document]) -> List[Document]:
    """
    Given a list of Document objects, return a newlist of Document objects
    containing only 'source' in metadata and the original page_content.
    """
    minimal_docs: list[Document] = []
    for doc in docs:
        src = doc.metadata.get("source")
        minimal_docs.append(
            Document(
                page_content=doc.page_content,
                metadata = {"source": src}
            )
        )
    return minimal_docs

# Split the documents into smaller chunks
def text_split(minimal_docs: list) -> list:
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 500,
        chunk_overlap = 20
    )
    text_chunks = text_splitter.split_documents(minimal_docs)
    return text_chunks

def download_embeddings():
    """
    Download and return the HuggingFace embeddings model.
    """
    model_name = "Sentence-transformers/all-MiniLM-L6-v2"
    embeddings = HuggingFaceEmbeddings(
        model_name = model_name
    )
    return embeddings