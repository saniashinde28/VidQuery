import os
import chromadb

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_core.documents import Document


CHROMA_DIR = "vector_db"
COLLECTION_NAME = "meeting_transcript"

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "500"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "50"))


def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"}
    )


def build_vector_store(transcript: str) -> Chroma:

    if not transcript.strip():
        raise ValueError("Transcript is empty. Cannot build vector store.")

    print("Building vector store...")

    # 1. Split transcript into chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        add_start_index=True
    )

    docs = splitter.create_documents(
        [transcript],
        metadatas=[{"source": "meeting_transcript"}]
    )

    # 2. Add chunk indices
    for i, doc in enumerate(docs):
        doc.metadata["chunk_index"] = i + 1

    # 3. Create persistent Chroma client
    client = chromadb.PersistentClient(path=CHROMA_DIR)

    # 4. Delete the previous video's collection
    try:
        client.delete_collection(name=COLLECTION_NAME)
        print("Previous collection deleted.")
    except Exception:
        print("No previous collection found.")

    # 5. Insert the new video's chunks
    vector_store = Chroma.from_documents(
        documents=docs,
        embedding=get_embeddings(),
        ids=[f"chunk_{i + 1:04d}" for i in range(len(docs))],
        collection_name=COLLECTION_NAME,
        persist_directory=CHROMA_DIR
    )

    print(f"Stored {len(docs)} chunks successfully.")

    return vector_store


def load_vector_store() -> Chroma:

    embeddings = get_embeddings()

    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=CHROMA_DIR
    )


def get_retriever(vector_store: Chroma, k: int = 4):

    return vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k}
    )
