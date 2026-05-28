import json
import math
import os
import sqlite3
import uuid
from datetime import datetime
from pathlib import Path

import openai
import streamlit as st
from PyPDF2 import PdfReader

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
DB_PATH = Path("vector_store.db")

st.title("PDF Upload Reader + Vector Store")
st.write("Upload a PDF, embed its text using OpenAI, and save chunks in a vector database.")

chunk_size = st.sidebar.number_input(
    "Chunk size (characters)", min_value=100, max_value=5000, value=1000, step=100
)
chunk_overlap = st.sidebar.number_input(
    "Chunk overlap (characters)", min_value=0, max_value=chunk_size // 2, value=200, step=50
)

uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])
query_text = st.text_input("Search saved vectors", value="")

if not OPENAI_API_KEY:
    st.warning("Set OPENAI_API_KEY in your environment before embedding text.")
else:
    openai.api_key = OPENAI_API_KEY


def init_vector_db(path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(path, check_same_thread=False)
    cursor = connection.cursor()
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS vectors (
            id TEXT PRIMARY KEY,
            source TEXT,
            chunk_index INTEGER,
            page_number INTEGER,
            content TEXT NOT NULL,
            metadata TEXT,
            embedding TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
        """
    )
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_source ON vectors (source)")
    connection.commit()
    return connection


def chunk_text(text: str, size: int, overlap: int) -> list[str]:
    if size <= 0:
        return [text]
    if overlap < 0:
        overlap = 0

    delimiters = ["\n", ".", ","]
    segments = []
    current = []

    for char in text:
        current.append(char)
        if char in delimiters:
            segments.append("".join(current).strip())
            current = []

    if current:
        segments.append("".join(current).strip())

    chunks: list[str] = []
    current_chunk = ""

    for segment in segments:
        if not segment:
            continue

        if len(current_chunk) + len(segment) <= size:
            current_chunk = (current_chunk + " " + segment).strip() if current_chunk else segment
        else:
            if current_chunk:
                chunks.append(current_chunk)
            if len(segment) > size:
                start = 0
                while start < len(segment):
                    end = min(start + size, len(segment))
                    chunks.append(segment[start:end].strip())
                    start = end
                current_chunk = ""
            else:
                current_chunk = segment

    if current_chunk:
        chunks.append(current_chunk)

    if overlap > 0 and len(chunks) > 1:
        overlapped_chunks: list[str] = []
        for i, chunk in enumerate(chunks):
            if i == 0:
                overlapped_chunks.append(chunk)
                continue
            prev = overlapped_chunks[-1]
            overlap_text = prev[-overlap:].strip()
            if overlap_text:
                overlapped_chunks.append((overlap_text + " " + chunk).strip())
            else:
                overlapped_chunks.append(chunk)
        return overlapped_chunks

    return chunks


def create_embeddings(texts: list[str]) -> list[list[float]]:
    response = openai.Embedding.create(
        model="text-embedding-3-small",
        input=texts,
    )
    return [item["embedding"] for item in response["data"]]


def save_vectors(connection: sqlite3.Connection, vectors: list[dict]) -> None:
    cursor = connection.cursor()
    for vector in vectors:
        cursor.execute(
            "INSERT OR REPLACE INTO vectors (id, source, chunk_index, page_number, content, metadata, embedding, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                vector["id"],
                vector["source"],
                vector["chunk_index"],
                vector["page_number"],
                vector["content"],
                json.dumps(vector["metadata"], ensure_ascii=False),
                json.dumps(vector["embedding"]),
                vector["created_at"],
            ),
        )
    connection.commit()


def load_all_vectors(connection: sqlite3.Connection) -> list[dict]:
    cursor = connection.cursor()
    rows = cursor.execute("SELECT id, source, chunk_index, page_number, content, metadata, embedding FROM vectors").fetchall()
    return [
        {
            "id": row[0],
            "source": row[1],
            "chunk_index": row[2],
            "page_number": row[3],
            "content": row[4],
            "metadata": json.loads(row[5]) if row[5] else {},
            "embedding": json.loads(row[6]),
        }
        for row in rows
    ]


def cosine_similarity(a: list[float], b: list[float]) -> float:
    dot_product = sum(x * y for x, y in zip(a, b))
    magnitude_a = math.sqrt(sum(x * x for x in a))
    magnitude_b = math.sqrt(sum(y * y for y in b))
    if magnitude_a == 0 or magnitude_b == 0:
        return 0.0
    return dot_product / (magnitude_a * magnitude_b)


def search_vectors(connection: sqlite3.Connection, query_embedding: list[float], top_n: int = 5) -> list[dict]:
    vectors = load_all_vectors(connection)
    scored = []
    for vector in vectors:
        score = cosine_similarity(query_embedding, vector["embedding"])
        scored.append({"score": score, **vector})

    scored.sort(key=lambda item: item["score"], reverse=True)
    return scored[:top_n]


with st.sidebar:
    st.markdown("## Vector database status")
    db_exists = DB_PATH.exists()
    st.write("Saved vectors:" if db_exists else "No database yet")
    if db_exists:
        conn = init_vector_db(DB_PATH)
        count = conn.execute("SELECT COUNT(*) FROM vectors").fetchone()[0]
        st.write(count)
    else:
        conn = init_vector_db(DB_PATH)
        st.write(0)

if uploaded_file is not None:
    try:
        reader = PdfReader(uploaded_file)
        full_text = []
        for page_num, page in enumerate(reader.pages, start=1):
            page_text = page.extract_text() or ""
            if page_text:
                full_text.append(f"\n\n--- Page {page_num} ---\n\n{page_text}")

        if not full_text:
            st.warning("No readable text found in this PDF.")
        else:
            document_text = "\n\n".join(full_text)
            chunks = chunk_text(document_text, chunk_size, chunk_overlap)
            st.success("PDF uploaded and text extracted successfully.")
            st.write(f"Total characters: {len(document_text)}")
            st.write(f"Total chunks: {len(chunks)}")

            if OPENAI_API_KEY:
                if st.button("Embed and save chunks to vector DB"):
                    with st.spinner("Creating embeddings and saving vectors..."):
                        embeddings = create_embeddings(chunks)
                        document_name = Path(uploaded_file.name).stem
                        vectors = []

                        for index, (chunk, embedding) in enumerate(zip(chunks, embeddings), start=1):
                            vectors.append(
                                {
                                    "id": f"{document_name}-{index}-{uuid.uuid4().hex}",
                                    "source": uploaded_file.name,
                                    "chunk_index": index,
                                    "page_number": None,
                                    "content": chunk,
                                    "metadata": {
                                        "source": uploaded_file.name,
                                        "chunk_size": len(chunk),
                                    },
                                    "embedding": embedding,
                                    "created_at": datetime.utcnow().isoformat() + "Z",
                                }
                            )

                        save_vectors(conn, vectors)
                        st.success(f"Saved {len(vectors)} chunks to {DB_PATH}")

            for index, chunk in enumerate(chunks, start=1):
                st.markdown(f"### Chunk {index}")
                st.write(chunk)
    except Exception as e:
        st.error(f"Could not read the uploaded PDF: {e}")

if query_text:
    if not OPENAI_API_KEY:
        st.warning("Search requires OPENAI_API_KEY.")
    else:
        with st.spinner("Embedding query and searching vector database..."):
            query_embedding = create_embeddings([query_text])[0]
            results = search_vectors(conn, query_embedding, top_n=5)

        if results:
            st.subheader("Top matching chunks")
            for result in results:
                st.write(f"Score: {result['score']:.4f}")
                st.write(f"Source: {result['source']}")
                st.write(result["content"])
                st.markdown("---")
        else:
            st.info("No saved chunks found in the vector database.")

conn.close()
