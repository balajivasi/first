import streamlit as st
from PyPDF2 import PdfReader

st.title("PDF Upload Reader")
st.write("Upload a PDF file and extract its text in configurable chunks.")

chunk_size = st.sidebar.number_input(
    "Chunk size (characters)", min_value=100, max_value=5000, value=1000, step=100
)
chunk_overlap = st.sidebar.number_input(
    "Chunk overlap (characters)", min_value=0, max_value=chunk_size // 2, value=200, step=50
)

uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])


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

    chunks = []
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
                # If a single segment is longer than size, break it by character boundaries.
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
        overlapped_chunks = []
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

            for index, chunk in enumerate(chunks, start=1):
                st.markdown(f"### Chunk {index}")
                st.write(chunk)
    except Exception as e:
        st.error(f"Could not read the uploaded PDF: {e}")
else:
    st.info("Please upload a PDF file to read its text content.")
