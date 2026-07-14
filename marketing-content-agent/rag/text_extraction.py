from __future__ import annotations

import io


def extract_text(content: bytes, filename: str) -> str:
    """
    Extract plain text from a document.
    Supports: .txt, .md, .csv, .json, .pdf, .docx
    Falls back to UTF-8 decode with error replacement for unknown types.
    """
    fname = filename.lower()

    if fname.endswith(".pdf"):
        try:
            import pdfminer.high_level as pdfminer
            return pdfminer.extract_text(io.BytesIO(content))
        except ImportError:
            try:
                import pypdf

                reader = pypdf.PdfReader(io.BytesIO(content))
                pages = [page.extract_text() or "" for page in reader.pages]
                return "\n\n".join(pages)
            except ImportError:
                return content.decode("utf-8", errors="replace")

    if fname.endswith(".docx"):
        try:
            import docx

            doc = docx.Document(io.BytesIO(content))
            return "\n\n".join(p.text for p in doc.paragraphs if p.text.strip())
        except ImportError:
            return content.decode("utf-8", errors="replace")

    return content.decode("utf-8", errors="replace")


def chunk_text(text: str, chunk_size: int = 512, chunk_overlap: int = 64) -> list[str]:
    """
    Chunk text using LangChain's RecursiveCharacterTextSplitter.
    Returns a list of non-empty string chunks.
    """
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_text(text)
    return [c.strip() for c in chunks if c.strip()]
