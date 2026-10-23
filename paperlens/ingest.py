"""PDF -> pages -> chunks. Every chunk remembers its paper name and page number."""
from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass
class Chunk:
    paper: str
    page: int  # 1-based, as printed by a PDF viewer
    text: str


def read_pdf(data: bytes) -> list[str]:
    """Return the text of each page."""
    import pymupdf  # PyMuPDF

    with pymupdf.open(stream=data, filetype="pdf") as doc:
        return [page.get_text("text") for page in doc]


def clean(text: str) -> str:
    text = text.replace("\u00ad", "")          # soft hyphens
    text = re.sub(r"-\n(?=\w)", "", text)      # join words split across lines
    text = re.sub(r"\s*\n\s*", " ", text)      # line breaks -> spaces
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def split_text(text: str, size: int = 900, overlap: int = 150) -> list[str]:
    """Pack sentences into chunks of about `size` characters with a small overlap."""
    sentences = re.split(r"(?<=[.!?])\s+", clean(text))
    chunks: list[str] = []
    cur = ""
    for sentence in sentences:
        pieces = (
            [sentence[i : i + size] for i in range(0, len(sentence), size - overlap)]
            if len(sentence) > size
            else [sentence]
        )
        for piece in pieces:
            if cur and len(cur) + len(piece) + 1 > size:
                chunks.append(cur)
                cur = (cur[-overlap:] + " " + piece).strip()
            else:
                cur = (cur + " " + piece).strip()
    if cur:
        chunks.append(cur)
    return [c for c in chunks if len(c) >= 40]


def pdf_to_chunks(data: bytes, paper_name: str, size: int = 500, overlap: int = 100) -> list[Chunk]:
    chunks: list[Chunk] = []
    for page_number, page_text in enumerate(read_pdf(data), start=1):
        for piece in split_text(page_text, size, overlap):
            chunks.append(Chunk(paper=paper_name, page=page_number, text=piece))
    return chunks