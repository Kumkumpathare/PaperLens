"""Ask the LLM to answer using only the retrieved excerpts, with citations."""
from __future__ import annotations

import os

from .ingest import Chunk

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

NOT_FOUND = "I could not find this in the uploaded papers."

SYSTEM_PROMPT = f"""You answer questions about research papers using ONLY the numbered excerpts provided.
Rules:
- Cite every claim with the excerpt number in square brackets, like [1] or [2][3].
- If the excerpts do not contain the answer, reply exactly: "{NOT_FOUND}"
- Do not use outside knowledge and do not guess.
- Keep the answer short and clear."""


def build_context(hits: list[tuple[Chunk, float]]) -> str:
    return "\n\n".join(
        f"[{i}] ({chunk.paper}, page {chunk.page})\n{chunk.text}" for i, (chunk, _) in enumerate(hits, start=1)
    )


def answer(question: str, hits: list[tuple[Chunk, float]]) -> str:
    if not hits:
        return NOT_FOUND

    from groq import Groq

    api_key = os.environ.get("GROQ_API_KEY")
    model = os.environ.get("GROQ_MODEL")
    if not api_key or not model:
        raise RuntimeError("Set GROQ_API_KEY and GROQ_MODEL in your .env file (see .env.example).")

    client = Groq(api_key=api_key)
    response = client.chat.completions.create(
        model=model,
        temperature=0,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Excerpts:\n{build_context(hits)}\n\nQuestion: {question}"},
        ],
    )
    return response.choices[0].message.content.strip()
