
from typing import List, Dict


def split_into_chunks(
    text: str,
    chunk_size: int = 1200,
    overlap: int = 200,
    metadata: Dict | None = None
) -> List[Dict]:
    """
    Split a financial filing into overlapping chunks.

    Each chunk contains:
    - text
    - chunk ID
    - source metadata
    """

    words = text.split()

    chunks = []

    start = 0
    chunk_id = 0

    while start < len(words):

        end = min(
            start + chunk_size,
            len(words)
        )

        chunk_text = " ".join(
            words[start:end]
        )

        chunk_metadata = {
            "chunk_id": chunk_id,
            "start_word": start,
            "end_word": end
        }

        if metadata:
            chunk_metadata.update(
                metadata
            )

        chunks.append({
            "text": chunk_text,
            "metadata": chunk_metadata
        })

        chunk_id += 1

        if end >= len(words):
            break

        start = end - overlap

    return chunks