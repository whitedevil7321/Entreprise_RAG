
from typing import List
import logfire
from langchain_text_splitters import RecursiveCharacterTextSplitter


def chunk_texts(text: str, chunk_size: int = 1500) -> List[str]:
    """
    Split text into smaller chunks for retrieval.
    """
    with logfire.span("Chunking Text", text_length=len(text)):
        if not text.strip():
            return []

        try:
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=200,
                separators=["\n\n", "\n", " ", ""]
            )

            chunks = splitter.split_text(text)

            return chunks

        except Exception as e:
            logfire.warning(
                "Recursive character text splitter failed",
                error=str(e)
            )
            return []
