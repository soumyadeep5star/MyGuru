import json
import logging
import os
from datetime import datetime
from typing import Dict, List, Tuple

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)


class SemanticChunkingService:
    """Reusable semantic chunking service with safe fallback behavior."""

    def __init__(
        self,
        embeddings,
        fallback_chunk_size: int = 1200,
        fallback_chunk_overlap: int = 150,
    ):
        self.embeddings = embeddings
        self.fallback_chunk_size = fallback_chunk_size
        self.fallback_chunk_overlap = fallback_chunk_overlap
        self._semantic_splitter = self._build_semantic_splitter()

    def _build_semantic_splitter(self):
        """Try semantic splitter first, fall back if package is unavailable."""
        try:
            from langchain_experimental.text_splitter import SemanticChunker

            logger.info("SemanticChunker initialized successfully")
            return SemanticChunker(
                self.embeddings,
                breakpoint_threshold_type="percentile",
                breakpoint_threshold_amount=85,
            )
        except Exception as exc:
            logger.warning(
                "SemanticChunker unavailable; falling back to RecursiveCharacterTextSplitter. Error: %s",
                exc,
            )
            return None

    def split_text(self, text: str) -> List[str]:
        if not text or not text.strip():
            return []

        if self._semantic_splitter is not None:
            semantic_docs = self._semantic_splitter.create_documents([text])
            return [doc.page_content.strip() for doc in semantic_docs if doc.page_content.strip()]

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.fallback_chunk_size,
            chunk_overlap=self.fallback_chunk_overlap,
        )
        return [chunk.strip() for chunk in splitter.split_text(text) if chunk.strip()]

    def chunk_documents(
        self,
        documents: List[Dict],
        source_name: str,
        force_single_chunk: bool = False,
    ) -> Tuple[List[Document], List[Dict]]:
        langchain_docs: List[Document] = []
        report_entries: List[Dict] = []

        for doc in documents:
            content = doc.get("content", "")
            if force_single_chunk:
                chunks = [content.strip()] if content and content.strip() else []
            else:
                chunks = self.split_text(content)

            report_entries.append(
                {
                    "source": source_name,
                    "doc_id": doc.get("id"),
                    "title": doc.get("title"),
                    "url": doc.get("url"),
                    "type": doc.get("type"),
                    "chunks": len(chunks),
                    "content_chars": len(content or ""),
                }
            )

            for idx, chunk in enumerate(chunks):
                metadata = {
                    "doc_id": doc.get("id"),
                    "title": doc.get("title"),
                    "space": doc.get("space"),
                    "space_key": doc.get("space_key"),
                    "type": doc.get("type"),
                    "url": doc.get("url"),
                    "chunk_index": idx,
                    "source": doc.get("url"),
                }
                langchain_docs.append(Document(page_content=chunk, metadata=metadata))

        return langchain_docs, report_entries

    @staticmethod
    def save_chunk_report(report_entries: List[Dict], output_path: str):
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        payload = {
            "generated_at": datetime.utcnow().isoformat() + "Z",
            "total_documents": len(report_entries),
            "total_chunks": sum(entry.get("chunks", 0) for entry in report_entries),
            "documents": report_entries,
        }

        with open(output_path, "w", encoding="utf-8") as file_obj:
            json.dump(payload, file_obj, indent=2)

        logger.info("Saved chunking report to %s", output_path)
