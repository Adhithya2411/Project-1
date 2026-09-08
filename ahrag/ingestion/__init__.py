"""Document ingestion: loading, structure-aware chunking, and seeding."""

from .chunker import chunk_document
from .loaders import LoaderError, load_text, supported_extensions
from .pipeline import IngestionResult, IngestionService

__all__ = [
    "IngestionResult",
    "IngestionService",
    "LoaderError",
    "chunk_document",
    "load_text",
    "supported_extensions",
]
