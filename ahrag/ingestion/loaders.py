"""Text extraction for ``.txt``, ``.md``, ``.pdf``, and ``.docx`` sources.

PDF and DOCX support degrade gracefully: if the optional parser is missing, a
:class:`LoaderError` names the package to install rather than silently
producing an empty document (which would look like a retrieval failure later).
"""

from __future__ import annotations

import re
from pathlib import Path

_SUPPORTED = {".txt", ".md", ".markdown", ".pdf", ".docx"}


class LoaderError(RuntimeError):
    """Raised when a source file cannot be turned into text."""


def supported_extensions() -> set[str]:
    """Return the file extensions the ingestion pipeline accepts."""
    return set(_SUPPORTED)


def load_text(path: str | Path) -> str:
    """Extract plain text from ``path``.

    Args:
        path: File to read. Extension determines the parser.

    Returns:
        Normalised text with Windows line endings and non-breaking spaces
        collapsed, and runs of 3+ blank lines reduced to 2.

    Raises:
        LoaderError: If the file is missing, the extension is unsupported, the
            required optional parser is absent, or the file yields no text.
    """
    target = Path(path)
    if not target.exists():
        raise LoaderError(f"File not found: {target}")
    suffix = target.suffix.lower()
    if suffix not in _SUPPORTED:
        raise LoaderError(
            f"Unsupported file type {suffix!r}. Supported: {sorted(_SUPPORTED)}"
        )

    if suffix in {".txt", ".md", ".markdown"}:
        raw = _read_plain(target)
    elif suffix == ".pdf":
        raw = _read_pdf(target)
    else:
        raw = _read_docx(target)

    text = _normalise(raw)
    if not text.strip():
        raise LoaderError(f"No extractable text in {target}")
    return text


def _read_plain(path: Path) -> str:
    """Read a UTF-8 text file, falling back to latin-1 on decode failure."""
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return path.read_text(encoding="latin-1", errors="replace")


def _read_pdf(path: Path) -> str:
    """Extract text from a PDF using ``pypdf``, page by page."""
    try:
        from pypdf import PdfReader
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise LoaderError(
            "PDF ingestion requires 'pypdf'. Install it with: pip install pypdf"
        ) from exc
    try:
        reader = PdfReader(str(path))
        pages = [(page.extract_text() or "") for page in reader.pages]
    except Exception as exc:  # pragma: no cover - malformed PDFs
        raise LoaderError(f"Could not parse PDF {path.name}: {exc}") from exc
    return "\n\n".join(pages)


def _read_docx(path: Path) -> str:
    """Extract paragraphs and table cells from a ``.docx`` file."""
    try:
        import docx  # type: ignore[import-untyped]
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise LoaderError(
            "DOCX ingestion requires 'python-docx'. Install: pip install python-docx"
        ) from exc
    try:
        document = docx.Document(str(path))
    except Exception as exc:  # pragma: no cover - malformed DOCX
        raise LoaderError(f"Could not parse DOCX {path.name}: {exc}") from exc

    parts: list[str] = [p.text for p in document.paragraphs if p.text.strip()]
    for table in document.tables:
        for row in table.rows:
            cells = [c.text.strip() for c in row.cells if c.text.strip()]
            if cells:
                parts.append(" | ".join(cells))
    return "\n\n".join(parts)


def _normalise(text: str) -> str:
    """Collapse line-ending and whitespace variation without losing structure."""
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace(" ", " ")
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()
