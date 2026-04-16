"""
Plain Text Parser with automatic encoding detection.
Handles .txt, .md, .csv, .log files.

CO1 Alignment: Document Representation - reading raw text files.
"""

from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class TextParser:
    """Extract text from plain text files with encoding detection."""

    def extract(self, file_path: Path) -> dict:
        """
        Extract text from a plain text file.

        Returns:
            dict with keys: 'text', 'metadata', 'file_path', 'file_type'
        """
        file_path = Path(file_path)
        result = {
            "file_path": str(file_path),
            "file_name": file_path.name,
            "file_type": "text",
            "text": "",
            "metadata": {},
        }

        try:
            # Try UTF-8 first (most common)
            text = self._read_with_encoding(file_path, "utf-8")
            if text is None:
                # Fall back to encoding detection
                text = self._read_with_detection(file_path)
            if text is None:
                # Last resort: latin-1 (never fails)
                text = self._read_with_encoding(file_path, "latin-1")

            result["text"] = text or ""
            result["metadata"] = {
                "size_bytes": file_path.stat().st_size,
            }

        except Exception as e:
            logger.error(f"Error parsing text file {file_path}: {e}")
            result["text"] = ""

        return result

    def _read_with_encoding(self, file_path: Path, encoding: str):
        """Try reading file with specific encoding."""
        try:
            with open(file_path, "r", encoding=encoding) as f:
                return f.read()
        except (UnicodeDecodeError, UnicodeError):
            return None

    def _read_with_detection(self, file_path: Path):
        """Detect encoding and read file."""
        try:
            import chardet

            with open(file_path, "rb") as f:
                raw_data = f.read()
            detected = chardet.detect(raw_data)
            encoding = detected.get("encoding", "utf-8")
            return raw_data.decode(encoding, errors="replace")
        except Exception:
            return None
