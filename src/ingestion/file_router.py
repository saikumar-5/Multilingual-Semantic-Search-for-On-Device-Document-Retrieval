"""
File Router - Routes files to the appropriate parser based on extension.
Walks directories recursively and processes all supported file types.

CO1 Alignment: Document Representation - managing multiple document formats.
"""

import os
from pathlib import Path
from typing import List
import logging

from src.config import SUPPORTED_EXTENSIONS, TESSERACT_CMD, OCR_LANGUAGES

logger = logging.getLogger(__name__)


class FileRouter:
    """Route files to correct parsers and manage document ingestion."""

    def __init__(self):
        from .pdf_parser import PDFParser
        from .docx_parser import DocxParser
        from .text_parser import TextParser
        from .excel_parser import ExcelParser
        from .image_ocr import ImageOCR

        # Initialize OCR engine (used by both image and PDF parsers)
        try:
            self.ocr_engine = ImageOCR(
                tesseract_cmd=TESSERACT_CMD, languages=OCR_LANGUAGES
            )
        except Exception as e:
            logger.warning(f"Tesseract not available, OCR disabled: {e}")
            self.ocr_engine = None

        # Initialize parsers
        self.parsers = {
            "pdf": PDFParser(ocr_engine=self.ocr_engine),
            "docx": DocxParser(),
            "text": TextParser(),
            "excel": ExcelParser(),
            "image": self.ocr_engine,
        }

    def get_file_type(self, file_path: Path) -> str:
        """Determine file type from extension."""
        ext = file_path.suffix.lower()
        for file_type, extensions in SUPPORTED_EXTENSIONS.items():
            if ext in extensions:
                return file_type
        return "unknown"

    def process_file(self, file_path: Path) -> dict:
        """
        Process a single file and extract its text.

        Returns:
            dict with extracted text and metadata, or None if unsupported.
        """
        file_path = Path(file_path)
        file_type = self.get_file_type(file_path)

        if file_type == "unknown":
            logger.debug(f"Skipping unsupported file: {file_path}")
            return None

        parser = self.parsers.get(file_type)
        if parser is None:
            logger.debug(f"No parser for file type '{file_type}': {file_path}")
            return None

        try:
            result = parser.extract(file_path)
            if result and result.get("text", "").strip():
                logger.info(
                    f"Extracted {len(result['text'])} chars from {file_path.name}"
                )
                return result
            else:
                logger.warning(f"No text extracted from {file_path.name}")
                return result
        except Exception as e:
            logger.error(f"Failed to process {file_path}: {e}")
            return None

    def scan_directory(self, directory: str) -> List[Path]:
        """
        Recursively scan a directory for supported files.

        Returns:
            List of Path objects for supported files.
        """
        directory = Path(directory)
        supported_files = []

        if not directory.exists():
            logger.error(f"Directory not found: {directory}")
            return supported_files

        for root, _dirs, files in os.walk(directory):
            for filename in files:
                file_path = Path(root) / filename
                if self.get_file_type(file_path) != "unknown":
                    supported_files.append(file_path)

        logger.info(
            f"Found {len(supported_files)} supported files in {directory}"
        )
        return sorted(supported_files)

    def process_directory(self, directory: str, progress_callback=None) -> List[dict]:
        """
        Process all supported files in a directory.

        Args:
            directory: Path to directory to scan.
            progress_callback: Optional callable(current, total, filename) for progress.

        Returns:
            List of document dicts with extracted text.
        """
        files = self.scan_directory(directory)
        documents = []

        for i, file_path in enumerate(files):
            if progress_callback:
                progress_callback(i + 1, len(files), file_path.name)

            doc = self.process_file(file_path)
            if doc and doc.get("text", "").strip():
                doc["doc_id"] = i
                documents.append(doc)

        logger.info(
            f"Successfully processed {len(documents)}/{len(files)} files"
        )
        return documents
