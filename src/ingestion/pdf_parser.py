"""
PDF Parser using PDFium rendering with OCR fallback.
Handles both text-based PDFs and scanned/image PDFs via OCR fallback.

CO1 Alignment: Document Representation - extracting text from PDF documents.
"""

import pypdfium2 as pdfium
from pathlib import Path
from typing import Optional
import logging

logger = logging.getLogger(__name__)


class PDFParser:
    """Extract text from PDF files using PDFium with OCR fallback."""

    def __init__(self, ocr_engine=None):
        """
        Args:
            ocr_engine: Optional ImageOCR instance for scanned PDF pages.
        """
        self.ocr_engine = ocr_engine

    def extract(self, file_path: Path) -> dict:
        """
        Extract text from a PDF file.

        Returns:
            dict with keys: 'text', 'pages', 'metadata', 'file_path', 'file_type'
        """
        file_path = Path(file_path)
        result = {
            "file_path": str(file_path),
            "file_name": file_path.name,
            "file_type": "pdf",
            "text": "",
            "pages": [],
            "metadata": {},
        }

        try:
            doc = pdfium.PdfDocument(str(file_path))
            result["metadata"] = {
                "page_count": len(doc),
                "title": "",
                "author": "",
            }

            all_text = []
            for page_num in range(len(doc)):
                page = doc[page_num]
                page_text = ""

                try:
                    text_page = page.get_textpage()
                    page_text = text_page.get_text_range().strip()
                    text_page.close()
                except Exception as e:
                    logger.debug(f"PDF text extraction failed on page {page_num + 1}: {e}")

                if len(page_text) < 20 and self.ocr_engine:
                    page_text = self._ocr_page(page, page_num)

                if page_text:
                    all_text.append(page_text)
                    result["pages"].append({
                        "page_num": page_num + 1,
                        "text": page_text,
                    })

                page.close()

            doc.close()
            result["text"] = "\n\n".join(all_text)

        except Exception as e:
            logger.error(f"Error parsing PDF {file_path}: {e}")
            result["text"] = ""

        return result

    def _ocr_page(self, page, page_num: int) -> str:
        """Extract text from a PDF page using OCR on a rendered image."""
        if not self.ocr_engine:
            return ""

        try:
            pil_image = page.render(scale=2.0).to_pil()
            ocr_text = self.ocr_engine.extract_from_image(pil_image)
            return ocr_text
        except Exception as e:
            logger.warning(f"Full-page OCR failed on page {page_num + 1}: {e}")
            return ""
