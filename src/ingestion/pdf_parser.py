"""
PDF Parser using PyMuPDF (fitz).
Handles both text-based PDFs and scanned/image PDFs via OCR fallback.

CO1 Alignment: Document Representation - extracting text from PDF documents.
"""

import fitz  # PyMuPDF
from pathlib import Path
from typing import Optional
import logging

logger = logging.getLogger(__name__)


class PDFParser:
    """Extract text from PDF files using PyMuPDF with OCR fallback."""

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
            doc = fitz.open(str(file_path))
            result["metadata"] = {
                "page_count": len(doc),
                "title": doc.metadata.get("title", ""),
                "author": doc.metadata.get("author", ""),
            }

            all_text = []
            for page_num, page in enumerate(doc):
                # Try direct text extraction first
                page_text = page.get_text("text").strip()

                # If page has very little text, try OCR on page images
                if len(page_text) < 20 and self.ocr_engine:
                    page_text = self._ocr_page(page, page_num)

                if page_text:
                    all_text.append(page_text)
                    result["pages"].append({
                        "page_num": page_num + 1,
                        "text": page_text,
                    })

            doc.close()
            result["text"] = "\n\n".join(all_text)

        except Exception as e:
            logger.error(f"Error parsing PDF {file_path}: {e}")
            result["text"] = ""

        return result

    def _ocr_page(self, page, page_num: int) -> str:
        """Extract text from a PDF page using OCR on its images."""
        if not self.ocr_engine:
            return ""

        texts = []
        try:
            image_list = page.get_images(full=True)
            for img_index, img_info in enumerate(image_list):
                xref = img_info[0]
                base_image = page.parent.extract_image(xref)
                if base_image:
                    image_bytes = base_image["image"]
                    ocr_text = self.ocr_engine.extract_from_bytes(image_bytes)
                    if ocr_text:
                        texts.append(ocr_text)
        except Exception as e:
            logger.warning(f"OCR failed on page {page_num + 1}: {e}")

        # Also try rendering the full page as image for OCR
        if not texts:
            try:
                pix = page.get_pixmap(dpi=200)
                image_bytes = pix.tobytes("png")
                ocr_text = self.ocr_engine.extract_from_bytes(image_bytes)
                if ocr_text:
                    texts.append(ocr_text)
            except Exception as e:
                logger.warning(f"Full-page OCR failed on page {page_num + 1}: {e}")

        return "\n".join(texts)
