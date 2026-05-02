"""
Image OCR using Tesseract.
Extracts text from images (JPG, PNG, AVIF, etc.) and scanned PDF pages.
Supports English, Hindi, and Telugu OCR.

CO1 Alignment: Document Representation - extracting text from image-based documents.
"""

from pathlib import Path
from io import BytesIO
import logging
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class ImageOCR:
    """Extract text from images using Tesseract."""

    def __init__(
        self,
        ocr_langs: Optional[List[str]] = None,
        tesseract_config: Optional[str] = None,
    ):
        """
        Args:
            ocr_langs: Language codes (e.g., ['en', 'hi', 'te']).
            tesseract_config: Optional config string for Tesseract.
        """
        self.ocr_langs = ocr_langs or ["en"]
        self._tesseract_config = tesseract_config or ""
        self._tesseract_lang = self._build_lang_string(self.ocr_langs)

        try:
            import pytesseract

            self._pytesseract = pytesseract
        except Exception as e:
            raise RuntimeError(f"Tesseract OCR not available: {e}")

    def extract(self, file_path: Path) -> dict:
        """
        Extract text from an image file using OCR.

        Returns:
            dict with keys: 'text', 'metadata', 'file_path', 'file_type'
        """
        file_path = Path(file_path)
        result = {
            "file_path": str(file_path),
            "file_name": file_path.name,
            "file_type": "image",
            "text": "",
            "metadata": {},
        }

        try:
            from PIL import Image

            img = Image.open(str(file_path))
            img = self._preprocess_image(img)

            text, meta = self._run_ocr(img)
            result["text"] = text
            result["metadata"] = {
                "width": img.width,
                "height": img.height,
                **meta,
            }

        except Exception as e:
            logger.error(f"Error OCR on image {file_path}: {e}")
            result["text"] = ""

        return result

    def extract_from_bytes(self, image_bytes: bytes) -> str:
        """Extract text from raw image bytes (used for PDF page images)."""
        try:
            from PIL import Image

            img = Image.open(BytesIO(image_bytes))
            img = self._preprocess_image(img)
            text, _meta = self._run_ocr(img)
            return text.strip()
        except Exception as e:
            logger.warning(f"OCR from bytes failed: {e}")
            return ""

    def extract_from_image(self, img) -> str:
        """Extract text from a PIL image object."""
        try:
            img = self._preprocess_image(img)
            text, _meta = self._run_ocr(img)
            return text.strip()
        except Exception as e:
            logger.warning(f"OCR from image failed: {e}")
            return ""

    def _run_ocr(self, img) -> Tuple[str, Dict[str, float]]:
        """Run OCR with the configured engine and return text + metadata."""
        return self._run_tesseract(img)

    def _run_tesseract(self, img) -> Tuple[str, Dict[str, float]]:
        if img.mode != "RGB":
            img = img.convert("RGB")

        data = self._pytesseract.image_to_data(
            img,
            lang=self._tesseract_lang,
            config=self._tesseract_config,
            output_type=self._pytesseract.Output.DICT,
        )

        texts = []
        confs = []
        for text, conf in zip(data.get("text", []), data.get("conf", [])):
            if text and text.strip():
                texts.append(text.strip())
            if conf not in (None, "", "-1"):
                try:
                    confs.append(float(conf))
                except ValueError:
                    continue

        avg_conf = sum(confs) / len(confs) if confs else 0.0
        return (
            " ".join(texts),
            {
                "avg_confidence": avg_conf,
                "word_count": len(texts),
                "engine": "tesseract",
            },
        )

    def _build_lang_string(self, langs: List[str]) -> str:
        lang_map = {
            "en": "eng",
            "hi": "hin",
            "te": "tel",
        }
        mapped = [lang_map.get(lang, lang) for lang in langs]
        return "+".join(mapped)

    def _preprocess_image(self, img):
        """Preprocess image for better OCR accuracy."""
        from PIL import ImageEnhance, ImageFilter

        # Convert to RGB if necessary
        if img.mode not in ("RGB", "L"):
            img = img.convert("RGB")

        # Convert to grayscale
        if img.mode != "L":
            img = img.convert("L")

        # Enhance contrast
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(1.5)

        # Slight sharpening
        img = img.filter(ImageFilter.SHARPEN)

        return img
