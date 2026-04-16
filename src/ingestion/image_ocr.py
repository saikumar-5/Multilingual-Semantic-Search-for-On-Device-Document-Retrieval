"""
Image OCR using pytesseract + Pillow.
Extracts text from images (JPG, PNG, AVIF, etc.) and scanned PDF pages.
Supports English, Hindi, and Telugu OCR.

CO1 Alignment: Document Representation - extracting text from image-based documents.
"""

from pathlib import Path
from io import BytesIO
import logging

logger = logging.getLogger(__name__)


class ImageOCR:
    """Extract text from images using Tesseract OCR."""

    def __init__(self, tesseract_cmd: str = None, languages: str = "eng+hin+tel"):
        """
        Args:
            tesseract_cmd: Path to tesseract executable.
            languages: OCR language string (e.g., 'eng+hin+tel').
        """
        import pytesseract

        if tesseract_cmd:
            pytesseract.pytesseract.tesseract_cmd = tesseract_cmd
        self.languages = languages
        self._pytesseract = pytesseract

    def extract(self, file_path: Path) -> dict:
        """
        Extract text from an image file using OCR.

        Returns:
            dict with keys: 'text', 'metadata', 'file_path', 'file_type'
        """
        from PIL import Image

        file_path = Path(file_path)
        result = {
            "file_path": str(file_path),
            "file_name": file_path.name,
            "file_type": "image",
            "text": "",
            "metadata": {},
        }

        try:
            img = Image.open(str(file_path))
            # Preprocess image for better OCR accuracy
            img = self._preprocess_image(img)

            # Run OCR
            ocr_data = self._pytesseract.image_to_data(
                img, lang=self.languages, output_type=self._pytesseract.Output.DICT
            )

            # Extract text with confidence filtering
            words = []
            confidences = []
            for i, word in enumerate(ocr_data["text"]):
                conf = int(ocr_data["conf"][i])
                if word.strip() and conf > 30:  # Filter low-confidence words
                    words.append(word.strip())
                    confidences.append(conf)

            result["text"] = " ".join(words)
            result["metadata"] = {
                "width": img.width,
                "height": img.height,
                "avg_confidence": (
                    sum(confidences) / len(confidences) if confidences else 0
                ),
                "word_count": len(words),
            }

        except Exception as e:
            logger.error(f"Error OCR on image {file_path}: {e}")
            result["text"] = ""

        return result

    def extract_from_bytes(self, image_bytes: bytes) -> str:
        """Extract text from raw image bytes (used for PDF page images)."""
        from PIL import Image

        try:
            img = Image.open(BytesIO(image_bytes))
            img = self._preprocess_image(img)
            text = self._pytesseract.image_to_string(img, lang=self.languages)
            return text.strip()
        except Exception as e:
            logger.warning(f"OCR from bytes failed: {e}")
            return ""

    def _preprocess_image(self, img):
        """Preprocess image for better OCR accuracy."""
        from PIL import Image, ImageEnhance, ImageFilter

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
