"""OCR utility module for fallback text extraction from scanned or image-based PDF pages."""

import sys
import os
import tempfile
import subprocess
from pathlib import Path
from typing import List, Optional, Union
import pypdf

from app.core.logging_config import logger

WINDOWS_OCR_SCRIPT = Path(__file__).resolve().parent / "windows_ocr.ps1"


def is_ocr_available() -> bool:
    """Returns True if Windows native OCR or pytesseract is available."""
    if sys.platform == "win32" and WINDOWS_OCR_SCRIPT.is_file():
        return True
    try:
        import pytesseract  # type: ignore
        return True
    except ImportError:
        return False


def ocr_image_file(image_path: Union[str, Path]) -> str:
    """Executes OCR on an image file using Windows native OCR or pytesseract fallback."""
    path = Path(image_path).resolve()
    if not path.is_file():
        logger.warning(f"OCR image target does not exist: {path}")
        return ""

    # 1. Prefer Windows Native OCR on Windows platforms
    if sys.platform == "win32" and WINDOWS_OCR_SCRIPT.is_file():
        try:
            cmd = [
                "powershell",
                "-NoProfile",
                "-NonInteractive",
                "-ExecutionPolicy",
                "Bypass",
                "-File",
                str(WINDOWS_OCR_SCRIPT),
                "-ImagePath",
                str(path),
            ]
            res = subprocess.run(cmd, capture_output=True, text=True, timeout=25, encoding="utf-8", errors="replace")
            if res.returncode == 0 and res.stdout.strip():
                return res.stdout.strip()
            elif res.returncode != 0:
                logger.warning(f"Windows OCR returned non-zero exit code ({res.returncode}): {res.stderr.strip()[:200]}")
        except Exception as e:
            logger.warning(f"Windows OCR invocation failed for '{path.name}': {str(e)}")

    # 2. Secondary fallback: pytesseract if installed
    try:
        import pytesseract  # type: ignore
        from PIL import Image  # type: ignore

        img = Image.open(path)
        text = pytesseract.image_to_string(img)
        if text.strip():
            return text.strip()
    except Exception as e:
        logger.debug(f"Pytesseract fallback unavailable or failed: {str(e)}")

    return ""


def ocr_image_bytes(image_bytes: bytes, suffix: str = ".png") -> str:
    """Writes image bytes to a temporary file and executes OCR."""
    if not image_bytes:
        return ""

    temp_path: Optional[Path] = None
    try:
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(image_bytes)
            temp_path = Path(tmp.name)

        return ocr_image_file(temp_path)
    finally:
        if temp_path and temp_path.is_file():
            try:
                temp_path.unlink()
            except Exception:
                pass


def extract_ocr_text_from_pdf_page(pdf_path: Union[str, Path], page_index: int) -> str:
    """Inspects a specific PDF page for embedded images and runs OCR on them."""
    path = Path(pdf_path)
    if not path.is_file():
        return ""

    try:
        reader = pypdf.PdfReader(str(path))
        if page_index >= len(reader.pages):
            return ""

        page = reader.pages[page_index]
        images = list(page.images)
        if not images:
            logger.info(f"Page {page_index + 1} has no embedded images for OCR.")
            return ""

        logger.info(f"Found {len(images)} image(s) on PDF page {page_index + 1}. Running OCR...")
        extracted_sections: List[str] = []

        for idx, img in enumerate(images):
            # Infer suffix from image name or default to .png
            ext = Path(img.name).suffix.lower() if hasattr(img, "name") and img.name else ".png"
            if not ext:
                ext = ".png"

            text = ocr_image_bytes(img.data, suffix=ext)
            if text.strip():
                extracted_sections.append(text.strip())

        combined_text = "\n\n".join(extracted_sections).strip()
        logger.info(
            f"OCR completed for page {page_index + 1}: extracted {len(combined_text)} characters from {len(images)} image(s)."
        )
        return combined_text

    except Exception as e:
        logger.error(f"Error during PDF page image OCR extraction for '{path.name}' page {page_index + 1}: {str(e)}")
        return ""
