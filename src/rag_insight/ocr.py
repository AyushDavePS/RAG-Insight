"""Optional local CPU OCR for scanned PDFs using PaddleOCR PP-OCRv5."""
import os
from io import BytesIO
from pathlib import Path


class PaddleOcrExtractor:
    """Lazy CPU-only PP-OCRv5 adapter that returns one text string per PDF page."""

    def __init__(self, language="en", render_dpi=200, model="mobile", cache_root=Path("data/models")):
        self.language = language
        self.render_dpi = render_dpi
        self.model = model
        self.model_id = f"PP-OCRv5-{model}"
        self.cache_root = Path(cache_root)
        self._engine = None

    def _load(self):
        if self._engine is not None:
            return self._engine
        self.cache_root.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("PADDLE_HOME", str(self.cache_root / "paddle"))
        os.environ.setdefault("PADDLE_PDX_CACHE_HOME", str(self.cache_root / "paddle-pdx"))
        try:
            from paddleocr import PaddleOCR
        except ImportError as error:
            raise RuntimeError(
                "OCR is enabled but PaddleOCR is unavailable. Install with: python -m pip install -e '.[ocr]'"
            ) from error
        try:
            common_options = {
                "device": "cpu",
                # Paddle 3.x oneDNN can fail on Windows CPU builds for the
                # PP-OCRv5 detector; plain CPU inference is slower but stable.
                "enable_mkldnn": False,
                "use_doc_orientation_classify": False,
                "use_doc_unwarping": False,
                "use_textline_orientation": False,
            }
            if self.language == "en":
                self._engine = PaddleOCR(
                    text_detection_model_name=f"PP-OCRv5_{self.model}_det",
                    text_recognition_model_name="en_PP-OCRv5_mobile_rec",
                    **common_options,
                )
            else:
                self._engine = PaddleOCR(
                    lang=self.language, ocr_version="PP-OCRv5", **common_options
                )
        except Exception as error:
            raise RuntimeError(f"Could not initialize {self.model_id} OCR on CPU: {error}") from error
        return self._engine

    @staticmethod
    def _texts(result):
        if hasattr(result, "json"):
            payload = result.json
            payload = payload() if callable(payload) else payload
        elif isinstance(result, dict):
            payload = result
        else:
            raise RuntimeError("PaddleOCR returned an unsupported result")
        values = payload.get("res", payload).get("rec_texts", [])
        if not isinstance(values, list) or any(not isinstance(value, str) for value in values):
            raise RuntimeError("PaddleOCR returned malformed recognized text")
        return "\n".join(value.strip() for value in values if value.strip())

    def extract_pdf(self, path: Path) -> list[str]:
        try:
            import numpy as np
            import pymupdf
            from PIL import Image
        except ImportError as error:
            raise RuntimeError(
                "OCR is enabled but PyMuPDF/Pillow are unavailable. Install with: python -m pip install -e '.[ocr]'"
            ) from error
        engine = self._load()
        try:
            document = pymupdf.open(path)
            scale = self.render_dpi / 72
            pages = []
            for page in document:
                pixmap = page.get_pixmap(matrix=pymupdf.Matrix(scale, scale), alpha=False)
                image = np.asarray(Image.open(BytesIO(pixmap.tobytes("png"))).convert("RGB"))
                prediction = engine.predict(image)
                if len(prediction) != 1:
                    raise RuntimeError("PaddleOCR did not return exactly one result for a PDF page")
                pages.append(self._texts(prediction[0]))
            return pages
        except RuntimeError:
            raise
        except Exception as error:
            raise RuntimeError(f"OCR failed for {path.name}: {error}") from error
