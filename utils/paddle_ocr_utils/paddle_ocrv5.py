import json
import os
from typing import Any, Dict, List, Optional


class PaddleOCRV5:
    def __init__(
        self,
        det_model_dir: Optional[str] = None,
        rec_model_dir: Optional[str] = None,
        device: Optional[str] = None,
        use_gpu: Optional[bool] = None,
        **kwargs: Any,
    ) -> None:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        models_dir = os.path.join(base_dir, "models")

        self.det_model_dir = det_model_dir or os.path.join(models_dir, "PP-OCRv5_server_det")
        self.rec_model_dir = rec_model_dir or os.path.join(models_dir, "PP-OCRv5_server_rec")
        self.device = device or self._infer_device(use_gpu)

        self._ocr_kwargs = {
            "text_detection_model_name": "PP-OCRv5_server_det",
            "text_recognition_model_name": "PP-OCRv5_server_rec",
            "text_detection_model_dir": self.det_model_dir,
            "text_recognition_model_dir": self.rec_model_dir,
            "use_doc_orientation_classify": False,
            "use_doc_unwarping": False,
            "use_textline_orientation": False,
        }
        if self.device:
            self._ocr_kwargs["device"] = self.device
        self._ocr_kwargs.update(kwargs)

        self._ocr = None

    @staticmethod
    def _infer_device(use_gpu: Optional[bool]) -> Optional[str]:
        if use_gpu is True:
            return "gpu:0"
        if use_gpu is False:
            return "cpu"
        return None

    @staticmethod
    def _ensure_dir_exists(path: str, label: str) -> None:
        if not os.path.isdir(path):
            raise FileNotFoundError(f"{label} not found: {path}")

    def _build_ocr(self):
        self._ensure_dir_exists(self.det_model_dir, "PaddleOCR det model dir")
        self._ensure_dir_exists(self.rec_model_dir, "PaddleOCR rec model dir")

        try:
            from paddleocr import PaddleOCR
        except ImportError as exc:
            raise ImportError(
                "paddleocr is not installed. Install paddleocr and paddlepaddle first."
            ) from exc

        self._ocr = PaddleOCR(**self._ocr_kwargs)
        return self._ocr

    @property
    def pipeline(self):
        if self._ocr is None:
            return self._build_ocr()
        return self._ocr

    def ocr(self, image: Any, need_detail: bool = False):
        results = self.pipeline.predict(image)
        parsed_results = [self._parse_result(result) for result in results]

        if len(parsed_results) == 1:
            parsed_result = parsed_results[0]
            if need_detail:
                return parsed_result
            return parsed_result["text"]

        if need_detail:
            return parsed_results
        return [parsed_result["text"] for parsed_result in parsed_results]

    def _parse_result(self, result: Any) -> Dict[str, Any]:
        raw_result = self._unwrap_result(result)

        texts = self._ensure_list(raw_result.get("rec_texts") or raw_result.get("texts"))
        scores = self._ensure_list(raw_result.get("rec_scores") or raw_result.get("scores"))
        polygons = self._ensure_list(raw_result.get("dt_polys") or raw_result.get("polys"))
        boxes = self._ensure_list(raw_result.get("rec_boxes") or raw_result.get("boxes"))

        detail_count = max(len(texts), len(scores), len(polygons), len(boxes), 0)
        details: List[Dict[str, Any]] = []

        for index in range(detail_count):
            text = self._pick_value(texts, index)
            text = "" if text is None else str(text).strip()
            if not text:
                continue

            detail: Dict[str, Any] = {"text": text}

            score = self._pick_value(scores, index)
            if score is not None:
                detail["score"] = self._normalize_scalar(score)

            polygon = self._pick_value(polygons, index)
            if polygon is not None:
                detail["polygon"] = self._normalize_value(polygon)

            box = self._pick_value(boxes, index)
            if box is not None:
                detail["box"] = self._normalize_value(box)

            details.append(detail)

        if not details and isinstance(raw_result.get("text"), str):
            details.append({"text": raw_result["text"].strip()})

        text = "\n".join(detail["text"] for detail in details if detail.get("text"))
        return {
            "text": text,
            "details": details,
        }

    @staticmethod
    def _unwrap_result(result: Any) -> Dict[str, Any]:
        data = result

        if hasattr(data, "res"):
            data = getattr(data, "res")
        elif hasattr(data, "json"):
            json_value = getattr(data, "json")
            data = json_value() if callable(json_value) else json_value
        elif hasattr(data, "to_dict"):
            data = data.to_dict()

        if isinstance(data, str):
            try:
                data = json.loads(data)
            except json.JSONDecodeError:
                return {"text": data}

        if isinstance(data, dict) and isinstance(data.get("res"), dict):
            return data["res"]
        if isinstance(data, dict):
            return data
        return {}

    @staticmethod
    def _ensure_list(value: Any) -> List[Any]:
        if value is None:
            return []
        if hasattr(value, "tolist"):
            value = value.tolist()
        if isinstance(value, list):
            return value
        if isinstance(value, tuple):
            return list(value)
        return [value]

    @staticmethod
    def _pick_value(values: List[Any], index: int) -> Any:
        if index >= len(values):
            return None
        return values[index]

    @staticmethod
    def _normalize_scalar(value: Any) -> Any:
        if hasattr(value, "item"):
            value = value.item()
        return value

    @classmethod
    def _normalize_value(cls, value: Any) -> Any:
        if hasattr(value, "tolist"):
            value = value.tolist()
        if isinstance(value, tuple):
            value = list(value)
        if isinstance(value, list):
            return [cls._normalize_value(item) for item in value]
        return cls._normalize_scalar(value)


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        sys.exit('用法: python paddle_ocrv5.py <图片目录>')
    img_dir = sys.argv[1]
    ocr = PaddleOCRV5()
    for img_name in sorted(os.listdir(img_dir)):
        if img_name.startswith('_debug'):
            continue
        img_path = os.path.join(img_dir, img_name)
        result = ocr.ocr(img_path, need_detail=False)
        print(f"Result for {img_name}: {result}")