import json
import os
from dataclasses import dataclass

import cv2
import numpy as np
import tqdm
from openpyxl import Workbook
from tqdm.contrib.logging import logging_redirect_tqdm

from env_setting import ROOT
from AshareData.utils.paddle_ocr_utils import PaddleOCRV5
from AshareData.utils.log_util import get_logger

logger = get_logger("Table_img_to_excel")

SUPPORTED_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


@dataclass(frozen=True)
class CellCrop:
    row_index: int
    start_col: int
    end_col: int
    box: tuple[int, int, int, int]

    @property
    def file_name(self) -> str:
        if self.start_col == self.end_col:
            return f"{self.row_index}-{self.start_col}.png"
        return f"{self.row_index}-col{self.start_col}_{self.end_col}.png"


@dataclass(frozen=True)
class CellImage:
    cell: CellCrop
    image: np.ndarray


class TableImgParse:
    def __init__(self, output_dir: str | None = None):
        self.output_dir = output_dir
        self._ocr: PaddleOCRV5 | None = None
        self.max_cell_side_height = 1000
        self.valid_row_start_x = 10
        self.valid_row_end_margin = 50
        self.valid_row_required_hits = 6
        self.valid_row_skip_after_hit = 50
        self.table_line_max_channel = 80
        self.table_line_max_channel_diff = 20
        self.watermark_bgr = np.array([225, 225, 225], dtype=np.uint8)
        self.watermark_tolerance = 8
        self.watermark_gray_diff = 6
        self.watermark_max_component_area = 400
        self.watermark_min_component_area = 6
        self.watermark_max_component_height = 80
        self.watermark_max_component_width = 120
        self.top_header_max_height = 160
        self.top_header_dark_threshold = 220
        self.top_header_very_dark_threshold = 180
        self.top_header_min_dark_ratio = 0.02
        self.top_header_min_very_dark_ratio = 0.01
        self.initial_red_band_search_limit = 180
        self.initial_red_band_start_tolerance = 10
        self.initial_red_band_min_rows = 3

    @staticmethod
    def _group_consecutive(values: np.ndarray) -> list[tuple[int, int]]:
        if len(values) == 0:
            return []

        groups: list[tuple[int, int]] = []
        start = int(values[0])
        prev = int(values[0])
        for value in values[1:]:
            value = int(value)
            if value == prev + 1:
                prev = value
                continue
            groups.append((start, prev))
            start = value
            prev = value
        groups.append((start, prev))
        return groups

    @staticmethod
    def _merge_nearby_positions(positions: list[int], min_gap: int) -> list[int]:
        if not positions:
            return []

        merged: list[int] = [positions[0]]
        for position in positions[1:]:
            if position - merged[-1] <= min_gap:
                merged[-1] = (merged[-1] + position) // 2
                continue
            merged.append(position)
        return merged

    @staticmethod
    def _ensure_table_side_borders(positions: list[int], max_index: int) -> list[int]:
        if max_index < 0:
            return positions

        normalized = sorted(set(int(position) for position in positions if 0 <= int(position) <= max_index))
        if not normalized:
            return [0, max_index] if max_index > 0 else [0]

        if normalized[0] > 3:
            normalized.insert(0, 0)
        else:
            normalized[0] = 0

        if max_index - normalized[-1] > 3:
            normalized.append(max_index)
        else:
            normalized[-1] = max_index
        return normalized

    def _should_preserve_top_header_band(self, image_bgr: np.ndarray, y_lines: list[int]) -> bool:
        if not y_lines:
            return False

        first_line_y = int(y_lines[0])
        if first_line_y <= 3 or first_line_y > self.top_header_max_height:
            return False

        header_band = image_bgr[:first_line_y, :]
        if header_band.size == 0:
            return False

        gray = cv2.cvtColor(header_band, cv2.COLOR_BGR2GRAY)
        start_x = min(self.valid_row_start_x, gray.shape[1])
        end_x = max(gray.shape[1] - self.valid_row_end_margin, 0)
        if end_x > start_x:
            gray = gray[:, start_x:end_x]
        if gray.size == 0:
            return False

        dark_ratio = float((gray < self.top_header_dark_threshold).mean())
        very_dark_ratio = float((gray < self.top_header_very_dark_threshold).mean())
        return (
            dark_ratio >= self.top_header_min_dark_ratio
            and very_dark_ratio >= self.top_header_min_very_dark_ratio
        )

    @staticmethod
    def _is_full_red_row(row: np.ndarray) -> bool:
        red_mask = (row[:, 0] > 180) & (row[:, 1] < 120) & (row[:, 2] < 120)
        return float(red_mask.mean()) > 0.75

    def _find_initial_red_end(self, image_rgb: np.ndarray) -> int:
        search_limit = min(self.initial_red_band_search_limit, image_rgb.shape[0])
        red_rows = np.array(
            [
                row_index
                for row_index in range(search_limit)
                if self._is_full_red_row(image_rgb[row_index])
            ],
            dtype=np.int32,
        )
        red_groups = self._group_consecutive(red_rows)
        for start, end in red_groups:
            if start > self.initial_red_band_start_tolerance:
                continue
            if end - start + 1 < self.initial_red_band_min_rows:
                continue
            return end + 1
        return 0

    def find_table_start_y(self, image_bgr: np.ndarray) -> int:
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        red_end = self._find_initial_red_end(image_rgb)
        if red_end <= 0:
            return 0

        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

        light_yellow_mask = (
            (image_rgb[:, :, 0] > 230)
            & (image_rgb[:, :, 1] > 210)
            & (image_rgb[:, :, 2] > 160)
            & (image_rgb[:, :, 2] < 235)
        )

        for row_index in range(red_end, image_rgb.shape[0] - 2):
            yellow_ratio = float(light_yellow_mask[row_index].mean())
            if yellow_ratio > 0.95 and float(light_yellow_mask[row_index + 1].mean()) > 0.95:
                return row_index

        for row_index in range(red_end, image_rgb.shape[0]):
            dark_ratio = float((gray[row_index] < 90).mean())
            if dark_ratio > 0.9:
                return row_index

        return red_end if red_end > 0 else 0

    def crop_top_decoration(self, image_bgr: np.ndarray) -> tuple[np.ndarray, int]:
        start_y = self.find_table_start_y(image_bgr)
        if start_y <= 0:
            return image_bgr, 0
        return image_bgr[start_y:, :].copy(), start_y

    @staticmethod
    def _build_line_valid_mask(image_bgr: np.ndarray) -> np.ndarray:
        """与 detect_line 判定标准一致的向量化掩码（H, W），满足条件为1。"""
        img16 = image_bgr.astype(np.int16)
        ch_max = img16.max(axis=2)
        ch_min = img16.min(axis=2)
        ch_spread = ch_max - ch_min

        # 条件1：暗灰色
        is_dark_gray = (ch_max <= 120) & (ch_spread <= 10)

        # 条件2：深色有色调 luminance = (114*B + 587*G + 299*R) / 1000
        luma = (114 * img16[:, :, 0] + 587 * img16[:, :, 1] + 299 * img16[:, :, 2]) // 1000
        is_dark_tinted = (luma <= 90) & (ch_max <= 150)

        # 条件3：特定绿色（与 detect_line 中 is_green 一致）
        is_green = (
            (np.abs(img16[:, :, 0] - 76) <= 10)
            & (np.abs(img16[:, :, 1] - 112) <= 10)
            & (np.abs(img16[:, :, 2] - 38) <= 10)
        )

        return (is_dark_gray | is_dark_tinted | is_green).astype(np.int32)

    @staticmethod
    def _has_run_of(arr1d: np.ndarray, min_run: int) -> bool:
        """arr1d 是一维0/1数组，判断是否存在连续 min_run 个1。"""
        cs = np.cumsum(arr1d)
        if len(cs) < min_run:
            return False
        diff = cs[min_run - 1:] - np.concatenate([[0], cs[:len(cs) - min_run]])
        return bool((diff >= min_run).any())

    def detect_grid_lines(self, image_bgr: np.ndarray) -> tuple[list[int], list[int]]:
        H, W = image_bgr.shape[:2]
        min_run = 50

        valid = self._build_line_valid_mask(image_bgr)  # (H, W)

        # 横线检测：每行中有连续 >= min_run 个有效像素
        cs_h = np.cumsum(valid, axis=1)
        pad_h = np.zeros((H, 1), dtype=np.int32)
        rs_h = cs_h[:, min_run - 1:] - np.hstack([pad_h, cs_h[:, :W - min_run]])
        h_rows = np.where(rs_h.max(axis=1) >= min_run)[0].astype(np.int32)

        # 竖线检测：每列中有连续 >= min_run 个有效像素
        cs_v = np.cumsum(valid, axis=0)
        pad_v = np.zeros((1, W), dtype=np.int32)
        rs_v = cs_v[min_run - 1:, :] - np.vstack([pad_v, cs_v[:H - min_run, :]])
        v_cols = np.where(rs_v.max(axis=0) >= min_run)[0].astype(np.int32)

        y_groups = self._group_consecutive(np.array(h_rows, dtype=np.int32))
        x_groups = self._group_consecutive(np.array(v_cols, dtype=np.int32))
        y_lines = [int((s + e) // 2) for s, e in y_groups]
        x_lines = [int((s + e) // 2) for s, e in x_groups]

        x_lines = self._merge_nearby_positions(x_lines, min_gap=3)
        y_lines = self._merge_nearby_positions(y_lines, min_gap=6)

        x_lines = self._ensure_table_side_borders(x_lines, image_bgr.shape[1] - 1)
        y_lines = self._ensure_table_side_borders(sorted(set(y_lines)), image_bgr.shape[0] - 1)
        if self._should_preserve_top_header_band(image_bgr, y_lines):
            y_lines = [0, *y_lines[1:]] if y_lines and y_lines[0] == 0 else [0, *y_lines]
        return x_lines, y_lines

    @staticmethod
    def _vertical_line_present(image_bgr: np.ndarray, x: int, top: int, bottom: int) -> bool:
        y0 = min(max(top + 2, 0), image_bgr.shape[0])
        y1 = min(max(bottom - 1, 0), image_bgr.shape[0])
        x0 = max(x - 1, 0)
        x1 = min(x + 2, image_bgr.shape[1])
        if y1 <= y0 or x1 <= x0:
            return False
        min_run = max(10, (y1 - y0) // 4)
        for xi in range(x0, x1):
            if TableImgParse.detect_line(image_bgr[y0:y1, xi, :], min_run=min_run):
                return True
        return False

    def extract_cell_boxes(
        self,
        image_bgr: np.ndarray,
        x_lines: list[int],
        y_lines: list[int],
    ) -> list[CellCrop]:
        cell_boxes: list[CellCrop] = []
        base_column_count = len(x_lines) - 1
        if base_column_count <= 0:
            return cell_boxes

        for row_index in range(len(y_lines) - 1):
            top = y_lines[row_index]
            bottom = y_lines[row_index + 1]
            if bottom - top <= 3:
                continue

            force_merge = (bottom - top) < 50
            separator_present = [
                False if force_merge else self._vertical_line_present(image_bgr, x_lines[col_index], top, bottom)
                for col_index in range(1, len(x_lines) - 1)
            ]

            start_col = 0
            for separator_index, has_separator in enumerate(separator_present, start=1):
                if not has_separator:
                    continue
                cell_boxes.append(
                    CellCrop(
                        row_index=row_index,
                        start_col=start_col,
                        end_col=separator_index - 1,
                        box=(x_lines[start_col], top, x_lines[separator_index], bottom),
                    )
                )
                start_col = separator_index

            cell_boxes.append(
                CellCrop(
                    row_index=row_index,
                    start_col=start_col,
                    end_col=base_column_count - 1,
                    box=(x_lines[start_col], top, x_lines[-1], bottom),
                )
            )
        return cell_boxes

    def _build_watermark_mask(self, image_bgr: np.ndarray) -> np.ndarray:
        lower = np.clip(self.watermark_bgr.astype(np.int16) - self.watermark_tolerance, 0, 255).astype(np.uint8)
        upper = np.clip(self.watermark_bgr.astype(np.int16) + self.watermark_tolerance, 0, 255).astype(np.uint8)
        base_mask = cv2.inRange(image_bgr, lower, upper)

        channel_diff = image_bgr.max(axis=2) - image_bgr.min(axis=2)
        low_saturation_mask = (channel_diff <= self.watermark_gray_diff).astype(np.uint8) * 255
        candidate_mask = cv2.bitwise_and(base_mask, low_saturation_mask)
        if cv2.countNonZero(candidate_mask) == 0:
            return candidate_mask

        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        dark_text_mask = (gray < 180).astype(np.uint8) * 255
        dark_text_guard = cv2.dilate(
            dark_text_mask,
            cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)),
            iterations=1,
        )
        candidate_mask = cv2.bitwise_and(candidate_mask, cv2.bitwise_not(dark_text_guard))
        if cv2.countNonZero(candidate_mask) == 0:
            return candidate_mask

        component_count, labels, stats, _ = cv2.connectedComponentsWithStats(candidate_mask, connectivity=8)
        filtered_mask = np.zeros_like(candidate_mask)
        for component_index in range(1, component_count):
            left = stats[component_index, cv2.CC_STAT_LEFT]
            top = stats[component_index, cv2.CC_STAT_TOP]
            width = stats[component_index, cv2.CC_STAT_WIDTH]
            height = stats[component_index, cv2.CC_STAT_HEIGHT]
            area = stats[component_index, cv2.CC_STAT_AREA]

            if area < self.watermark_min_component_area or area > self.watermark_max_component_area:
                continue
            if width > self.watermark_max_component_width or height > self.watermark_max_component_height:
                continue

            component_mask = (labels == component_index)
            density = area / max(width * height, 1)
            if density > 0.6:
                continue

            roi = gray[max(top - 1, 0):min(top + height + 1, gray.shape[0]), max(left - 1, 0):min(left + width + 1, gray.shape[1])]
            if roi.size == 0:
                continue
            bright_ratio = float((roi > 205).mean())
            if bright_ratio < 0.7:
                continue

            filtered_mask[component_mask] = 255
        return filtered_mask

    def remove_watermark(self, image_bgr: np.ndarray) -> np.ndarray:
        mask = self._build_watermark_mask(image_bgr)
        if cv2.countNonZero(mask) == 0:
            return image_bgr

        cleaned = image_bgr.copy()
        white_background = np.full_like(cleaned, 255)
        cleaned[mask > 0] = white_background[mask > 0]
        return cleaned

    @staticmethod
    def _read_image(img_path: str) -> np.ndarray | None:
        return cv2.imread(img_path)

    def _iter_supported_images(self, img_dir: str) -> list[tuple[str, str]]:
        return [
            (img_name, os.path.join(img_dir, img_name))
            for img_name in sorted(os.listdir(img_dir))
            if os.path.isfile(os.path.join(img_dir, img_name)) and self._is_supported_image_file(img_name)
        ]

    def _get_valid_row_slice(self, row: np.ndarray) -> np.ndarray:
        row_width = row.shape[0]
        start_x = min(self.valid_row_start_x, row_width)
        end_x = max(row_width - self.valid_row_end_margin, 0)
        if end_x <= start_x:
            return row
        return row[start_x:end_x]

    def _extract_cell_images(
        self,
        image_bgr: np.ndarray,
        cell_boxes: list[CellCrop],
    ) -> list[CellImage]:
        cell_images: list[CellImage] = []
        for cell in cell_boxes:
            left, top, right, bottom = cell.box
            crop = image_bgr[top + 1:bottom, left + 1:right]
            if crop.size == 0:
                continue
            cell_images.append(CellImage(cell=cell, image=self.remove_watermark(crop)))
        return cell_images

    @staticmethod
    def detect_line(
        pixels: np.ndarray,
        min_run: int = 50,
        max_gray: int = 120,
        max_spread: int = 10,
        max_step: int = 10,
        max_dark_luma: int = 90,
        max_dark_channel: int = 150,
    ) -> bool:
        """Detect a visually dark line in a 1-D strip of BGR pixels.

        A valid line segment requires >= min_run consecutive pixels where:
          1. Pixel is dark gray, or visually dark after luminance fallback
          2. Adjacent coherence  : max absolute per-channel diff to previous pixel <= max_step

        When condition 2 breaks (color jumps), a fresh run starts from the current pixel
        (provided it still satisfies condition 1).

        Usage::
          detect_line(image_bgr[y, :, :])   # horizontal scan at row y
          detect_line(image_bgr[:, x, :])   # vertical scan at column x
        """
        run = 0
        prev: np.ndarray | None = None
        for pixel in pixels:
            p16 = pixel.astype(np.int16)
            channel_max = int(p16.max())
            channel_min = int(p16.min())
            is_dark_gray = channel_max <= max_gray and (channel_max - channel_min) <= max_spread
            luminance = (114 * int(p16[0]) + 587 * int(p16[1]) + 299 * int(p16[2])) // 1000
            is_dark_tinted = luminance <= max_dark_luma and channel_max <= max_dark_channel
            is_green = (
                abs(int(p16[0]) - 76) <= 10
                and abs(int(p16[1]) - 112) <= 10
                and abs(int(p16[2]) - 38) <= 10
            )
            if not is_dark_gray and not is_dark_tinted and not is_green:
                run = 0
                prev = None
                continue
            if prev is not None and int(np.abs(p16 - prev).max()) > max_step:
                run = 1
                prev = p16
                continue
            run += 1
            prev = p16
            if run >= min_run:
                return True
        return False

    def _is_valid_black_row(self, row: np.ndarray) -> bool:
        row_slice = self._get_valid_row_slice(row)
        if row_slice.size == 0:
            return False
        return self.detect_line(row_slice)

    def _has_enough_valid_rows(self, image_bgr: np.ndarray) -> bool:
        hit_count = 0
        row_index = 0
        total_rows = image_bgr.shape[0]

        while row_index < total_rows:
            row = image_bgr[row_index]
            if self._is_valid_black_row(row):
                hit_count += 1
                if hit_count >= self.valid_row_required_hits:
                    return True
                row_index += self.valid_row_skip_after_hit
                continue
            row_index += 1
        return False

    def _parse_image_layout_from_bgr(self, image_bgr: np.ndarray, img_path: str) -> dict[str, object]:
        cropped_image, crop_top = self.crop_top_decoration(image_bgr)
        x_lines, y_lines = self.detect_grid_lines(cropped_image)
        cell_boxes = self.extract_cell_boxes(cropped_image, x_lines, y_lines)
        cell_images = self._extract_cell_images(cropped_image, cell_boxes)
        return {
            "image_path": img_path,
            "original_height": int(image_bgr.shape[0]),
            "cropped_image": cropped_image,
            "crop_top": crop_top,
            "x_lines": x_lines,
            "y_lines": y_lines,
            "cell_boxes": cell_boxes,
            "cell_images": cell_images,
            "cell_count": len(cell_images),
        }

    def _parse_image_layout(self, img_path: str) -> dict[str, object]:
        image_bgr = self._read_image(img_path)
        if image_bgr is None:
            raise FileNotFoundError(f"Unable to read image: {img_path}")
        return self._parse_image_layout_from_bgr(image_bgr, img_path)

    @staticmethod
    def _is_valid_layout(parsed_image: dict[str, object]) -> bool:
        x_lines = parsed_image["x_lines"]
        y_lines = parsed_image["y_lines"]
        cell_count = parsed_image["cell_count"]
        original_height = parsed_image["original_height"]
        return len(x_lines) > 3 and len(y_lines) > 3 and cell_count > 4 and original_height > 1000

    @staticmethod
    def _clear_output_dir(output_dir: str) -> None:
        if not os.path.isdir(output_dir):
            return
        for file_name in os.listdir(output_dir):
            file_path = os.path.join(output_dir, file_name)
            if os.path.isfile(file_path):
                os.remove(file_path)

    def _get_ocr(self) -> PaddleOCRV5:
        if self._ocr is None:
            self._ocr = PaddleOCRV5()
        return self._ocr

    @staticmethod
    def _normalize_ocr_text(value: object) -> str:
        if isinstance(value, list):
            return "".join(TableImgParse._normalize_ocr_text(item) for item in value).strip()
        text = "" if value is None else str(value)
        return "".join(part.strip() for part in text.splitlines() if part.strip()).strip()

    def _has_oversized_cell_image(self, cell_images: list[CellImage]) -> bool:
        for cell_image in cell_images:
            height = cell_image.image.shape[0]
            if height > self.max_cell_side_height:
                return True
        return False

    def _build_content_rows(self, cell_images: list[CellImage]) -> list[list[str]]:
        ocr = self._get_ocr()
        row_map: dict[int, list[tuple[int, str]]] = {}
        ordered_cells = sorted(
            cell_images,
            key=lambda item: (item.cell.row_index, item.cell.start_col, item.cell.end_col),
        )

        for cell_image in ordered_cells:
            text = self._normalize_ocr_text(ocr.ocr(cell_image.image, need_detail=False))
            row_map.setdefault(cell_image.cell.row_index, []).append((cell_image.cell.start_col, text))

        return [
            [text for _, text in sorted(row_cells, key=lambda item: item[0])]
            for _, row_cells in sorted(row_map.items(), key=lambda item: item[0])
        ]

    def save_cell_images(
        self,
        image_bgr: np.ndarray,
        output_dir: str,
        cell_boxes: list[CellCrop],
    ) -> None:
        os.makedirs(output_dir, exist_ok=True)
        for cell_image in self._extract_cell_images(image_bgr, cell_boxes):
            cv2.imwrite(os.path.join(output_dir, cell_image.cell.file_name), cell_image.image)

    @staticmethod
    def save_debug_overlay(
        image_bgr: np.ndarray,
        output_dir: str,
        x_lines: list[int],
        y_lines: list[int],
        cell_boxes: list[CellCrop],
    ) -> None:
        debug_image = image_bgr.copy()
        for x in x_lines:
            cv2.line(debug_image, (x, 0), (x, image_bgr.shape[0] - 1), (255, 0, 0), 1)
        for y in y_lines:
            cv2.line(debug_image, (0, y), (image_bgr.shape[1] - 1, y), (0, 255, 0), 1)
        for cell in cell_boxes:
            left, top, right, bottom = cell.box
            cv2.rectangle(debug_image, (left, top), (right, bottom), (0, 0, 255), 1)
        cv2.imwrite(os.path.join(output_dir, "_debug_grid.png"), debug_image)

    @staticmethod
    def build_default_output_dir(image_path: str) -> str:
        image_name = os.path.splitext(os.path.basename(image_path))[0]
        return os.path.join(os.path.dirname(image_path), "table_img_test", image_name)

    @staticmethod
    def _is_supported_image_file(file_name: str) -> bool:
        return os.path.splitext(file_name)[1].lower() in SUPPORTED_IMAGE_EXTENSIONS

    @staticmethod
    def _build_img_dir_result(
        invalid_imgs: list[str],
        valid_imgs: list[str] | None = None,
        except_imgs: list[str] | None = None,
        content_rows: list[list[str]] | None = None,
    ) -> dict[str, object]:
        result: dict[str, object] = {
            "invalid_imgs": invalid_imgs,
            "except_imgs": except_imgs or [],
        }
        if valid_imgs is not None:
            result["valid_imgs"] = valid_imgs
        if content_rows is not None:
            result["content_rows"] = content_rows
        return result

    def _parse_candidate_image(
        self,
        img_path: str,
        image_bgr: np.ndarray | None,
    ) -> tuple[str, dict[str, object] | None]:
        if image_bgr is None:
            return "invalid", None

        parsed_image = self._parse_image_layout_from_bgr(image_bgr, img_path)
        if parsed_image is None or not self._is_valid_layout(parsed_image):
            return "invalid", None
        if self._has_oversized_cell_image(parsed_image["cell_images"]):
            return "except", None
        return "valid", parsed_image

    def extract_table_cells(
        self,
        image_path: str,
        output_dir: str | None = None,
        clear_output_dir: bool = True,
    ) -> dict[str, object]:
        parsed_image = self._parse_image_layout(image_path)
        cropped_image = parsed_image["cropped_image"]
        x_lines = parsed_image["x_lines"]
        y_lines = parsed_image["y_lines"]
        cell_boxes = parsed_image["cell_boxes"]

        final_output_dir = output_dir or self.output_dir or self.build_default_output_dir(image_path)
        os.makedirs(final_output_dir, exist_ok=True)
        if clear_output_dir:
            self._clear_output_dir(final_output_dir)
        self.save_cell_images(cropped_image, final_output_dir, cell_boxes)
        self.save_debug_overlay(cropped_image, final_output_dir, x_lines, y_lines, cell_boxes)

        return {
            "image_path": image_path,
            "output_dir": final_output_dir,
            "crop_top": parsed_image["crop_top"],
            "x_lines": x_lines,
            "y_lines": y_lines,
            "cell_count": parsed_image["cell_count"],
        }

    def get_valid_img(self, img_dir: str) -> dict[str, list[str]]:
        valid_imgs: list[str] = []
        invalid_imgs: list[str] = []

        for img_name, img_path in self._iter_supported_images(img_dir):
            image_bgr = self._read_image(img_path)
            if image_bgr is None:
                invalid_imgs.append(img_name)
                continue
            if not self._has_enough_valid_rows(image_bgr):
                invalid_imgs.append(img_name)
                continue
            valid_imgs.append(img_name)

        return {
            "valid_imgs": valid_imgs,
            "invalid_imgs": invalid_imgs,
        }

    def _collect_valid_layout_imgs(self, img_dir: str) -> dict[str, object]:
        valid_img_result = self.get_valid_img(img_dir)
        candidate_valid_imgs = valid_img_result["valid_imgs"]
        invalid_imgs = list(valid_img_result["invalid_imgs"])
        valid_imgs: list[str] = []
        except_imgs: list[str] = []
        parsed_valid_images: dict[str, dict[str, object]] = {}

        for img_name in candidate_valid_imgs:
            img_path = os.path.join(img_dir, img_name)
            status, parsed_image = self._parse_candidate_image(
                img_path,
                self._read_image(img_path),
            )
            if status == "invalid":
                invalid_imgs.append(img_name)
                continue
            if status == "except":
                except_imgs.append(img_name)
                continue

            valid_imgs.append(img_name)
            if parsed_image is not None:
                parsed_valid_images[img_name] = parsed_image

        return {
            "valid_imgs": valid_imgs,
            "invalid_imgs": invalid_imgs,
            "except_imgs": except_imgs,
            "parsed_valid_images": parsed_valid_images,
        }

    def parse_img_dir(self, img_dir: str) -> dict[str, object]:
        classify_result = self._collect_valid_layout_imgs(img_dir)
        valid_imgs = classify_result["valid_imgs"]
        invalid_imgs = classify_result["invalid_imgs"]
        except_imgs = classify_result["except_imgs"]
        parsed_valid_images = classify_result["parsed_valid_images"]
        content_rows: list[list[str]] = []
        if except_imgs:
            return self._build_img_dir_result(
                invalid_imgs=invalid_imgs,
                except_imgs=except_imgs,
                content_rows=[],
            )

        for img_name in valid_imgs:
            parsed_image = parsed_valid_images[img_name]
            cell_images = parsed_image["cell_images"]
            content_rows.extend(self._build_content_rows(cell_images))

        return self._build_img_dir_result(
            invalid_imgs=invalid_imgs,
            except_imgs=except_imgs,
            content_rows=content_rows,
        )

    def parse(self, img_dir: str) -> dict[str, object]:
        return self.parse_img_dir(img_dir)

    def crop_image(self, img_path: str, test: bool = False) -> dict[str, object] | None:
        parsed_image = self._parse_image_layout(img_path)
        cropped_image = parsed_image["cropped_image"]
        if not test:
            return parsed_image

        output_dir = self.output_dir or self.build_default_output_dir(img_path)
        os.makedirs(output_dir, exist_ok=True)
        self._clear_output_dir(output_dir)
        cv2.imwrite(os.path.join(output_dir, "_cropped.png"), cropped_image)
        self.extract_table_cells(img_path, output_dir, clear_output_dir=False)
        return None


def _list_img_dirs(base_image_dir: str) -> list[str]:
    return [
        os.path.join(base_image_dir, name)
        for name in sorted(os.listdir(base_image_dir))
        if os.path.isdir(os.path.join(base_image_dir, name))
    ]


def _write_json_file(file_path: str, data: dict[str, object]) -> None:
    with open(file_path, "w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)


def json_to_excel(json_file: str, excel_file: str | None = None) -> str:
    with open(json_file, "r", encoding="utf-8") as file:
        data = json.load(file)

    def to_sheet_title(raw_name: object, used_titles: set[str]) -> str:
        title = str(raw_name or "未命名板块").strip() or "未命名板块"
        invalid_chars = set('[]:*?/\\')
        title = "".join("_" if char in invalid_chars else char for char in title)
        title = title[:31] or "Sheet"

        if title not in used_titles:
            used_titles.add(title)
            return title

        index = 2
        while True:
            suffix = f"_{index}"
            candidate = f"{title[:31 - len(suffix)]}{suffix}"
            if candidate not in used_titles:
                used_titles.add(candidate)
                return candidate
            index += 1

    def normalize_rows(rows: object) -> list[dict[str, object]]:
        if rows is None:
            return []
        if isinstance(rows, list):
            normalized_rows: list[dict[str, object]] = []
            for row in rows:
                if isinstance(row, dict):
                    normalized_rows.append(row)
                    continue
                if isinstance(row, (list, tuple)):
                    normalized_rows.append(
                        {f"col_{index + 1}": value for index, value in enumerate(row)}
                    )
                    continue
                normalized_rows.append({"value": row})
            return normalized_rows
        if isinstance(rows, dict):
            candidate_keys = ("rows", "data", "items", "list", "content_rows")
            for key in candidate_keys:
                value = rows.get(key)
                if isinstance(value, list):
                    return normalize_rows(value)
            return [rows]
        return [{"value": rows}]

    def to_cell_value(value: object) -> object:
        if value is None:
            return ""
        if isinstance(value, (dict, list)):
            return json.dumps(value, ensure_ascii=False)
        return value

    def group_rows_by_block(json_data: object) -> dict[str, list[dict[str, object]]]:
        block_field_names = ("板块", "板块名称", "block", "block_name", "concept", "概念")

        if isinstance(json_data, dict):
            grouped_rows: dict[str, list[dict[str, object]]] = {}
            top_level_list_keys = ("data", "rows", "items", "list")
            for key in top_level_list_keys:
                value = json_data.get(key)
                if isinstance(value, list):
                    return group_rows_by_block(value)

            for block_name, block_rows in json_data.items():
                grouped_rows[str(block_name)] = normalize_rows(block_rows)
            return grouped_rows

        if isinstance(json_data, list):
            grouped_rows: dict[str, list[dict[str, object]]] = {}
            for row in normalize_rows(json_data):
                block_name = "未分类板块"
                for field_name in block_field_names:
                    field_value = row.get(field_name)
                    if field_value not in (None, ""):
                        block_name = str(field_value)
                        break
                grouped_rows.setdefault(block_name, []).append(row)
            return grouped_rows

        raise ValueError(f"Unsupported json structure: {type(json_data).__name__}")

    grouped_data = group_rows_by_block(data)
    if not grouped_data:
        raise ValueError(f"No sheet data found in json file: {json_file}")

    final_excel_file = excel_file or f"{os.path.splitext(json_file)[0]}.xlsx"
    workbook = Workbook()
    default_sheet = workbook.active
    workbook.remove(default_sheet)
    used_titles: set[str] = set()

    for block_name, rows in grouped_data.items():
        sheet = workbook.create_sheet(title=to_sheet_title(block_name, used_titles))
        headers: list[str] = []
        seen_headers: set[str] = set()
        for row in rows:
            for key in row.keys():
                key_str = str(key)
                if key_str in seen_headers:
                    continue
                seen_headers.add(key_str)
                headers.append(key_str)

        if not headers:
            headers = ["value"]

        sheet.append(headers)
        for row in rows:
            sheet.append([to_cell_value(row.get(header, "")) for header in headers])

        for column_cells in sheet.columns:
            column_values = ["" if cell.value is None else str(cell.value) for cell in column_cells]
            max_length = max((len(value) for value in column_values), default=0)
            sheet.column_dimensions[column_cells[0].column_letter].width = min(max(max_length + 2, 10), 60)

    workbook.save(final_excel_file)
    return final_excel_file

def zhangting_reason_process(json_file: str) -> str:
    import csv
    import re

    with open(json_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    content_rows = data.get("content_rows", [])
    if not content_rows:
        return ""

    date = os.path.basename(os.path.dirname(json_file))

    result_rows: list[list[str]] = []
    current_conception = ""
    current_descs: list[str] = []
    state = "expect_conception"

    for row in content_rows:
        if len(row) == 1:
            text = row[0].strip()
            if state == "expect_conception":
                current_conception = re.sub(r'\*\d+$', '', text).strip()
                current_descs = []
                state = "in_group"
            else:
                current_descs.append(text)
        elif len(row) >= 2:
            state = "expect_conception"
            first = row[0].strip()
            match = re.match(r'^(.+?)(\d{6})$', first)
            if match:
                name = match.group(1)
                code = match.group(2)
            else:
                name = first
                code = ""
            descs = "\n".join(current_descs)
            if len(row) == 7:
                _, _, amount, last_uplimit_time, uplimit_days, sub_conception, uplimit_reason = row
            else:
                amount = ""
                last_uplimit_time = ""
                uplimit_days = ""
                sub_conception = ""
                uplimit_reason = ""
                print(f"Warning: unexpected row format in {json_file}: {row}")
            result_rows.append(
                [
                    date,
                    code,
                    name,
                    current_conception,
                    descs,
                    amount.strip(),
                    last_uplimit_time.strip(),
                    uplimit_days.strip(),
                    sub_conception.strip(),
                    uplimit_reason.strip(),
                ]
            )

    csv_file = os.path.join(os.path.dirname(json_file), "uplimit_flatten.csv")
    with open(csv_file, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "date",
            "code",
            "name",
            "conception",
            "descs",
            "amount",
            "last_uplimit_time",
            "uplimit_days",
            "sub_conception",
            "uplimit_reason",
        ])
        writer.writerows(result_rows)

    return csv_file


def zhangting_reason_process_all() -> None:
    base_image_dir = os.path.join(
        ROOT,
        "AshareData/dataset/scrap_data/zhangtingimgs/",
    )
    img_dirs = _list_img_dirs(base_image_dir)
    processed = 0
    skipped = 0

    for img_dir in tqdm.tqdm(img_dirs, desc="Flatten uplimit.json"):
        uplimit_file = os.path.join(img_dir, "uplimit.json")
        if not os.path.exists(uplimit_file):
            skipped += 1
            continue
        csv_file = zhangting_reason_process(uplimit_file)
        if csv_file:
            processed += 1
        else:
            skipped += 1

    logger.info(f"Flatten done: processed={processed}, skipped={skipped}")


def debug_valid_img_dirs(save_crops: bool = False) -> None:
    base_image_dir = os.path.join(
        ROOT,
        "AshareData/dataset/scrap_data/zhangtingimgs/",
    )
    parser = TableImgParse(output_dir=None)
    img_dirs = _list_img_dirs(base_image_dir)

    for img_dir in tqdm.tqdm(img_dirs, desc="Debug valid img dirs"):
        if not img_dir.endswith("2026-04-08"):
            continue
        print("process: ", img_dir)
        classify_result = parser._collect_valid_layout_imgs(img_dir)
        valid_imgs = classify_result["valid_imgs"]
        invalid_imgs = classify_result["invalid_imgs"]
        except_imgs = classify_result["except_imgs"]

        result = parser._build_img_dir_result(
            invalid_imgs=invalid_imgs,
            valid_imgs=valid_imgs,
            except_imgs=except_imgs,
        )
        output_path = os.path.join(img_dir, "uplimit_img.json")
        _write_json_file(output_path, result)

        if save_crops:
            for img_name in valid_imgs:
                crop_result = parser.extract_table_cells(os.path.join(img_dir, img_name))
                print(f"  crops saved -> {crop_result['output_dir']}")

def main() -> None:
    BASE_IMAGE_DIR = os.path.join(
        ROOT,
        "AshareData/dataset/scrap_data/zhangtingimgs/"
    )

    extractor = TableImgParse(output_dir=None)
    saved_results: list[str] = []
    skipped_results: list[str] = []
    img_dirs = _list_img_dirs(BASE_IMAGE_DIR)

    with tqdm.tqdm(total=len(img_dirs), desc="Processing img dirs") as pbar, logging_redirect_tqdm():
        for img_dir in img_dirs:
            uplimit_file = os.path.join(img_dir, "uplimit.json")
            if os.path.exists(uplimit_file):
                skipped_results.append(uplimit_file)
                logger.info(f"Skipped existing result: {uplimit_file}")
                pbar.update(1)
                continue
            try:
                result = extractor.parse_img_dir(img_dir)
                _write_json_file(uplimit_file, result)
                saved_results.append(uplimit_file)
                logger.info(f"Processed result: {uplimit_file}")
            except Exception as e:
                logger.error(f"Error processing {img_dir}: {e}")
                skipped_results.append(uplimit_file)
            pbar.update(1)

    result = {
        "saved_results": saved_results,
        "skipped_results": skipped_results,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    # process zhangting reason
    zhangting_reason_process_all()

if __name__ == "__main__":
    main()
    #debug_valid_img_dirs(save_crops=True)