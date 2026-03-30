from __future__ import annotations

import io
from dataclasses import dataclass
from pathlib import Path
from typing import List

import cv2
import fitz
import numpy as np
from PIL import Image


@dataclass
class VisualCrop:
    index: int
    title: str
    page_label: str
    image_bytes: bytes


def preprocess_image_for_detection(image_bgr: np.ndarray, deep_scan: bool = False) -> np.ndarray:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

    if deep_scan:
        gray = cv2.fastNlMeansDenoising(gray, None, 15, 7, 21)
        gray = cv2.equalizeHist(gray)
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
    else:
        gray = cv2.GaussianBlur(gray, (3, 3), 0)

    return gray


def _merge_boxes(boxes: List[list[int]], gap: int = 12) -> List[list[int]]:
    if not boxes:
        return []

    changed = True
    merged = boxes[:]

    while changed:
        changed = False
        new_boxes: List[list[int]] = []
        used = [False] * len(merged)

        for i, a in enumerate(merged):
            if used[i]:
                continue
            ax1, ay1, ax2, ay2 = a
            current = a[:]

            for j, b in enumerate(merged):
                if i == j or used[j]:
                    continue
                bx1, by1, bx2, by2 = b

                overlap = not (
                    current[2] + gap < bx1
                    or bx2 + gap < current[0]
                    or current[3] + gap < by1
                    or by2 + gap < current[1]
                )

                if overlap:
                    current = [
                        min(current[0], bx1),
                        min(current[1], by1),
                        max(current[2], bx2),
                        max(current[3], by2),
                    ]
                    used[j] = True
                    changed = True

            used[i] = True
            new_boxes.append(current)

        merged = new_boxes

    return merged


def detect_visual_boxes(image_bgr: np.ndarray, deep_scan: bool = False) -> List[list[int]]:
    gray = preprocess_image_for_detection(image_bgr, deep_scan=deep_scan)

    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    kernel_size = (11, 11) if deep_scan else (9, 9)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
    morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2 if deep_scan else 1)

    contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    h, w = gray.shape
    min_area = max(3500, int((h * w) * (0.004 if deep_scan else 0.006)))
    boxes: List[list[int]] = []

    for c in contours:
        x, y, bw, bh = cv2.boundingRect(c)
        area = bw * bh
        if area < min_area:
            continue
        if bw < 45 or bh < 45:
            continue

        aspect = bw / max(bh, 1)
        if 0.18 <= aspect <= 10:
            boxes.append([x, y, x + bw, y + bh])

    return _merge_boxes(boxes, gap=18 if deep_scan else 12)


def crop_visuals_from_image(image_bgr: np.ndarray, page_label: str, deep_scan: bool = False) -> List[VisualCrop]:
    boxes = detect_visual_boxes(image_bgr, deep_scan=deep_scan)
    visuals: List[VisualCrop] = []

    for i, (x1, y1, x2, y2) in enumerate(boxes, start=1):
        pad = 8 if deep_scan else 5
        x1 = max(0, x1 - pad)
        y1 = max(0, y1 - pad)
        x2 = min(image_bgr.shape[1], x2 + pad)
        y2 = min(image_bgr.shape[0], y2 + pad)

        crop = image_bgr[y1:y2, x1:x2]
        if crop.size == 0:
            continue

        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        buf = io.BytesIO()
        Image.fromarray(crop_rgb).save(buf, format="PNG")

        visuals.append(
            VisualCrop(
                index=i,
                title=f"Detected visual {i}",
                page_label=page_label,
                image_bytes=buf.getvalue(),
            )
        )

    return visuals


def detect_visuals_from_image_bytes(file_bytes: bytes, deep_scan: bool = False) -> List[VisualCrop]:
    arr = np.frombuffer(file_bytes, dtype=np.uint8)
    image_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if image_bgr is None:
        return []
    return crop_visuals_from_image(image_bgr, "Input", deep_scan=deep_scan)


def detect_visuals_from_pdf_bytes(
    file_bytes: bytes,
    max_pages: int = 12,
    zoom: float = 1.8,
    deep_scan: bool = False,
) -> List[VisualCrop]:
    doc = fitz.open(stream=file_bytes, filetype="pdf")
    visuals: List[VisualCrop] = []
    running_index = 1

    try:
        total_pages = min(len(doc), max_pages)
        for page_num in range(total_pages):
            page = doc[page_num]
            pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom))
            png_bytes = pix.tobytes("png")

            arr = np.frombuffer(png_bytes, dtype=np.uint8)
            image_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if image_bgr is None:
                continue

            page_visuals = crop_visuals_from_image(
                image_bgr,
                page_label=f"Page {page_num + 1}",
                deep_scan=deep_scan,
            )

            for item in page_visuals:
                item.index = running_index
                running_index += 1
                visuals.append(item)

        return visuals
    finally:
        doc.close()


def detect_visuals(file_bytes: bytes, filename: str, deep_scan: bool = False) -> List[VisualCrop]:
    suffix = Path(filename).suffix.lower()
    if suffix == ".pdf":
        return detect_visuals_from_pdf_bytes(file_bytes, deep_scan=deep_scan)
    if suffix in [".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"]:
        return detect_visuals_from_image_bytes(file_bytes, deep_scan=deep_scan)
    return []
