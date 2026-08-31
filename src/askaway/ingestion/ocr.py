import statistics
from pathlib import Path

import numpy as np
import pymupdf

from askaway.models import Page

try:
    import easyocr
except ImportError:
    easyocr = None


_reader = None


def get_reader():
    global _reader

    if easyocr is None:
        raise RuntimeError(
            "OCR dependencies are not installed. "
            'Install them with: pip install -e ".[ocr]"'
        )

    if _reader is None:
        _reader = easyocr.Reader(
            ["ar", "en"],
            gpu=True,
        )

    return _reader


def page_to_image(
    page,
    scale: float = 2.0,
) -> np.ndarray:
    matrix = pymupdf.Matrix(scale, scale)

    pixmap = page.get_pixmap(
        matrix=matrix,
        alpha=False,
    )

    image = np.frombuffer(
        pixmap.samples,
        dtype=np.uint8,
    )

    return image.reshape(
        pixmap.height,
        pixmap.width,
        pixmap.n,
    )


def _box_center(box) -> tuple[float, float]:
    x_values = [point[0] for point in box]
    y_values = [point[1] for point in box]

    return (
        sum(x_values) / len(x_values),
        sum(y_values) / len(y_values),
    )


def _box_height(box) -> float:
    y_values = [point[1] for point in box]
    return max(y_values) - min(y_values)


def _group_into_rows(results: list) -> list[list]:
    if not results:
        return []

    heights = [
        _box_height(box)
        for box, _, _ in results
    ]

    typical_height = statistics.median(heights)
    row_tolerance = max(10.0, typical_height * 0.6)

    ordered = sorted(
        results,
        key=lambda item: _box_center(item[0])[1],
    )

    rows = []

    for item in ordered:
        _, center_y = _box_center(item[0])

        if not rows:
            rows.append([item])
            continue

        previous_row = rows[-1]

        previous_y = sum(
            _box_center(existing[0])[1]
            for existing in previous_row
        ) / len(previous_row)

        if abs(center_y - previous_y) <= row_tolerance:
            previous_row.append(item)
        else:
            rows.append([item])

    return rows


def extract_page_with_ocr(page) -> str:
    reader = get_reader()
    image = page_to_image(page)

    results = reader.readtext(
        image,
        detail=1,
        paragraph=False,
    )

    useful_results = [
        item
        for item in results
        if item[1].strip() and item[2] >= 0.25
    ]

    rows = _group_into_rows(useful_results)

    lines = []

    for row in rows:
        # Arabic pages are read primarily from right to left.
        row.sort(
            key=lambda item: _box_center(item[0])[0],
            reverse=True,
        )

        line = " | ".join(
            text.strip()
            for _, text, _ in row
        )

        if line:
            lines.append(line)

    return "\n".join(lines)




def extract_pdf_with_ocr(
    path: Path,
    document_id: str,
) -> list[Page]:
    pages = []

    with pymupdf.open(path) as document:
        total_pages = len(document)

        for page_index, pdf_page in enumerate(document):
            page_number = page_index + 1

            print(
                f"OCR: {path.name} "
                f"[{page_number}/{total_pages}]"
            )

            text = extract_page_with_ocr(pdf_page)

            pages.append(
                Page(
                    document_id=document_id,
                    filename=path.name,
                    page_number=page_number,
                    text=text,
                )
            )

    return pages