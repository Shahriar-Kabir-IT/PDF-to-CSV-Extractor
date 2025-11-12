import base64
import io
import json
import os
import re
from typing import List, Optional, Tuple, Dict, Any

import fitz  # PyMuPDF
import pandas as pd

try:
    # OpenAI official SDK
    from openai import OpenAI
except Exception:
    OpenAI = None  # type: ignore

try:
    import requests
except Exception:
    requests = None  # type: ignore


def normalize_amount(value: Any) -> str:
    """Normalize European-formatted numeric strings like '2.531,05' -> '2531.05'.
    Returns string to keep consistent CSV typing without guessing decimals for non-numbers.
    """
    if value is None:
        return ""
    s = str(value).strip()
    if not s:
        return ""
    # Replace thousands separators and comma decimal
    s = s.replace(".", "").replace(",", ".")
    # Keep only digits and decimal point
    cleaned = []
    dot_seen = False
    for ch in s:
        if ch.isdigit():
            cleaned.append(ch)
        elif ch == "." and not dot_seen:
            cleaned.append(ch)
            dot_seen = True
    out = "".join(cleaned)
    return out


def _detect_table_rect(page: "fitz.Page") -> Optional[fitz.Rect]:
    """Detect table area using header keywords. Returns a rectangle or None."""
    header_variants = [
        "Our Document",
        "Your Invoice No.",
        "Reference",
        "Gross amount",
    ]
    rects = []
    for kw in header_variants:
        try:
            hits = page.search_for(kw)
            if hits:
                rects.append(hits[0])
        except Exception:
            continue
    if not rects:
        return None
    # Horizontal coverage based on headers
    x0 = min(r.x0 for r in rects) - 10
    x1 = max(r.x1 for r in rects) + 10
    # Vertical: start slightly above headers, end above footer or near bottom
    y0 = min(r.y0 for r in rects) - 20
    y0 = max(y0, 0)
    # Try to find 'Sum total' to cap bottom
    try:
        sum_hits = page.search_for("Sum total")
    except Exception:
        sum_hits = []
    if sum_hits:
        y1 = sum_hits[0].y1 + 20
    else:
        y1 = min(page.rect.y1, (min(r.y0 for r in rects) + (page.rect.y1 - min(r.y0 for r in rects)) * 0.6))
        # Ensure at least half page height below headers
        y1 = max(y1, min(r.y0 for r in rects) + (page.rect.y1 - min(r.y0 for r in rects)) * 0.5)
        y1 = min(y1, page.rect.y1 - 20)
    return fitz.Rect(x0, y0, x1, y1)


def crop_pdf_to_images(
    pdf_bytes: bytes,
    bbox: Optional[Tuple[float, float, float, float]] = None,
    scale: float = 2.0,
) -> List[bytes]:
    """Return list of PNG bytes for each page, optionally cropped to bbox or auto-detected table area.
    bbox is (x0, y0, x1, y1). If None, auto-detect table area via header keywords; fallback to full page.
    """
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    images: List[bytes] = []
    matrix = fitz.Matrix(scale, scale)
    for page_index in range(len(doc)):
        page = doc.load_page(page_index)
        rect: fitz.Rect
        if bbox:
            rect = fitz.Rect(*bbox)
        else:
            detected = _detect_table_rect(page)
            rect = detected if detected else page.rect
        pm = page.get_pixmap(clip=rect, matrix=matrix)
        images.append(pm.tobytes(output="png"))
    return images


def _parse_json_from_text(text: str) -> Any:
    """Attempt to parse JSON from LLM text. Falls back to extracting the first JSON array/object substring."""
    try:
        return json.loads(text)
    except Exception:
        # Try to salvage array/object
        start_obj = text.find("{")
        start_arr = text.find("[")
        start = min(x for x in [start_obj, start_arr] if x != -1) if (start_obj != -1 or start_arr != -1) else -1
        if start == -1:
            raise
        end_obj = text.rfind("}")
        end_arr = text.rfind("]")
        end = max(end_obj, end_arr)
        if end == -1:
            raise
        snippet = text[start : end + 1]
    return json.loads(snippet)


def _openai_extract_rows(
    image_png: bytes,
    api_key: Optional[str],
    model: str,
    system_prompt: Optional[str] = None,
) -> List[Dict[str, Any]]:
    if OpenAI is None:
        raise RuntimeError("openai package not available. Please install openai.")
    client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))
    if not client:
        raise RuntimeError("OPENAI_API_KEY is not set and api_key not provided.")

    b64 = base64.b64encode(image_png).decode("utf-8")
    data_url = f"data:image/png;base64,{b64}"

    # Stricter schema-driven prompt to avoid truncating codes and mis-mapping columns
    user_prompt = (
        "You are given a cropped image of an invoice payment table. Extract DATA ROWS only, "
        "returning EXACT JSON (no extra text). Each object MUST have these 6 keys: "
        "Our_Document, Your_Invoice_No, Reference, Cash_Discount, WHT_amount, Gross_amount.\n"
        "Rules:\n"
        "- Preserve FULL cell text; do not drop letters or suffixes.\n"
        "- Our_Document: numeric code, typically 10+ digits (e.g., starts with 510).\n"
        "- Your_Invoice_No: alphanumeric; MUST keep letters (e.g., 33496OG, 34472DE). Numbers-only here are likely wrong.\n"
        "- Reference: keep hyphenated codes as seen (e.g., 410197-0052-OG). If blank, use empty string.\n"
        "- Cash_Discount, WHT_amount, Gross_amount: copy as strings; do not sum; include thousands separators/commas as shown.\n"
        "- Exclude header and 'Sum total' or totals rows.\n"
        "Output ONLY a JSON array.\n\n"
        "Example output:\n"
        "[{'Our_Document':'5101181320','Your_Invoice_No':'33496OG','Reference':'410197-0052-OG','Cash_Discount':'0,00','WHT_amount':'0,00','Gross_amount':'2.531,05'}]"
    )

    # First try chat.completions (widely supported), then fallback to responses API
    try:
        chat_messages = []
        if system_prompt:
            chat_messages.append({"role": "system", "content": system_prompt})
        chat_messages.append(
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_prompt},
                    {"type": "input_image", "image_url": data_url},
                ],
            }
        )
        comp = client.chat.completions.create(
            model=model,
            messages=chat_messages,
            temperature=0,
        )
        text = comp.choices[0].message.content or ""
        parsed = _parse_json_from_text(text)
    except Exception:
        # Fallback to Responses API
        resp = client.responses.create(
            model=model,
            input=[
                {"role": "user", "content": [
                    {"type": "text", "text": user_prompt},
                    {"type": "input_image", "image_url": data_url},
                ]}
            ],
        )
        text = getattr(resp, "output_text", "")
        parsed = _parse_json_from_text(text)

    if isinstance(parsed, dict) and "rows" in parsed:
        rows = parsed.get("rows", [])
    elif isinstance(parsed, list):
        rows = parsed
    else:
        rows = []
    return rows


def _column_boundaries_from_headers(page: "fitz.Page") -> Optional[List[Tuple[float, float]]]:
    """Find header positions and derive x ranges for each column.
    Returns list of (x_start, x_end) for columns in order:
    [Our_Document, Your_Invoice_No, Reference, Cash_Discount, WHT_amount, Gross_amount]
    """
    header_variants = {
        "Our Document": ["Our Document", "Our_Document", "Document", "Doc"],
        "Your Invoice No": ["Your Invoice No", "Your Invoice No.", "Your_Invoice_No", "Your_Invoice_No."],
        "Reference": ["Reference", "Ref", "Reference No", "Ref No"],
        "Cash Discount": ["Cash Discount", "Cash_Discount", "Discount", "Disc"],
        "WHT amount": ["WHT amount", "WHT_amount", "WHT", "Tax", "Withholding"],
        "Gross amount": ["Gross amount", "Gross_amount", "Gross", "Amount", "Total"],
    }

    # Search for each canonical header using its variants
    found: List[Tuple[str, fitz.Rect]] = []
    for canonical, variants in header_variants.items():
        rect = None
        for v in variants:
            try:
                hits = page.search_for(v)
            except Exception:
                hits = []
            if hits:
                rect = hits[0]
                break
        if rect is not None:
            found.append((canonical, rect))

    if len(found) < 3:
        return None

    # Sort by x position to get reading order columns
    found.sort(key=lambda x: x[1].x0)
    centers = [r[1].x0 + (r[1].x1 - r[1].x0) / 2 for r in found]

    # Build boundaries by midpoints between consecutive centers; widen first and last
    boundaries: List[Tuple[float, float]] = []
    for i in range(len(centers)):
        if i == 0:
            left = max(0.0, centers[i] - 100)
        else:
            left = (centers[i - 1] + centers[i]) / 2
        if i == len(centers) - 1:
            right = page.rect.x1
        else:
            right = (centers[i] + centers[i + 1]) / 2
        boundaries.append((left, right))

    # If we have >=6, truncate to 6; if fewer, extrapolate to 6 evenly across page
    if len(boundaries) >= 6:
        return boundaries[:6]
    width = page.rect.x1 - page.rect.x0
    step = width / 6
    return [(i * step, (i + 1) * step) for i in range(6)]


def _infer_boundaries_from_words(page: "fitz.Page", rect: fitz.Rect) -> Optional[List[Tuple[float, float]]]:
    """Infer 6 column boundaries by clustering word x centers inside rect.
    Simple quantile-based split avoids dependency on headers.
    """
    words = page.get_text("words")
    xs = [((w[0] + w[2]) / 2) for w in words if rect.contains(fitz.Rect(w[0], w[1], w[2], w[3]))]
    if len(xs) < 30:  # need enough words to infer columns
        return None
    xs.sort()
    # Split into 6 equal index segments
    n = len(xs)
    cuts = [xs[int(n * i / 6)] for i in range(7)]  # 7 points define 6 ranges
    boundaries: List[Tuple[float, float]] = []
    for i in range(6):
        left = cuts[i]
        right = cuts[i + 1]
        # pad boundaries slightly
        boundaries.append((max(0.0, left - 20), min(page.rect.x1, right + 20)))
    return boundaries


def _extract_rows_via_text(page: "fitz.Page") -> List[Dict[str, Any]]:
    """Extract rows using PDF text, mapping words to columns by x ranges below headers.
    Preserves alphanumeric invoice numbers and references.
    """
    rect = _detect_table_rect(page) or page.rect
    boundaries = _column_boundaries_from_headers(page)
    if not boundaries:
        # Fall back to word-based boundary inference
        rect = _detect_table_rect(page) or page.rect
        boundaries = _infer_boundaries_from_words(page, rect)
        if not boundaries:
            return []

    # Get words within table rect
    words = page.get_text("words")  # x0, y0, x1, y1, text, block, line, word
    rows: List[Dict[str, Any]] = []
    # Group by y proximity instead of line index to avoid font-line splits
    words_in_rect = [w for w in words if rect.contains(fitz.Rect(w[0], w[1], w[2], w[3]))]
    words_in_rect.sort(key=lambda w: (w[1], w[0]))  # sort by y, then x

    current_y = None
    line_words: List[List] = []
    y_threshold = 3.0
    for w in words_in_rect:
        y = w[1]
        if current_y is None:
            current_y = y
            line_words = [w]
            continue
        if abs(y - current_y) <= y_threshold:
            line_words.append(w)
        else:
            # finalize previous line
            row = _words_to_row(line_words, boundaries)
            if row:
                rows.append(row)
            current_y = y
            line_words = [w]
    if line_words:
        row = _words_to_row(line_words, boundaries)
        if row:
            rows.append(row)

    # Filter out header and total rows
    def is_header(r: Dict[str, Any]) -> bool:
        header_texts = {"Our Document", "Your Invoice No.", "Reference", "Cash Discount", "WHT amount", "Gross amount"}
        return any((str(v).strip() in header_texts) for v in r.values())

    def is_total(r: Dict[str, Any]) -> bool:
        return any("sum total" in str(v).lower() for v in r.values())

    out = []
    for r in rows:
        if is_header(r) or is_total(r):
            continue
        # Basic sanity: must have Our_Document and Gross_amount
        if not str(r.get("Our_Document", "")).strip() and not str(r.get("Gross_amount", "")).strip():
            continue
        out.append(r)
    return out


def _words_to_row(line_words: List[List], boundaries: List[Tuple[float, float]]) -> Optional[Dict[str, Any]]:
    if not line_words:
        return None
    cols = ["Our_Document", "Your_Invoice_No", "Reference", "Cash_Discount", "WHT_amount", "Gross_amount"]
    cells = {c: [] for c in cols}
    # Build a full line text to help recover Reference if column split failed
    full_line_text_parts = []
    for w in sorted(line_words, key=lambda x: x[0]):
        x_center = (w[0] + w[2]) / 2
        text = w[4]
        full_line_text_parts.append(text)
        # assign to nearest boundary
        idx = None
        for i, (x0, x1) in enumerate(boundaries):
            if x0 <= x_center <= x1:
                idx = i
                break
        if idx is None:
            # if outside, snap to nearest by distance
            distances = [min(abs(x_center - x0), abs(x_center - x1)) for (x0, x1) in boundaries]
            idx = distances.index(min(distances))
        cells[cols[idx]].append(text)
    row = {k: " ".join(v).strip() for k, v in cells.items()}
    # Helper to clean and recover Reference codes
    def _clean_reference(text: str, fallback_line: str) -> str:
        # Remove visual artifacts like underscores and excessive spaces
        cleaned = re.sub(r"_+", " ", text or "").strip()
        # Robust pattern: 6 digits - 3-4 digits - 2 alphanum
        m = re.search(r"\b\d{6}-\d{3,4}-[A-Za-z0-9]{2}\b", cleaned)
        if m:
            return m.group(0)
        # Try recovery from full line text if not found in cell
        m2 = re.search(r"\b\d{6}-\d{3,4}-[A-Za-z0-9]{2}\b", fallback_line)
        if m2:
            return m2.group(0)
        # Final cleanup: collapse spaces and return cleaned text
        return cleaned

    # Recover Reference and strip artifacts
    line_text = " ".join(full_line_text_parts)
    row["Reference"] = _clean_reference(row.get("Reference", ""), line_text)
    return row


def _ollama_extract_rows(
    image_png: bytes,
    model: str = "llava:latest",
    system_prompt: Optional[str] = None,
) -> List[Dict[str, Any]]:
    if requests is None:
        raise RuntimeError("requests package not available. Please install requests.")
    b64 = base64.b64encode(image_png).decode("utf-8")
    url = "http://localhost:11434/api/generate"
    prompt = (
        (system_prompt + "\n\n") if system_prompt else ""
    ) + (
        "You are given a cropped image of an invoice payment table. Extract DATA ROWS only as JSON. "
        "Required keys per object: Our_Document, Your_Invoice_No, Reference, Cash_Discount, WHT_amount, Gross_amount. "
        "Preserve full alphanumeric codes (do not drop letters). Exclude 'Sum total' rows. Output ONLY a JSON array."
    )
    payload = {
        "model": model,
        "prompt": prompt,
        "images": [b64],
        "stream": False,
    }
    r = requests.post(url, json=payload, timeout=120)
    r.raise_for_status()
    text = r.json().get("response", "")
    parsed = _parse_json_from_text(text)
    if isinstance(parsed, list):
        return parsed
    if isinstance(parsed, dict) and "rows" in parsed:
        return parsed["rows"]
    return []


def extract_table_with_llm(
    pdf_bytes: bytes,
    bbox: Optional[Tuple[float, float, float, float]] = None,
    scale: float = 3.0,
    use_ollama: bool = False,
    openai_model: str = "gpt-4o-mini",
    openai_api_key: Optional[str] = None,
    ollama_model: str = "llava:latest",
) -> List[Dict[str, Any]]:
    """Crop pages -> images, run LLM vision, merge rows.
    Returns list of dict rows with the required columns.
    """
    # First attempt: text-based extraction (exact characters), page by page
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    text_rows_all: List[Dict[str, Any]] = []
    for page_index in range(len(doc)):
        page = doc.load_page(page_index)
        text_rows_all.extend(_extract_rows_via_text(page))

    # Second attempt: vision-based extraction as backup
    images = crop_pdf_to_images(pdf_bytes, bbox=bbox, scale=scale)
    vision_rows_all: List[Dict[str, Any]] = []
    for img in images:
        try:
            rows = (
                _ollama_extract_rows(img, model=ollama_model)
                if use_ollama
                else _openai_extract_rows(img, api_key=openai_api_key, model=openai_model)
            )
        except Exception:
            rows = []
        vision_rows_all.extend(rows)

    def has_alpha_in_invoice(rows: List[Dict[str, Any]]) -> bool:
        count = 0
        for r in rows:
            s = str(r.get("Your_Invoice_No", r.get("Your_Invoice_No.", "")))
            if any(ch.isalpha() for ch in s):
                count += 1
        return count >= max(1, int(0.3 * len(rows)))

    # Choose the better set: prefer text when it includes alphanumeric invoice codes or more rows
    if text_rows_all and (has_alpha_in_invoice(text_rows_all) or len(text_rows_all) >= len(vision_rows_all)):
        return text_rows_all
    return vision_rows_all


def save_csv(rows: List[Dict[str, Any]], output_path: str) -> str:
    """Save rows to CSV and return path."""
    if not rows:
        # create empty CSV with headers to be consistent
        df = pd.DataFrame(
            columns=[
                "Our_Document",
                "Your_Invoice_No",
                "Reference",
                "Cash_Discount",
                "WHT_amount",
                "Gross_amount",
            ]
        )
    else:
        df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    return output_path