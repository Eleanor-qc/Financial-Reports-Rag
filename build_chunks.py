import os
import re
import time
import pickle
from typing import List, Dict
from collections import Counter
import pdfplumber
import numpy as np
from statistics import median
from google.cloud import storage

# =========================
# Config
# =========================
PROJECT_ID = PROJECT_ID
LOCATION = LOCATION
BUCKET_NAME = BUCKET_NAME

COMPANIES = ["Alphabet", "Amazon", "Microsoft", "Oracle"]
YEARS = range(2020, 2026)
PDF_TEMPLATE = "{company} 10-K/{company} {year}.pdf"

TARGET_ITEMS = {
    "Item 1.": "BUSINESS",
    "Item 1A.": "RISK_FACTORS",
    "Item 1C.": "CYBERSECURITY",
    "Item 7.": "MDA",
    "Item 7A.": "MARKET_RISK"
}

TARGET_ITEM_RE = re.compile(r"^\s*Item\s+(1A|1C|7A|1|7)\b", re.IGNORECASE)
ITEM_RE = re.compile(r"^\s*Item\s+\d+[A-Z]?\b", re.IGNORECASE)
ITEM5_RE = re.compile(r"^\s*Item\s+5\b", re.IGNORECASE)


# =========================
# Helper functions
# =========================
def is_toc_page_simple(lines, page_num, max_front_pages=5, min_item_lines=5):
    if page_num > max_front_pages:
        return False
    has_item5 = any(ITEM5_RE.match(l["text"]) for l in lines)
    if not has_item5:
        return False
    item_lines = sum(1 for l in lines if ITEM_RE.match(l["text"]))
    return item_lines >= min_item_lines


def detect_item_start(line):
    text = line["text"]

    if not ITEM_RE.match(text):
        return None
    if line["bold_ratio"] < 0.6:
        return None

    m = TARGET_ITEM_RE.match(text)
    if m:
        k = m.group(1).upper()
        label = f"Item {k}."
        return label if label in TARGET_ITEMS else "__OTHER_ITEM__"
    return "__OTHER_ITEM__"


def label_lines_with_items(lines: List[Dict], start_item=None):
    current_item = start_item
    labeled = []

    for line in lines:
        detected = detect_item_start(line)
        if detected in TARGET_ITEMS:
            current_item = detected
        elif detected == "__OTHER_ITEM__":
            current_item = None

        labeled.append({**line, "item": current_item})

    return labeled, current_item

END_PUNCT_RE = re.compile(r"[\.!?;:]\s*$")

def norm_line(s: str) -> str:
    s = s.replace("’", "'")
    s = re.sub(r"\s+", " ", s)
    return s

def extract_lines_from_words(page, y_tol=3):

    words = page.extract_words(use_text_flow=True)
    if not words:
        return []

    words = sorted(words, key=lambda w: (w["top"], w["x0"]))

    lines = []
    cur = [words[0]]
    base_top = words[0]["top"]

    for w in words[1:]:
        if abs(w["top"] - base_top) <= y_tol:
            cur.append(w)
        else:
            lines.append(cur)
            cur = [w]
            base_top = w["top"]

    lines.append(cur)

    out = []

    for line_words in lines:

        line_words = sorted(line_words, key=lambda w: w["x0"])

        text = " ".join(w["text"] for w in line_words).strip()

        if not text:
            continue

        # bbox
        x0 = min(w["x0"] for w in line_words)
        x1 = max(w["x1"] for w in line_words)
        top = min(w["top"] for w in line_words)
        bottom = max(w["bottom"] for w in line_words)

        # 提取这一行的 chars
        chars = [
            c for c in page.chars
            if top <= c["top"] <= bottom
        ]

        # bold ratio
        bold_chars = [
            c for c in chars
            if "bold" in c["fontname"].lower()
        ]

        bold_ratio = len(bold_chars) / len(chars) if chars else 0

        out.append({
            "page": page.page_number,
            "text": text,
            "top": top,
            "bottom": bottom,
            "x0": x0,
            "x1": x1,
            "bold_ratio": bold_ratio,
            "break_before": False
        })

    return out[:-1]
    
def mark_paragraph_breaks(lines):
    gaps = [lines[i]["top"] - lines[i-1]["bottom"] for i in range(1, len(lines))]
    base_gap = median(gaps) if gaps else 0.0

    for i in range(1, len(lines)):
        gap = lines[i]["top"] - lines[i-1]["bottom"]
        if gap > (base_gap * 1.1):
            lines[i]["break_before"] = True

    return lines

PART_RE = re.compile(
    r"^\s*PART\s+[IVXLC]+\b",
    re.IGNORECASE
)

HEADER_ITEM_LIST_RE = re.compile(
    r"^\s*Item\s+\d+[A-Z]?(?:\s*,\s*\d+[A-Z]?)*\s*$",
    re.IGNORECASE
)

PART_RE = re.compile(
    r"^\s*PART\s+[IVXLC]+\b",
    re.IGNORECASE
)

HEADER_ITEM_LIST_RE = re.compile(
    r"^\s*Item\s+\d+[A-Z]?(?:\s*,\s*\d+[A-Z]?)*\s*$",
    re.IGNORECASE
)

def learn_header_templates(pdf, top_k=5, top_band_pct=0.25, min_frac=0.6):
    """
    learn repetitive part of the top band of each page as header
    but avoid pattern like: PART I / Item 1B,2,3,4
    """

    n = len(pdf.pages)
    end_page = n - 5
    start_page = 5

    if start_page >= end_page:
        return set()

    header_counter = Counter()
    pages_used = 0

    for p_idx in range(start_page, end_page):

        page = pdf.pages[p_idx]
        H = page.height

        lines = extract_lines_from_words(page)

        if not lines:
            continue

        pages_used += 1

        top_band = top_band_pct * H

        top_lines = sorted(
            [l for l in lines if l["top"] < top_band],
            key=lambda x: x["top"]
        )[:top_k]

        for l in top_lines:

            text = l["text"].strip()

            if PART_RE.match(text):
                continue

            if HEADER_ITEM_LIST_RE.match(text):
                continue

            t = norm_line(text)

            if t:
                header_counter[t] += 1

    if pages_used == 0:
        return set()

    thresh = int(min_frac * pages_used)

    header_templates = {
        t for t, c in header_counter.items()
        if c >= thresh
    }

    return header_templates
    
def drop_headers(lines, page_height, header_templates, top_band_pct=0.25):

    top_band = top_band_pct * page_height

    return [
        l for l in lines
        if not (
            l["top"] <= top_band and (
                PART_RE.match(l["text"].strip())
                or HEADER_ITEM_LIST_RE.match(l["text"].strip())
                or norm_line(l["text"]) in header_templates
            )
        )
    ]

def process_single_pdf(pdf_path: str, company: str, year: int):
    all_chunks = []

    with pdfplumber.open(pdf_path) as pdf:
        header_templates = learn_header_templates(pdf)

        current_item_global = None
        paragraph_buffer = []
        current_item = None
        chunk_idx = 0

        def flush(buf, item):
            nonlocal chunk_idx
            if not buf or item not in TARGET_ITEMS:
                return

            text = " ".join(buf).strip()
            if not text:
                return

            safe_item = item.replace(" ", "").replace(".", "")
            chunk_id = f"{company}_{year}_{safe_item}_{chunk_idx}"
            title = f"{company} {year} {item} #{chunk_idx}"

            all_chunks.append({
                "chunk_id": chunk_id,
                "task_type": "RETRIEVAL_DOCUMENT",
                "title": title,
                "content": text,
                "metadata": {
                    "company": company,
                    "year": year,
                    "item": item,
                    "item_type": TARGET_ITEMS[item],
                    "chunk_id": chunk_id
                }
            })
            chunk_idx += 1

        for page in pdf.pages:
            lines = extract_lines_from_words(page)
            if not lines:
                continue

            if is_toc_page_simple(lines, page.page_number):
                continue

            lines = drop_headers(lines, page.height, header_templates, top_band_pct=0.25)
            lines = mark_paragraph_breaks(lines)
            lines, current_item_global = label_lines_with_items(
                lines, start_item=current_item_global
            )
            lines = [l for l in lines if l.get("item") in TARGET_ITEMS]

            for l in lines:
                item = l["item"]
                text = l["text"].strip()

                if item != current_item:
                    flush(paragraph_buffer, current_item)
                    paragraph_buffer = []
                    current_item = item

                if l.get("break_before") and paragraph_buffer:
                    flush(paragraph_buffer, current_item)
                    paragraph_buffer = []

                paragraph_buffer.append(text)

        flush(paragraph_buffer, current_item)

    return all_chunks

def build_chunks():
    chunks = []

    for company in COMPANIES:
        for year in YEARS:
            pdf_file = PDF_TEMPLATE.format(company=company, year=year)

            if not os.path.exists(pdf_file):
                print(f"Missing file: {pdf_file}, skipping.")
                continue

            print(f"Processing {pdf_file}...")
            year_chunks = process_single_pdf(pdf_file, company, year)
            chunks.extend(year_chunks)

    print(f"Total chunks prepared: {len(chunks)}")
    return chunks

def save_and_upload_chunks(chunks):
    chunks_text = [
        {
            "chunk_id": item["chunk_id"],
            "title": item["title"],
            "content": item["content"]
        }
        for item in chunks
    ]

    with open("chunks_text.pkl", "wb") as f:
        pickle.dump(chunks_text, f)

    chunks_metadata = [item["metadata"] for item in chunks]

    with open("chunks_metadata.pkl", "wb") as f:
        pickle.dump(chunks_metadata, f)

        client = storage.Client(project="qc2360-ieor4526-fall2025")
    
    bucket = client.bucket("qc2360-fall2025-bucket")
    
    # Upload chunks_text.pkl
    blob = bucket.blob("rag/chunks_text.pkl")
    blob.upload_from_filename("chunks_text.pkl")
    
    # Upload chunks_metadata.pkl
    blob = bucket.blob("rag/chunks_metadata.pkl")
    blob.upload_from_filename("chunks_metadata.pkl")

    print("Saved and uploaded chunks_text.pkl and chunks_metadata.pkl")

def main():
    chunks_generated = build_chunks()
    save_and_upload_chunks(chunks_generated)

    print("Chunks uploaded.")
