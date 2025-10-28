import pdfplumber
import io
from typing import Any, IO
import re


def _normalize_spaces(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def _is_in_range_context(context: str) -> bool:
    ctx = context.lower()
    if "ref" in ctx or "range" in ctx or "reference" in ctx:
        return True
    if re.search(r"\d(?:\.|\d)*\s*[-–]\s*\d", ctx):
        return True
    return False


def _extract_numeric_candidates(text_segment: str) -> list[float]:
    segment = text_segment
    number_matches = list(re.finditer(r"(?<![\d.])(\d+(?:\.\d+)?)(?![\d.])", segment))
    candidates: list[float] = []
    for m in number_matches:
        left = segment[max(0, m.start() - 25):m.start()]
        right = segment[m.end(): m.end() + 25]
        if _is_in_range_context(left + right) or re.search(r"^\s*[-–]\s*\d", right):
            continue
        try:
            candidates.append(float(m.group(1)))
        except ValueError:
            continue
    return candidates


def _detect_table_columns(lines: list[str]) -> dict | None:
    for idx, line in enumerate(lines):
        low = line.lower()
        if "result" in low and ("reference" in low or "ref" in low or "range" in low):
            try:
                result_start = low.index("result")
            except ValueError:
                continue
            next_indices = [
                i for i in [
                    low.find("unit", result_start + 1),
                    low.find("reference", result_start + 1),
                    low.find("ref.", result_start + 1),
                    low.find("ref ", result_start + 1),
                    low.find("range", result_start + 1),
                ] if i != -1
            ]
            result_end = min(next_indices) if next_indices else None
            return {
                "header_index": idx,
                "result_start": result_start,
                "result_end": result_end,
            }
    return None


def _open_pdf(src: Any):
    if isinstance(src, (bytes, bytearray)):
        return pdfplumber.open(io.BytesIO(src))
    # File-like object
    if hasattr(src, "read"):
        try:
            src.seek(0)
        except Exception:
            pass
        return pdfplumber.open(src)
    # Path
    if isinstance(src, str):
        return pdfplumber.open(src)
    raise TypeError("Unsupported PDF input type; expected path, bytes, or file-like object")


def extract_kft_from_pdf(filepath_or_bytes: Any) -> dict:
    with _open_pdf(filepath_or_bytes) as pdf:
        raw_text = "\n".join(page.extract_text() or "" for page in pdf.pages)

    print("DEBUG: raw PDF text snippet:", raw_text[:200], "…")

    lines = [_normalize_spaces(ln) for ln in raw_text.splitlines() if ln and ln.strip()]
    lower_lines = [ln.lower() for ln in lines]

    label_synonyms: dict[str, list[str]] = {
        "Creatinine": [
            "serum creatinine", "creatinine", "s. creatinine", "creatinine (serum)",
        ],
        "Urea": [
            "urea", "bun", "blood urea", "urea nitrogen", "s. urea",
        ],
        "Potassium": [
            "potassium", "k+", "k +", "k ", "serum potassium", "k (serum)",
        ],
        "Sodium": [
            "sodium", "na+", "na +", "na ", "serum sodium", "na (serum)",
        ],
        "Phosphorus": [
            "phosphorus", "phosphate", "inorganic phosphorus", "serum phosphate",
        ],
    }

    results: dict[str, float] = {k: 0.0 for k in label_synonyms.keys()}

    plausible_ranges: dict[str, tuple[float, float]] = {
        "Creatinine": (0.1, 25.0),
        "Urea": (1.0, 300.0),
        "Potassium": (2.0, 8.0),
        "Sodium": (100.0, 170.0),
        "Phosphorus": (1.0, 12.0),
    }

    def is_plausible(label: str, value: float) -> bool:
        lo, hi = plausible_ranges[label]
        return lo <= value <= hi

    def group_words_into_lines(words):
        if not words:
            return []
        words_sorted = sorted(words, key=lambda w: (w["top"], w["x0"]))
        lines_grouped = []
        current = {"top": None, "bottom": None, "words": []}
        tol = 2.5
        for w in words_sorted:
            if current["top"] is None:
                current = {"top": w["top"], "bottom": w["bottom"], "words": [w]}
                continue
            if abs(w["top"] - current["top"]) <= tol:
                current["words"].append(w)
                current["bottom"] = max(current["bottom"], w["bottom"])
            else:
                current_words = sorted(current["words"], key=lambda a: a["x0"])  # left-to-right
                line_text = _normalize_spaces(" ".join(t["text"] for t in current_words))
                lines_grouped.append({
                    "top": current["top"],
                    "bottom": current["bottom"],
                    "x0": min(t["x0"] for t in current_words),
                    "x1": max(t["x1"] for t in current_words),
                    "text": line_text,
                    "tokens": current_words,
                })
                current = {"top": w["top"], "bottom": w["bottom"], "words": [w]}
        if current["words"]:
            current_words = sorted(current["words"], key=lambda a: a["x0"])  # left-to-right
            line_text = _normalize_spaces(" ".join(t["text"] for t in current_words))
            lines_grouped.append({
                "top": current["top"],
                "bottom": current["bottom"],
                "x0": min(t["x0"] for t in current_words),
                "x1": max(t["x1"] for t in current_words),
                "text": line_text,
                "tokens": current_words,
            })
        return lines_grouped

    def detect_result_band_from_header(lines_grouped):
        ref_aliases = ["ref", "range", "reference", "interval"]

        for ln in lines_grouped:
            low = ln["text"].lower()
            if "result" not in low:
                continue
            if not any(a in low for a in ref_aliases):
                pass
            tokens = [t for t in ln["tokens"] if t["text"].strip()]
            tokens_sorted = sorted(tokens, key=lambda t: t["x0"])
            mids = [(i, (tok["x0"] + tok["x1"]) / 2.0, tok) for i, tok in enumerate(tokens_sorted)]
            idx_result = None
            for i, mid, tok in mids:
                if tok["text"].strip().lower().startswith("result"):
                    idx_result = i
                    break
            if idx_result is None:
                continue
            prev_mid = mids[idx_result - 1][1] if idx_result - 1 >= 0 else None
            this_mid = mids[idx_result][1]
            next_mid = mids[idx_result + 1][1] if idx_result + 1 < len(mids) else None
            pad = 5.0
            x_min = (prev_mid + this_mid) / 2.0 if prev_mid is not None else (mids[idx_result][2]["x0"] - pad)
            x_max = (this_mid + next_mid) / 2.0 if next_mid is not None else (mids[idx_result][2]["x1"] + 120.0)
            return {"x_min": x_min, "x_max": x_max, "header_text": ln["text"]}
        return None

    def text_in_band(tokens, x_min, x_max) -> str:
        chosen = [t["text"] for t in tokens if not (t["x1"] < x_min or t["x0"] > x_max)]
        return _normalize_spaces(" ".join(chosen))

    def first_number_from_text(s: str) -> float | None:
        for m in re.finditer(r"(?<![\d.])[-+]?\d+(?:\.\d+)?(?![\d.])", s):
            span_text = s[max(0, m.start() - 5): m.end() + 5]
            if re.search(r"\d\s*[-–]\s*\d", span_text):
                continue
            try:
                return float(m.group(0))
            except ValueError:
                continue
        return None

    columns = _detect_table_columns(lines)

    def slice_result_cell(line: str) -> str:
        if not columns:
            return line
        start = columns["result_start"]
        end = columns["result_end"]
        cell = line[start:end] if end is not None else line[start:]
        return cell

    def find_value_for_synonyms(syns: list[str]) -> float | None:
        for idx, low_line in enumerate(lower_lines):
            if not any(syn in low_line for syn in syns):
                continue
            candidates = _extract_numeric_candidates(slice_result_cell(lines[idx]))
            if candidates:
                return candidates[0]
            for look_ahead in (1, 2):
                if idx + look_ahead >= len(lines):
                    break
                candidates = _extract_numeric_candidates(slice_result_cell(lines[idx + look_ahead]))
                if candidates:
                    return candidates[0]
        return None

    try:
        with _open_pdf(filepath_or_bytes) as pdf:
            for page in pdf.pages:
                words = page.extract_words(x_tolerance=1.5, y_tolerance=2.5, keep_blank_chars=False, use_text_flow=False)
                lines_grouped = group_words_into_lines(words)
                band = detect_result_band_from_header(lines_grouped)
                if band:
                    print(f"DEBUG: header='{band['header_text']}' result_band=({band['x_min']:.1f},{band['x_max']:.1f})")
                if not band:
                    continue
                x_min, x_max = band["x_min"], band["x_max"]
                for label, syns in label_synonyms.items():
                    if results[label] and results[label] != 0.0:
                        continue
                    for idx, ln in enumerate(lines_grouped):
                        low = ln["text"].lower()
                        if not any(s in low for s in syns):
                            continue
                        band_text = text_in_band(ln["tokens"], x_min, x_max)
                        num = first_number_from_text(band_text)
                        if num is None and idx + 1 < len(lines_grouped):
                            band_text_next = text_in_band(lines_grouped[idx + 1]["tokens"], x_min, x_max)
                            num = first_number_from_text(band_text_next)
                        if num is not None and is_plausible(label, num):
                            results[label] = num
                            break
                if all(results[k] != 0.0 for k in results.keys()):
                    break
    except Exception as e:
        print(f"DEBUG: geometry-aware extraction skipped due to error: {e}")

    for label in results.keys():
        if results[label] == 0.0:
            value = find_value_for_synonyms(label_synonyms[label])
            if value is not None and is_plausible(label, value):
                results[label] = value
            elif value is not None:
                if value < 1000:
                    results[label] = value
                else:
                    print(f"WARNING: {label} candidate {value} looks implausible, ignoring")

    print("DEBUG: extracted KFT values:", results)
    return results


# Compatibility wrappers expected by callers/tests

def parse_kft_report(filepath: str) -> dict:
    return extract_kft_from_pdf(filepath)


def parse_kft(filepath: str) -> dict:
    return extract_kft_from_pdf(filepath)
