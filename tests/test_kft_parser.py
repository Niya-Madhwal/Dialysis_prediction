import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional


def _find_project_root(start: Optional[Path] = None) -> Path:
    here = (start or Path(__file__).resolve()).parent
    for parent in [here, *here.parents]:
        if (parent / "app").is_dir():
            return parent
    return Path.cwd()


def _import_parser():
    # Ensure project root (where `app/` lives) is importable
    root = _find_project_root()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    try:
        # Preferred: our updated pdfplumber-based extractor
        from app.kft_parser import extract_kft_from_pdf  # type: ignore
        return extract_kft_from_pdf
    except Exception as first_error:  # pragma: no cover
        # Fallback: try OCR-based parser (may be Windows-specific paths)
        try:
            from app.kft_parser import extract_kft_from_pdf  # type: ignore
            return extract_kft_from_pdf
        except Exception as second_error:
            print("Failed to import KFT parsers:")
            print(" - app.templates.kft_parser error:", repr(first_error))
            print(" - app.kft_parser error:", repr(second_error))
            raise


def _resolve_default_pdf() -> Optional[str]:
    root = _find_project_root()
    candidates = [
        str(root / "uploads" / "kft-test-normal-report-format-example-sample-template-drlogy-lab-report.pdf"),
        str(root / "sample.pdf"),
    ]
    for path in candidates:
        if os.path.isfile(path):
            return path
    return None


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Test KFT PDF extractor and print results as JSON")
    parser.add_argument("pdf", nargs="?", help="Path to KFT report PDF. If omitted, tries a default sample path.")
    args = parser.parse_args(argv)

    pdf_path = args.pdf or _resolve_default_pdf()
    if not pdf_path:
        print("Error: No PDF path provided and no default sample PDF found.")
        print("Usage: python test_kft_parser.py /path/to/report.pdf")
        return 2

    if not os.path.isfile(pdf_path):
        print(f"Error: File does not exist: {pdf_path}")
        return 2

    try:
        extractor = _import_parser()
    except Exception:
        return 1

    try:
        result: Dict[str, Any] = extractor(pdf_path)
    except Exception as e:
        print(f"Extraction failed: {e}")
        return 1

    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

