"""Clean query synonym dictionary using strict intent-preserving rules.

Rules:
1) Keep only close semantic variants (drop broad/general terms).
2) Remove synonyms that change intent or are overly generic.
3) Remove duplicates and self-maps.
4) Keep at most N synonyms per term (default 2).

Usage:
    python -m tools.clean_synonym_dictionary
    python -m tools.clean_synonym_dictionary --in-place
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Set

from src.config import QUERY_SYNONYMS_PATH

# Broad/general terms that often hurt precision.
BROAD_TERMS: Dict[str, Set[str]] = {
    "en": {
        "internet", "system", "service", "services", "thing", "general",
        "data", "information", "details", "record", "records", "document", "documents",
    },
    "hi": {
        "जानकारी", "सूचना", "डेटा", "सेवा", "सामान्य",
    },
    "te": {
        "సమాచారం", "డేటా", "సేవ", "సాధారణ",
    },
}


def normalize(text: str, lang: str) -> str:
    t = text.strip()
    return t.lower() if lang == "en" else t


def clean_synonyms_for_term(term: str, synonyms: List[str], lang: str, max_per_term: int) -> List[str]:
    out: List[str] = []
    seen: Set[str] = set()

    term_norm = normalize(term, lang)
    blocked = BROAD_TERMS.get(lang, set())

    for s in synonyms:
        if not isinstance(s, str):
            continue
        cand = s.strip()
        if not cand:
            continue

        cand_norm = normalize(cand, lang)

        # Remove self-synonyms and duplicates.
        if cand_norm == term_norm or cand_norm in seen:
            continue

        # Remove broad/general terms.
        if cand_norm in blocked:
            continue

        # Avoid phrase-length mismatch that may alter intent.
        if len(cand.split()) != len(term.split()):
            continue

        seen.add(cand_norm)
        out.append(cand)

        if len(out) >= max_per_term:
            break

    return out


def clean_dictionary(data: Dict[str, Dict[str, List[str]]], max_per_term: int) -> Dict[str, Dict[str, List[str]]]:
    cleaned: Dict[str, Dict[str, List[str]]] = {}

    for lang, mapping in data.items():
        if not isinstance(mapping, dict):
            continue

        lang_cleaned: Dict[str, List[str]] = {}
        for term, synonyms in mapping.items():
            if not isinstance(term, str) or not isinstance(synonyms, list):
                continue

            cleaned_synonyms = clean_synonyms_for_term(term, synonyms, lang, max_per_term)
            if cleaned_synonyms:
                lang_cleaned[term] = cleaned_synonyms

        cleaned[lang] = lang_cleaned

    return cleaned


def main():
    parser = argparse.ArgumentParser(description="Clean synonym dictionary")
    parser.add_argument("--input", type=str, default=str(QUERY_SYNONYMS_PATH))
    parser.add_argument("--output", type=str, default="")
    parser.add_argument("--max-per-term", type=int, default=2)
    parser.add_argument("--in-place", action="store_true")
    args = parser.parse_args()

    in_path = Path(args.input)
    if not in_path.exists():
        raise FileNotFoundError(f"Input synonym dictionary not found: {in_path}")

    with open(in_path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    if not isinstance(raw, dict):
        raise ValueError("Input synonym dictionary must be a JSON object")

    cleaned = clean_dictionary(raw, max_per_term=max(1, int(args.max_per_term)))

    out_path = in_path if args.in_place else Path(args.output) if args.output else in_path.with_name(in_path.stem + ".cleaned.json")

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(cleaned, f, indent=2, ensure_ascii=False)

    print(f"Cleaned dictionary written to: {out_path}")


if __name__ == "__main__":
    main()
