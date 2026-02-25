#!/usr/bin/env python3
"""
Build zoo_v2.csv — 2-column (name, expression) CSV for factor redundancy checking.

Source: data/factorlib/all_factors_library_merged_multi_llm_v2.json
Output: data/factorlib/zoo_v2.csv

Usage:
    python scripts/build_zoo_csv.py [--src PATH] [--out PATH]
"""
import argparse
import json
import pathlib
import sys

REPO = pathlib.Path(__file__).resolve().parents[1]
DEFAULT_SRC = REPO / "data/factorlib/all_factors_library_merged_multi_llm_v2.json"
DEFAULT_OUT = REPO / "data/factorlib/zoo_v2.csv"


def build(src: pathlib.Path, out: pathlib.Path, quality_filter: str | None = "high_quality") -> int:
    data = json.loads(src.read_text())
    factors = data.get("factors", {})

    rows = []
    for name, meta in factors.items():
        if quality_filter and meta.get("quality") != quality_filter:
            continue
        expr = (meta.get("factor_expression") or "").strip()
        if not expr:
            continue
        # Escape commas inside expressions by quoting the field
        rows.append((name, expr))

    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        f.write("name,expression\n")
        for name, expr in rows:
            # RFC-4180 quoting: wrap in quotes, escape inner quotes by doubling
            safe_name = f'"{name}"' if "," in name or '"' in name else name
            safe_expr = f'"{expr.replace(chr(34), chr(34)*2)}"'
            f.write(f"{safe_name},{safe_expr}\n")

    print(f"✅  Wrote {len(rows)} factors → {out}")
    return len(rows)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src", default=str(DEFAULT_SRC))
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--quality", default="high_quality",
                    help="quality_filter value; use 'all' to include every factor")
    args = ap.parse_args()

    src = pathlib.Path(args.src)
    if not src.exists():
        print(f"❌  Source not found: {src}", file=sys.stderr)
        sys.exit(1)

    n = build(src, pathlib.Path(args.out),
              quality_filter=None if args.quality == "all" else args.quality)
    if n == 0:
        print("⚠️  No factors written — check --quality filter", file=sys.stderr)
        sys.exit(1)
