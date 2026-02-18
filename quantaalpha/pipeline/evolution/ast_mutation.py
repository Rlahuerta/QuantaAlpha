"""
AST-level expression mutation for Phase C diversity.

Provides three layers of expression mutation hints for LLM mutation prompts:

1. **Window scaling** — perturb integer window params by ×0.5 / ×2
2. **Operator substitution** — swap semantically-similar temporal operators
   (e.g. TS_MEAN → EMA/WMA/DECAYLINEAR, TS_STD → TS_ZSCORE)
3. **Structural wrapping** — suggest RANK/ZSCORE cross-sectional wrappers
   and DELTA(·,1) temporal differentiation if not already present

None of these require an LLM call; they run in microseconds and are appended
to the mutation-round prompt so the LLM is guided toward unexplored variants.
"""

from __future__ import annotations

import re
import random
from typing import Sequence


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Temporal operators whose first numeric argument is a window parameter.
_TEMPORAL_OPS = {
    "TS_MEAN", "TS_STD", "TS_MAX", "TS_MIN", "TS_SUM", "TS_RANK",
    "TS_ZSCORE", "TS_CORR", "TS_COVARIANCE",
    "DELTA", "DELAY", "EMA", "WMA", "DECAYLINEAR", "SUMAC",
    "SMA",  # SMA(A, n, m) — n is window
}

# Operator substitution map: operator → list of semantically similar alternatives.
# Groups: weighted moving averages, dispersion, extremes, cross-sectional norms.
_OP_SUBSTITUTIONS: dict[str, list[str]] = {
    "TS_MEAN":     ["EMA", "WMA", "DECAYLINEAR"],
    "EMA":         ["TS_MEAN", "WMA", "DECAYLINEAR"],
    "WMA":         ["TS_MEAN", "EMA", "DECAYLINEAR"],
    "DECAYLINEAR": ["TS_MEAN", "EMA", "WMA"],
    "TS_STD":      ["TS_ZSCORE"],
    "TS_ZSCORE":   ["TS_STD"],
    "TS_MAX":      ["TS_MIN"],
    "TS_MIN":      ["TS_MAX"],
    "RANK":        ["ZSCORE"],
    "ZSCORE":      ["RANK"],
}

# Cross-sectional wrappers to suggest when the expression lacks one.
_CS_WRAPPERS = ("RANK", "ZSCORE")

# Temporal wrappers that add a derivative / momentum dimension.
_TEMPORAL_WRAPPERS = ("DELTA",)

# Pattern: FUNC_NAME(... , <integer> ...)  — matches the integer window argument.
_WINDOW_PATTERN = re.compile(
    r"(?<![.\d])(\b(?:" + "|".join(_TEMPORAL_OPS) + r")\b)"  # operator name
    r"(\s*\([^)]*?,\s*)"                                       # args before window
    r"(\b\d+\b)"                                               # window integer
    r"(\s*(?:[,)]|$))",                                        # trailing , or )
    re.IGNORECASE,
)

# Pattern to detect if an expression is already wrapped by a cross-sectional op.
_CS_WRAP_PATTERN = re.compile(
    r"^\s*(?:RANK|ZSCORE)\s*\(", re.IGNORECASE
)

# Pattern to detect if an expression is already differentiated.
_DELTA_WRAP_PATTERN = re.compile(
    r"^\s*DELTA\s*\(", re.IGNORECASE
)


# ---------------------------------------------------------------------------
# Layer 1: Window scaling
# ---------------------------------------------------------------------------

def _scale_window(n: int, factor: float, min_val: int = 2, max_val: int = 240) -> int:
    """Scale window n by factor, clip to [min_val, max_val], round to nearest int."""
    scaled = round(n * factor)
    return max(min_val, min(max_val, scaled))


def perturb_expression_windows(
    expression: str,
    scales: Sequence[float] = (0.5, 2.0),
    rng: random.Random | None = None,
) -> list[str]:
    """
    Return a list of window-scaled variants of *expression*.

    All integer window parameters inside temporal operator calls are scaled
    simultaneously by each factor in *scales*.

    Example:
        >>> perturb_expression_windows("RANK(TS_MEAN($return, 20) / TS_STD($return, 10))")
        ["RANK(TS_MEAN($return, 10) / TS_STD($return, 5))",   # ×0.5
         "RANK(TS_MEAN($return, 40) / TS_STD($return, 20))"]  # ×2.0
    """
    if rng is None:
        rng = random.Random(42)

    windows: list[tuple[int, int, int]] = []
    for m in _WINDOW_PATTERN.finditer(expression):
        windows.append((m.start(3), m.end(3), int(m.group(3))))

    if not windows:
        return []

    seen: set[str] = set()
    result: list[str] = []
    for scale in scales:
        chars = list(expression)
        for start, end, val in reversed(windows):
            chars[start:end] = list(str(_scale_window(val, scale)))
        perturbed = "".join(chars)
        if perturbed != expression and perturbed not in seen:
            seen.add(perturbed)
            result.append(perturbed)
    return result


# ---------------------------------------------------------------------------
# Layer 2: Operator substitution
# ---------------------------------------------------------------------------

def substitute_operators(expression: str, max_per_op: int = 1) -> list[str]:
    """
    Return variants where one temporal operator is replaced by a sibling.

    For each operator found in *expression* that has known substitutes,
    produces *max_per_op* replacement variants (the first alternatives listed
    in ``_OP_SUBSTITUTIONS``).

    Example:
        >>> substitute_operators("RANK(TS_MEAN($return, 20) / TS_STD($return, 10))")
        ["RANK(EMA($return, 20) / TS_STD($return, 10))",
         "RANK(TS_MEAN($return, 20) / TS_ZSCORE($return, 10))"]
    """
    seen: set[str] = set()
    result: list[str] = []

    for op, substitutes in _OP_SUBSTITUTIONS.items():
        # Case-insensitive whole-word match for the operator name
        pattern = re.compile(r"(?<!\w)" + re.escape(op) + r"(?=\s*\()", re.IGNORECASE)
        if not pattern.search(expression):
            continue
        for sub in substitutes[:max_per_op]:
            variant = pattern.sub(sub, expression, count=1)
            if variant != expression and variant not in seen:
                seen.add(variant)
                result.append(variant)

    return result


# ---------------------------------------------------------------------------
# Layer 3: Structural wrapping
# ---------------------------------------------------------------------------

def wrap_expression(expression: str) -> list[str]:
    """
    Suggest structural wrappers not already present on *expression*.

    - Cross-sectional: RANK(·) and ZSCORE(·) if expression lacks them.
    - Temporal derivative: DELTA(·, 1) if expression lacks it.

    Returns a list of (label, wrapped_expression) tuples for prompt formatting.
    """
    results: list[tuple[str, str]] = []
    stripped = expression.strip()

    # Cross-sectional normalization
    if not _CS_WRAP_PATTERN.match(stripped):
        for op in _CS_WRAPPERS:
            results.append((f"{op} normalization", f"{op}({stripped})"))

    # Temporal differentiation (1-period change of the factor)
    if not _DELTA_WRAP_PATTERN.match(stripped):
        results.append(("temporal derivative (momentum)", f"DELTA({stripped}, 1)"))

    return results


# ---------------------------------------------------------------------------
# Combined hint builder
# ---------------------------------------------------------------------------

def build_ast_mutation_hint(
    best_expression: str,
    factor_name: str = "",
    scales: Sequence[float] = (0.5, 2.0),
) -> str:
    """
    Build a multi-section prompt hint covering all three mutation axes.

    Sections included (only if non-empty):
      1. Window-parameter variants (×0.5, ×2)
      2. Operator-substitution variants
      3. Structural wrappers (RANK, ZSCORE, DELTA)

    Returns an empty string if no mutations are possible.
    """
    label = f" for `{factor_name}`" if factor_name else ""
    sections: list[str] = [f"\n### AST Expression Variants{label}",
                           f"Base: `{best_expression}`\n"]

    # 1. Window scaling
    window_variants = perturb_expression_windows(best_expression, scales=scales)
    if window_variants:
        sections.append("**Time-horizon variants** (window ×0.5 / ×2):")
        for v in window_variants:
            sections.append(f"  - `{v}`")

    # 2. Operator substitution
    op_variants = substitute_operators(best_expression)
    if op_variants:
        sections.append("\n**Operator-substitution variants** (similar semantics, different weighting):")
        for v in op_variants:
            sections.append(f"  - `{v}`")

    # 3. Structural wrappers
    wrap_variants = wrap_expression(best_expression)
    if wrap_variants:
        sections.append("\n**Structural wrappers** (unexplored normalization / differentiation):")
        for desc, v in wrap_variants:
            sections.append(f"  - `{v}`  ← {desc}")

    # Return empty string if nothing was generated
    if len(sections) <= 2:
        return ""

    sections.append(
        "\nPick the most promising variant or combine ideas; rename the factor to reflect the change."
    )
    return "\n".join(sections)
