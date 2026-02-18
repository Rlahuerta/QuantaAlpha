"""
AST-level expression parameter mutation.

Takes a valid factor expression and perturbs integer window parameters by ×0.5
or ×2, producing a structurally identical factor that operates over a different
time horizon.  Used during mutation rounds to generate concrete expression hints
for the LLM hypothesis generator without requiring a full LLM call.

Typical use-case: parent mined `TS_MEAN($return, 20)` → perturbation suggests
`TS_MEAN($return, 10)` and `TS_MEAN($return, 40)` as sibling factors to explore.
"""

from __future__ import annotations

import re
import random
from typing import Sequence


# Temporal operators whose first numeric argument is a window parameter.
_TEMPORAL_OPS = {
    "TS_MEAN", "TS_STD", "TS_MAX", "TS_MIN", "TS_SUM", "TS_RANK",
    "TS_ZSCORE", "TS_CORR", "TS_COVARIANCE",
    "DELTA", "DELAY", "EMA", "WMA", "DECAYLINEAR", "SUMAC",
    "SMA",  # SMA(A, n, m) — n is window
}

# Pattern: FUNC_NAME(... , <integer> ...)  — matches the integer window argument
# We look for integer literals that appear as a standalone argument (comma-separated).
_WINDOW_PATTERN = re.compile(
    r"(?<![.\d])(\b(?:" + "|".join(_TEMPORAL_OPS) + r")\b)"  # operator name
    r"(\s*\([^)]*?,\s*)"                                       # args before window
    r"(\b\d+\b)"                                               # window integer
    r"(\s*(?:[,)]|$))",                                        # trailing , or )
    re.IGNORECASE,
)


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
    Return a list of perturbed variants of *expression*.

    For each integer window parameter found inside a temporal operator call,
    one variant is produced per scale factor.  Scales are applied independently
    per parameter so that all windows in an expression change together.

    Args:
        expression: Valid factor DSL expression string.
        scales: Multipliers to apply to each window integer (default: ×0.5, ×2).
        rng: Optional Random instance for reproducibility.

    Returns:
        List of (possibly duplicate-free) perturbed expression strings.
        Returns empty list if no window parameters are found.

    Example:
        >>> perturb_expression_windows("RANK(TS_MEAN($return, 20) / TS_STD($return, 10))")
        [
            "RANK(TS_MEAN($return, 10) / TS_STD($return, 5))",   # ×0.5
            "RANK(TS_MEAN($return, 40) / TS_STD($return, 20))",  # ×2.0
        ]
    """
    if rng is None:
        rng = random.Random(42)

    # Find all window integers in temporal operator positions.
    # We use a two-pass approach: locate positions, then substitute.
    windows: list[tuple[int, int, int]] = []  # (match_start_of_int, end_of_int, value)

    for m in _WINDOW_PATTERN.finditer(expression):
        # group 3 is the integer window
        start = m.start(3)
        end = m.end(3)
        val = int(m.group(3))
        windows.append((start, end, val))

    if not windows:
        return []

    variants = []
    for scale in scales:
        # Rebuild expression with all windows scaled simultaneously.
        # Work right-to-left to keep positions valid.
        chars = list(expression)
        for (start, end, val) in reversed(windows):
            new_val = str(_scale_window(val, scale))
            chars[start:end] = list(new_val)
        perturbed = "".join(chars)
        if perturbed != expression:
            variants.append(perturbed)

    # Deduplicate while preserving order
    seen: set[str] = set()
    result = []
    for v in variants:
        if v not in seen:
            seen.add(v)
            result.append(v)
    return result


def build_ast_mutation_hint(
    best_expression: str,
    factor_name: str = "",
    scales: Sequence[float] = (0.5, 2.0),
) -> str:
    """
    Build a prompt hint string describing window-perturbed variants of *best_expression*.

    Intended for inclusion in the mutation-round ``strategy_suffix`` so the LLM
    knows which expression variants have not yet been explored.

    Returns an empty string if no window parameters exist in the expression.
    """
    variants = perturb_expression_windows(best_expression, scales=scales)
    if not variants:
        return ""

    label = f" for `{factor_name}`" if factor_name else ""
    lines = [
        f"\n### AST Window-Parameter Variants{label}",
        f"Base expression: `{best_expression}`",
        "The following time-horizon variants have NOT yet been evaluated — "
        "consider them as starting points:\n",
    ]
    for v in variants:
        lines.append(f"  - `{v}`")
    lines.append(
        "\nIf you adopt one of these, adjust the factor name to reflect the new window size."
    )
    return "\n".join(lines)
