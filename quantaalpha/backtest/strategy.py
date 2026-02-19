"""Signal-strength weighted portfolio strategy for Qlib backtest.

Extends WeightStrategyBase to allocate capital proportional to prediction
magnitude rather than equal weights. High-confidence signals receive larger
allocations; turnover is controlled by the n_drop parameter (same semantics
as TopkDropoutStrategy).

**Important**: This strategy inherits from WeightStrategyBase, which rebalances
to the exact target weight every trading day. Because cross-sectional signal ranks
shift daily, all 50 positions see their target weights change slightly (~0.5-1%/day),
generating ~18% daily portfolio turnover and ~8-9%/year cost drag at standard
Chinese A-share transaction costs (0.05% open + 0.15% close). This erases the
benefit of signal-weighted allocation relative to the equal-weight TopkDropoutStrategy.

For profitable use, pair with:
  - Weekly rebalancing frequency (rebalance only on every 5th trading day)
  - Or a "membership-change-only" variant that fixes weights until stocks exit

For daily execution use TopkDropoutStrategy (equal weight) which achieves much
lower costs (~0.19%/year) while still capturing the selection signal.
"""
import copy
from typing import Dict, List

import numpy as np
import pandas as pd
from qlib.contrib.strategy.signal_strategy import WeightStrategyBase


class SignalWeightedTopkStrategy(WeightStrategyBase):
    """Top-k selection with prediction-strength proportional weights.

    Args:
        topk (int): Number of stocks in portfolio.
        n_drop (int): Max stocks replaced per trading day (turnover control).
        weight_scheme (str): One of:
            - 'linear_rank'  (default) – weight ∝ cross-sectional rank (1..k)
            - 'softmax'      – weight ∝ exp(score) after max-shift
            - 'score'        – weight ∝ score shifted to be non-negative
            - 'equal'        – equal weight (equivalent to TopkDropoutStrategy
                               but without its hold_thresh / method_sell logic)
    """

    def __init__(
        self,
        *,
        topk: int,
        n_drop: int,
        weight_scheme: str = "linear_rank",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.topk = topk
        self.n_drop = n_drop
        self.weight_scheme = weight_scheme

    def generate_target_weight_position(
        self, score, current, trade_start_time, trade_end_time
    ) -> Dict[str, float]:
        """Return {stock_id: weight} for the target portfolio on this date."""
        if isinstance(score, pd.DataFrame):
            score = score.iloc[:, 0]
        score = score.dropna()
        if score.empty:
            return {}

        current_stocks: List[str] = current.get_stock_list()
        current_set = set(current_stocks)

        # Full desired top-k by today's signal
        all_sorted = score.sort_values(ascending=False)
        desired_top_k = set(all_sorted.index[: self.topk])

        # Apply n_drop turnover constraint
        if current_set:
            # Sell worst-scoring current stocks that fell out of desired_top_k
            to_remove_candidates = sorted(
                current_set - desired_top_k,
                key=lambda s: score.get(s, -np.inf),  # ascending: remove worst first
            )
            to_remove = set(to_remove_candidates[: self.n_drop])

            # Buy top new candidates not already held
            to_add_candidates = sorted(
                desired_top_k - current_set,
                key=lambda s: score.get(s, -np.inf),
                reverse=True,  # descending: add best first
            )
            # Symmetric replacement: add as many as we remove
            to_add = set(to_add_candidates[: len(to_remove)])

            portfolio = (current_set - to_remove) | to_add

            # Pad to topk on the first day or after gaps
            if len(portfolio) < self.topk:
                extra = [s for s in all_sorted.index if s not in portfolio]
                portfolio |= set(extra[: self.topk - len(portfolio)])
        else:
            # Initialisation: take straight top-k
            portfolio = desired_top_k

        # Compute per-stock weights
        portfolio_list = list(portfolio)
        portfolio_scores = score.reindex(portfolio_list).fillna(score.min())

        if self.weight_scheme == "linear_rank":
            ranks = portfolio_scores.rank(ascending=True, method="average")
            weights = ranks / ranks.sum()
        elif self.weight_scheme == "softmax":
            arr = portfolio_scores.values.astype(float)
            arr -= arr.max()  # numerical stability
            exp_w = np.exp(arr)
            weights = pd.Series(exp_w / exp_w.sum(), index=portfolio_scores.index)
        elif self.weight_scheme == "score":
            shifted = portfolio_scores - portfolio_scores.min() + 1e-8
            weights = shifted / shifted.sum()
        else:  # 'equal'
            weights = pd.Series(
                np.ones(len(portfolio_list)) / len(portfolio_list),
                index=portfolio_list,
            )

        return dict(zip(weights.index, weights.values))
