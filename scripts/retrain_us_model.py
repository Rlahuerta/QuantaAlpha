#!/usr/bin/env python3
"""
scripts/retrain_us_model.py — Monthly US model retraining.

Loads the latest US factor library, runs the backtest training pipeline
(train split only), and saves a timestamped LightGBM model + updates
the meta_path in configs/live.yaml.

Usage:
    python scripts/retrain_us_model.py [--config configs/backtest_us.yaml]
                                       [--factor-json data/factorlib/all_factors_library_us_elite150.json]
                                       [--live-yaml configs/live.yaml]
                                       [--dry-run]
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Retrain US LightGBM model")
    p.add_argument("--config", default="configs/backtest_us.yaml",
                   help="Backtest config YAML path")
    p.add_argument("--factor-json", default=None,
                   help="Factor library JSON (default: from config)")
    p.add_argument("--live-yaml", default="configs/live.yaml",
                   help="Live trading config to update with new meta_path")
    p.add_argument("--model-dir", default="data/models",
                   help="Directory to save trained model artefacts")
    p.add_argument("--dry-run", action="store_true",
                   help="Load and validate factors only; skip actual training")
    p.add_argument("--verbose", "-v", action="store_true")
    return p.parse_args()


def _update_live_yaml(live_yaml_path: Path, meta_path: str) -> None:
    """Patch the meta_path entry in live.yaml in-place."""
    data = yaml.safe_load(live_yaml_path.read_text())
    data.setdefault("model", {})["meta_path"] = meta_path
    live_yaml_path.write_text(yaml.dump(data, default_flow_style=False, allow_unicode=True))
    log.info("Updated %s → model.meta_path = %s", live_yaml_path, meta_path)


def retrain(
    config_path: str | Path,
    factor_json: str | Path | None,
    model_dir: str | Path,
    live_yaml_path: str | Path | None = None,
    dry_run: bool = False,
) -> Path:
    """
    Run backtest training pipeline and save artefacts.

    Returns
    -------
    Path
        Path to the saved meta JSON (``{model_dir}/us_retrain_{timestamp}_meta.json``).
    """
    from quantaalpha.backtest.runner import BacktestRunner

    config_path = Path(config_path)
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    # Load backtest config
    cfg = yaml.safe_load(config_path.read_text())

    # Resolve factor JSON
    if factor_json is None:
        factor_json = cfg.get("factor_source", {}).get("custom", {}).get("factor_json")
    if not factor_json:
        raise ValueError("--factor-json not provided and not found in config")
    factor_json = Path(factor_json)
    if not factor_json.exists():
        raise FileNotFoundError(f"Factor library not found: {factor_json}")

    # Timestamp for this run
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"us_retrain_{ts}"

    # Inject model save paths into config
    cfg.setdefault("model", {})
    cfg["model"]["save_dir"] = str(model_dir)
    cfg["model"]["name"] = run_name

    log.info("Retraining model: %s", run_name)
    log.info("Factor library  : %s", factor_json)
    log.info("Config          : %s", config_path)
    log.info("Dry run         : %s", dry_run)

    runner = BacktestRunner(cfg)
    runner.run(
        factor_json=str(factor_json),
        factor_source="custom",
        output_name=run_name,
        dry_run=dry_run,
    )

    meta_path = model_dir / f"{run_name}_meta.json"

    if dry_run:
        log.info("Dry run — no model saved")
        # Write a placeholder meta so callers can still check the path
        meta_path.write_text(json.dumps({"dry_run": True, "run_name": run_name}))
    else:
        if not meta_path.exists():
            log.warning("Expected meta file not found at %s", meta_path)
        else:
            log.info("Model saved: %s", meta_path)
            # Optionally update live.yaml
            if live_yaml_path:
                _update_live_yaml(Path(live_yaml_path), str(meta_path))

    return meta_path


def main() -> int:
    args = _parse_args()
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    try:
        meta = retrain(
            config_path=args.config,
            factor_json=args.factor_json,
            model_dir=args.model_dir,
            live_yaml_path=args.live_yaml,
            dry_run=args.dry_run,
        )
        log.info("Retrain complete → %s", meta)
        return 0
    except Exception as exc:
        log.error("Retrain failed: %s", exc, exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
