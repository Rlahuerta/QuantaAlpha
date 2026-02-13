"""
QuantaAlpha custom workspace.

Overrides rdagent QlibFBWorkspace: project-level factor_template overrides default YAML;
base files (read_exp_res.py, etc.) still from rdagent; init empty git repo in workspace to suppress qlib recorder git output.
"""

import subprocess
import os
from pathlib import Path

from rdagent.scenarios.qlib.experiment.workspace import QlibFBWorkspace as _RdagentQlibFBWorkspace
from rdagent.log import rdagent_logger as logger

_CUSTOM_TEMPLATE_DIR = Path(__file__).resolve().parent / "factor_template"


class QlibFBWorkspace(_RdagentQlibFBWorkspace):
    """
    Override rdagent QlibFBWorkspace: inject project factor_template/ YAML over defaults;
    init empty git repo in workspace to avoid qlib recorder git help output.
    """

    def __init__(self, template_folder_path: Path, *args, **kwargs) -> None:
        super().__init__(template_folder_path, *args, **kwargs)
        if _CUSTOM_TEMPLATE_DIR.exists():
            self.inject_code_from_folder(_CUSTOM_TEMPLATE_DIR)
            logger.info(f"Overrode rdagent default config with project template: {_CUSTOM_TEMPLATE_DIR}")

    def before_execute(self) -> None:
        """Init workspace prerequisites before qlib backtest execution."""
        super().before_execute()

        # Ensure qlib default provider path exists when mining is launched without run.sh.
        qlib_data_dir = os.environ.get("QLIB_DATA_DIR") or os.environ.get("QLIB_PROVIDER_URI")
        if qlib_data_dir:
            source = Path(qlib_data_dir).expanduser().resolve()
            target = Path.home() / ".qlib" / "qlib_data" / "cn_data"
            if source.exists():
                target.parent.mkdir(parents=True, exist_ok=True)
                if target.is_symlink():
                    if target.resolve() != source:
                        target.unlink()
                        target.symlink_to(source)
                elif not target.exists():
                    target.symlink_to(source)
                elif target.resolve() != source:
                    logger.warning(f"qlib symlink target differs and is not a symlink: {target}")
            else:
                logger.warning(f"QLIB data path does not exist: {source}")

        git_dir = self.workspace_path / ".git"
        if not git_dir.exists():
            try:
                subprocess.run(
                    ["git", "init"],
                    cwd=str(self.workspace_path),
                    capture_output=True,
                    timeout=5,
                )
            except Exception:
                pass
