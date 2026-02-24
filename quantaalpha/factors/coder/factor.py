from __future__ import annotations

import json
import subprocess
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, Union, Optional

import pandas as pd
from filelock import FileLock

from quantaalpha.coder.costeer.task import CoSTEERTask
from quantaalpha.factors.coder.config import FACTOR_COSTEER_SETTINGS
from quantaalpha.core.exception import CodeFormatError, CustomRuntimeError, NoOutputError
from quantaalpha.core.experiment import Experiment, FBWorkspace
from quantaalpha.core.utils import cache_with_pickle
from quantaalpha.llm.client import md5_hash


@dataclass
class FactorExecutionResult:
    """
    Result of factor code execution.

    Attributes:
        success: Whether execution completed successfully
        feedback: Execution output/error message
        result: Factor value DataFrame (None if failed)
        error: Exception object if execution failed (None if succeeded)
    """
    success: bool
    feedback: str
    result: Optional[pd.DataFrame]
    error: Optional[Exception] = None


class FactorTask(CoSTEERTask):
    # TODO:  generalized the attributes into the Task
    # - factor_* -> *
    def __init__(
        self,
        factor_name,
        factor_description,
        factor_formulation,
        factor_expression = None,
        *args,
        variables: dict = None,
        resource: str = None,
        factor_implementation: bool = False,
        **kwargs,
    ) -> None:
        if variables is None:
            variables = {}
        self.factor_name = (
            factor_name  # TODO: remove it in the later version. Keep it only for pickle version compatibility
        )
        self.factor_description = factor_description
        self.factor_formulation = factor_formulation
        self.factor_expression = factor_expression
        self.variables = variables
        self.factor_resources = resource
        self.factor_implementation = factor_implementation
        super().__init__(name=factor_name, *args, **kwargs)

    def get_task_information(self):
        return f"""factor_name: {self.factor_name}
factor_description: {self.factor_description}
factor_formulation: {self.factor_formulation}
variables: {str(self.variables)}"""
    

    def get_task_description(self):
        return f"""factor_name: {self.factor_name}
factor_description: {self.factor_description}"""

    def get_task_information_and_implementation_result(self):
        return {
            "factor_name": self.factor_name,
            "factor_description": self.factor_description,
            "factor_formulation": self.factor_formulation,
            "factor_expression": self.factor_expression,
            "variables": str(self.variables),
            "factor_implementation": str(self.factor_implementation),
        }

    @staticmethod
    def from_dict(dict):
        return FactorTask(**dict)

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}[{self.factor_name}]>"


class FactorFBWorkspace(FBWorkspace):
    """
    This class is used to implement a factor by writing the code to a file.
    Input data and output factor value are also written to files.
    """

    # TODO: (Xiao) think raising errors may get better information for processing
    FB_EXEC_SUCCESS = "Execution succeeded without error."
    FB_CODE_NOT_SET = "code is not set."
    FB_EXECUTION_SUCCEEDED = "Execution succeeded without error."
    FB_OUTPUT_FILE_NOT_FOUND = "\nExpected output file not found."
    FB_OUTPUT_FILE_FOUND = "\nExpected output file found."

    def __init__(
        self,
        *args,
        raise_exception: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.raise_exception = raise_exception

    def hash_func(self, data_type: str = "Debug") -> str:
        """
        Generate a cache key that uniquely identifies this factor execution.

        Includes:
        - data_type: Debug, Train, etc.
        - code: The factor.py source code
        - factor_name: Unique identifier for the factor
        - variables: Any custom variables (if present)

        This prevents cache collisions between different factors with identical code.
        """
        if "factor.py" not in self.code_dict or self.raise_exception:
            return None

        content = {
            "data_type": data_type,
            "code": self.code_dict["factor.py"],
            "factor_name": self.target_task.factor_name,
        }

        # Include variables if present to differentiate factors with same code but different data
        if hasattr(self.target_task, 'variables') and self.target_task.variables:
            content["variables"] = str(sorted(self.target_task.variables.items()))

        return md5_hash(json.dumps(content, sort_keys=True))

    @cache_with_pickle(hash_func)
    def execute(self, data_type: str = "Debug") -> FactorExecutionResult:
        """
        Execute the implementation and get the factor value by the following steps:
        1. make the directory in workspace path
        2. write the code to the file in the workspace path
        3. link all the source data to the workspace path folder
        if call_factor_py is True:
            4. execute the code
        else:
            4. generate a script from template to import the factor.py dump get the factor value to result.h5
        5. read the factor value from the output file in the workspace path folder

        Returns:
            FactorExecutionResult with success status, feedback, result DataFrame, and optional error

        Regarding the cache mechanism:
        1. We will store the function's return value to ensure it behaves as expected.
        - The cached information will include a FactorExecutionResult dataclass
        """
        super().execute()
        if self.code_dict is None or "factor.py" not in self.code_dict:
            if self.raise_exception:
                raise CodeFormatError(self.FB_CODE_NOT_SET)
            else:
                return FactorExecutionResult(
                    success=False,
                    feedback=self.FB_CODE_NOT_SET,
                    result=None,
                    error=CodeFormatError(self.FB_CODE_NOT_SET)
                )
        with FileLock(self.workspace_path / "execution.lock"):
            # Set data path for all versions
            source_data_path = (
                Path(
                    FACTOR_COSTEER_SETTINGS.data_folder_debug,
                )
                if data_type == "Debug"  # FIXME: (yx) don't think we should use a debug tag for this.
                else Path(
                    FACTOR_COSTEER_SETTINGS.data_folder,
                )
            )

            # Use absolute path
            if not source_data_path.is_absolute():
                source_data_path = self.workspace_path.parent.parent.parent / source_data_path
            else:
                source_data_path = Path(source_data_path).absolute()

            source_data_path.mkdir(exist_ok=True, parents=True)
            code_path = self.workspace_path / f"factor.py"

            # Ensure data path exists and has files
            if source_data_path.exists() and any(source_data_path.iterdir()):
                self.link_all_files_in_folder_to_workspace(source_data_path, self.workspace_path)
            else:
                from quantaalpha.log import logger
                logger.warning(f"Data folder {source_data_path} does not exist or is empty. Skipping linking.")

            execution_feedback = self.FB_EXECUTION_SUCCEEDED
            execution_success = False
            execution_error = None

            if self.target_task.version == 1:
                execution_code_path = code_path
            elif self.target_task.version == 2:
                execution_code_path = self.workspace_path / f"{uuid.uuid4()}.py"
                execution_code_path.write_text((Path(__file__).parent / "factor_execution_template.txt").read_text())

            try:
                # Set PYTHONPATH to include the project root so quantaalpha can be imported
                import os
                import resource

                # Whitelist of safe environment variables to pass to subprocess
                SAFE_ENV_VARS = frozenset({
                    'PATH', 'HOME', 'USER', 'CONDA_PREFIX', 'PYTHONPATH',
                    'LD_LIBRARY_PATH', 'LANG', 'LC_ALL', 'TZ'
                })

                # Create restricted environment (avoid leaking secrets)
                env = {k: v for k, v in os.environ.items() if k in SAFE_ENV_VARS}
                project_root = Path(__file__).parent.parent.parent.parent
                env['PYTHONPATH'] = str(project_root)

                # Define resource limit function for Unix systems
                def set_limits():
                    # Limit memory to 2GB
                    resource.setrlimit(resource.RLIMIT_AS, (2 * 1024**3, 2 * 1024**3))
                    # Limit CPU time to timeout value
                    resource.setrlimit(resource.RLIMIT_CPU, (
                        FACTOR_COSTEER_SETTINGS.file_based_execution_timeout,
                        FACTOR_COSTEER_SETTINGS.file_based_execution_timeout
                    ))
                    # Disable core dumps (prevent sensitive data leakage)
                    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))

                subprocess.check_output(
                    [FACTOR_COSTEER_SETTINGS.python_bin, str(execution_code_path.absolute())],
                    shell=False,
                    cwd=self.workspace_path.absolute(),
                    stderr=subprocess.STDOUT,
                    timeout=FACTOR_COSTEER_SETTINGS.file_based_execution_timeout,
                    env=env,
                    preexec_fn=set_limits if os.name != 'nt' else None,
                )
                execution_success = True
            except subprocess.CalledProcessError as e:
                import site

                execution_feedback = (
                    e.output.decode()
                    .replace(str(execution_code_path.parent.absolute()), r"/path/to")
                    .replace(str(site.getsitepackages()[0]), r"/path/to/site-packages")
                )
                if len(execution_feedback) > 2000:
                    execution_feedback = (
                        execution_feedback[:1000] + "....hidden long error message...." + execution_feedback[-1000:]
                    )
                if self.raise_exception:
                    raise CustomRuntimeError(execution_feedback)
                else:
                    execution_error = CustomRuntimeError(execution_feedback)
            except subprocess.TimeoutExpired:
                execution_feedback += f"Execution timeout error and the timeout is set to {FACTOR_COSTEER_SETTINGS.file_based_execution_timeout} seconds."
                if self.raise_exception:
                    raise CustomRuntimeError(execution_feedback)
                else:
                    execution_error = CustomRuntimeError(execution_feedback)

            workspace_output_file_path = self.workspace_path / "result.h5"
            if workspace_output_file_path.exists() and execution_success:
                try:
                    executed_factor_value_dataframe = pd.read_hdf(workspace_output_file_path)
                    execution_feedback += self.FB_OUTPUT_FILE_FOUND
                except Exception as e:
                    execution_feedback += f"Error found when reading hdf file: {e}"[:1000]
                    executed_factor_value_dataframe = None
            else:
                execution_feedback += self.FB_OUTPUT_FILE_NOT_FOUND
                executed_factor_value_dataframe = None
                if self.raise_exception:
                    raise NoOutputError(execution_feedback)
                else:
                    execution_error = NoOutputError(execution_feedback)

        return FactorExecutionResult(
            success=execution_success and executed_factor_value_dataframe is not None,
            feedback=execution_feedback,
            result=executed_factor_value_dataframe,
            error=execution_error
        )

    def __str__(self) -> str:
        # NOTE:
        # If the code cache works, the workspace will be None.
        return f"File Factor[{self.target_task.factor_name}]: {self.workspace_path}"

    def __repr__(self) -> str:
        return self.__str__()

    @staticmethod
    def from_folder(task: FactorTask, path: Union[str, Path], **kwargs):
        path = Path(path)
        code_dict = {}
        for file_path in path.iterdir():
            if file_path.suffix == ".py":
                code_dict[file_path.name] = file_path.read_text()
        return FactorFBWorkspace(target_task=task, code_dict=code_dict, **kwargs)


FactorExperiment = Experiment
FeatureExperiment = Experiment
