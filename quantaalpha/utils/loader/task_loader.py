import json
from pathlib import Path
from typing import Sequence

from quantaalpha.factors.coder.factor import FactorTask
from quantaalpha.contrib.model.coder.model import ModelFBWorkspace, ModelTask
from quantaalpha.core.experiment import Loader, WsLoader


class FactorTaskLoader(Loader[FactorTask]):
    pass


class ModelTaskLoader(Loader[ModelTask]):
    pass


class ModelTaskLoaderJson(ModelTaskLoader):

    def __init__(self, json_uri: str) -> None:
        super().__init__()
        self.json_uri = json_uri

    def load(self, *argT, **kwargs) -> Sequence[ModelTask]:
        # json is supposed to be in the format of {model_name: dict{model_data}}
        model_dict = json.load(open(self.json_uri, "r"))
        model_impl_task_list = []
        for model_name, model_data in model_dict.items():
            model_impl_task = ModelTask(
                name=model_name,
                description=model_data["description"],
                formulation=model_data["formulation"],
                variables=model_data["variables"],
                model_type=model_data["model_type"],
            )
            model_impl_task_list.append(model_impl_task)
        return model_impl_task_list


class ModelWsLoader(WsLoader[ModelTask, ModelFBWorkspace]):
    def __init__(self, path: Path) -> None:
        self.path = Path(path)

    def load(self, task: ModelTask) -> ModelFBWorkspace:
        assert task.name is not None
        mti = ModelFBWorkspace(task)
        mti.prepare()
        with open(self.path / f"{task.name}.py", "r") as f:
            code = f.read()
        mti.inject_code(**{"model.py": code})
        return mti
