from pathlib import Path

import pytest

import quantaalpha.core.experiment as exp_module
from quantaalpha.core.experiment import Experiment, FBWorkspace, Loader, Task, Workspace, WsLoader


class _DummyTask(Task):
    def get_task_information(self) -> str:
        return f"{self.name}-{self.version}"


class _BaseWorkspace(Workspace):
    def execute(self, *args, **kwargs):
        return super().execute(*args, **kwargs)

    def copy(self):
        return super().copy()


class _BaseWsLoader(WsLoader):
    def load(self, task):
        return super().load(task)


class _BaseLoader(Loader):
    def load(self, *args, **kwargs):
        return super().load(*args, **kwargs)


def test_abstract_base_methods_raise_not_implemented():
    task = _DummyTask(name="task-a", version=2)
    ws = _BaseWorkspace(target_task=task)
    assert task.get_task_information() == "task-a-2"
    assert ws.target_task is task

    with pytest.raises(NotImplementedError):
        ws.execute()
    with pytest.raises(NotImplementedError):
        ws.copy()
    with pytest.raises(NotImplementedError):
        _BaseWsLoader().load(task)
    with pytest.raises(NotImplementedError):
        _BaseLoader().load(task)


def test_fbworkspace_prepare_inject_copy_execute_clear_and_str(monkeypatch, tmp_path):
    monkeypatch.setattr(exp_module.RD_AGENT_SETTINGS, "workspace_path", tmp_path)
    ws = FBWorkspace(target_task=_DummyTask("main-task"))

    ws.prepare()
    assert ws.workspace_path.exists()

    ws.inject_code(**{"code.py": "print('ok')", "nested/conf.yaml": "k: v"})
    assert (ws.workspace_path / "code.py").exists()
    assert (ws.workspace_path / "nested" / "conf.yaml").exists()
    assert "File: code.py" in ws.code
    assert "File: nested/conf.yaml" in ws.code
    assert len(ws.get_files()) >= 2

    ws_copy = ws.copy()
    assert ws_copy is not ws
    assert ws_copy.code_dict == ws.code_dict

    assert ws.execute() is None
    assert "target_task.name='main-task'" in str(ws)

    ws_without_task = FBWorkspace()
    assert "workspace_path" in str(ws_without_task)
    assert "target_task.name" not in str(ws_without_task)

    ws.clear()
    assert ws.code_dict == {}
    assert not ws.workspace_path.exists()


def test_link_all_files_and_inject_from_folder(monkeypatch, tmp_path):
    monkeypatch.setattr(exp_module.RD_AGENT_SETTINGS, "workspace_path", tmp_path)
    data_path = tmp_path / "data_src"
    linux_ws = tmp_path / "linux_ws"
    windows_ws = tmp_path / "windows_ws"
    load_folder = tmp_path / "load_src"
    data_path.mkdir()
    linux_ws.mkdir()
    windows_ws.mkdir()
    load_folder.mkdir()

    (data_path / "a.txt").write_text("A", encoding="utf-8")
    (data_path / "b.txt").write_text("B", encoding="utf-8")
    (linux_ws / "a.txt").write_text("old", encoding="utf-8")

    monkeypatch.setattr(exp_module.platform, "system", lambda: "Linux")
    FBWorkspace.link_all_files_in_folder_to_workspace(data_path, linux_ws)
    assert (linux_ws / "a.txt").is_symlink()
    assert (linux_ws / "b.txt").is_symlink()

    linked_pairs = []

    def _fake_link(src, dst):
        linked_pairs.append((src, dst))
        Path(dst).write_text(Path(src).read_text(encoding="utf-8"), encoding="utf-8")

    monkeypatch.setattr(exp_module.platform, "system", lambda: "Windows")
    monkeypatch.setattr(exp_module.os, "link", _fake_link)
    FBWorkspace.link_all_files_in_folder_to_workspace(data_path, windows_ws)
    assert len(linked_pairs) == 2
    assert (windows_ws / "a.txt").read_text(encoding="utf-8") == "A"

    (load_folder / "x.py").write_text("x=1", encoding="utf-8")
    (load_folder / "y.yaml").write_text("y: 1", encoding="utf-8")
    (load_folder / "z.md").write_text("# z", encoding="utf-8")
    (load_folder / "ignore.txt").write_text("ignore", encoding="utf-8")
    (load_folder / "sub").mkdir()
    (load_folder / "sub" / "n.py").write_text("n=1", encoding="utf-8")

    ws = FBWorkspace(target_task=_DummyTask("loader-task"))
    ws.inject_code_from_folder(load_folder)
    assert "x.py" in ws.code_dict
    assert "y.yaml" in ws.code_dict
    assert "z.md" in ws.code_dict
    assert "sub/n.py" in ws.code_dict
    assert "ignore.txt" not in ws.code_dict


def test_experiment_initialization_defaults(monkeypatch, tmp_path):
    monkeypatch.setattr(exp_module.RD_AGENT_SETTINGS, "workspace_path", tmp_path)
    tasks = [_DummyTask("t1"), _DummyTask("t2")]
    based = [FBWorkspace(target_task=_DummyTask("base"))]

    exp = Experiment(sub_tasks=tasks, based_experiments=based)

    assert exp.sub_tasks == tasks
    assert exp.sub_workspace_list == [None, None]
    assert exp.based_experiments == based
    assert exp.result is None
    assert exp.sub_results == {}
    assert exp.experiment_workspace is None
