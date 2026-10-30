from pathlib import Path
import sys

backend = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(backend))
from Utilities.pipeline_paths import bootstrap
bootstrap()
import export_keyframes
import step_colmap


def test_colmap_subprocess_preserves_path_and_sets_qt_platform(monkeypatch):
    monkeypatch.setenv("PATH", "/custom/colmap/bin")
    captured = {}

    def fake_run(command, *, check, env):
        captured.update(command=command, check=check, env=env)

    monkeypatch.setattr(step_colmap.subprocess, "run", fake_run)

    step_colmap._run_colmap(["colmap", "help"])

    assert captured["env"]["PATH"] == "/custom/colmap/bin"
    assert captured["env"]["QT_QPA_PLATFORM"] == "xcb"
    assert captured["check"] is True


def test_mapper_uses_only_images_in_the_feature_database(tmp_path, monkeypatch):
    images = tmp_path / "images"
    images.mkdir()
    frame = images / "frame.jpg"
    frame.touch()
    work = tmp_path / "work"
    work.mkdir()
    commands = []

    def fake_run_colmap(command):
        commands.append(command)
        if command[1] == "mapper":
            output = Path(command[command.index("--output_path") + 1])
            assert output.is_dir()
            (output / "0").mkdir()

    def fake_write_model_text(_model_path, output_path):
        output_path.mkdir(parents=True)
        (output_path / "images.txt").write_text(
            "1 1 0 0 0 0 0 0 1 frame.jpg\n\n", encoding="utf-8")

    monkeypatch.setattr(step_colmap, "_run_colmap", fake_run_colmap)
    monkeypatch.setattr(step_colmap, "write_model_text", fake_write_model_text)

    result, low_sift = step_colmap.run_image_only_colmap(images, [frame], work, "sequential")

    mapper = next(command for command in commands if command[1] == "mapper")
    assert "--image_list_path" not in mapper
    assert result == work / "text_models" / "0"
    assert low_sift == set()



def test_model_pruning_preserves_colmap_path(tmp_path, monkeypatch):
    monkeypatch.setenv("PATH", "/custom/colmap/bin")
    sparse = tmp_path / "sparse"
    sparse.mkdir()
    (sparse / "images.txt").write_text(
        "1 1 0 0 0 0 0 0 1 frame.jpg\n\n", encoding="utf-8")
    captured = []

    def fake_run(command, *, check, stdout, env):
        captured.append((command, check, env))

    monkeypatch.setattr(export_keyframes.subprocess, "run", fake_run)

    export_keyframes.prune_model(str(sparse), {"frame.jpg"})

    assert len(captured) == 2
    assert all(env["PATH"] == "/custom/colmap/bin" for _, _, env in captured)
    assert all(env["QT_QPA_PLATFORM"] == "xcb" for _, _, env in captured)
