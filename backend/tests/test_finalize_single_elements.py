import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "06_mesh_elaboration"))
import finalize_single_elements as step4


def test_step_four_fits_wall_contours_before_finalizing_ceilings(monkeypatch, tmp_path):
    events = []

    def fit_walls(workspace):
        events.append(("walls", workspace))
        return {"output_ply": str(tmp_path / "fitted_walls.ply")}

    def fit_ceilings(workspace, *, wall_support):
        events.append(("ceilings", workspace, wall_support))
        return {"output_ply": "ceilings.ply", "manifest": "ceilings.json"}

    monkeypatch.setattr(step4, "fit_enclosing_walls_to_regularized_contours", fit_walls)
    monkeypatch.setattr(step4, "regularize_ceiling_planes", fit_ceilings)

    result = step4.finalize_single_elements(tmp_path)

    assert events == [
        ("walls", tmp_path),
        ("ceilings", tmp_path, tmp_path / "fitted_walls.ply"),
    ]
    assert result["ceiling"]["output_ply"] == "ceilings.ply"
