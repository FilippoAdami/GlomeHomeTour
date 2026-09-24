#!/usr/bin/env python3
"""Step 1 -- per-image quality gate (blur / exposure / texture).

In:  ``<workspace>/images/`` + ``transforms.json``.
Out: survivors in place; rejects moved to ``<workspace>/discarded/``.

This is stage 1 of the old three-stage filter and nothing more. The parallax and
keyframe stages moved to step 3, where real COLMAP geometry replaces their
estimated depths. ``QualityGate`` itself is reused unchanged -- its thresholds
are tuned against real captures (see ``00_ingestion/project_history.md``).

    python 00_ingestion/step_filter_quality.py [--workspace DIR]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

_backend_dir = Path(__file__).resolve().parents[1]
if str(_backend_dir) not in sys.path:
    sys.path.insert(0, str(_backend_dir))
from Utilities.pipeline_paths import bootstrap

bootstrap()

from Utilities.pipeline_step import StepContext, is_done
from Utilities.scene_io import load_scene, merge_back, split_scene
from quality_gate import QualityGate

DEFAULT_WORKSPACE = _backend_dir / "current_scene"
STAGE_DIRNAME = "00_ingestion"
DISCARD_DIRNAME = "discarded"


def filter_quality(workspace: Path, stage_dir: Path, ctx: StepContext,
                    gate: QualityGate | None = None) -> None:
    gate = gate or QualityGate()
    scene = load_scene(stage_dir, images_root=workspace)
    keyframes = scene.keyframes()
    total = len(keyframes)
    ctx.note(f"Evaluating {total} frames")

    with ctx.timer("evaluate"):
        result = gate.evaluate(keyframes)

    reject_names = [Path(keyframes[i].file_path).name for i in result.discarded_indices]
    kept, rejected = split_scene(stage_dir, stage_dir / DISCARD_DIRNAME, reject_names,
                                  images_root=workspace)

    pct = 100.0 * kept / max(1, total)
    ctx.metric("total_in", total)
    ctx.metric("frames_processed", total)
    ctx.metric("kept", kept)
    ctx.metric("discarded", rejected)
    ctx.metric("kept_pct", round(pct, 1))
    ctx.metric("reasons", {
        "blur": result.summary["rejected_blur"],
        "exposure": result.summary["rejected_exposure"],
        "texture": result.summary["rejected_texture"],
    })

    ctx.note(f"Kept {kept}/{total} ({pct:.1f}%), discarded {rejected}")
    ctx.note(f"  blur={result.summary['rejected_blur']} "
             f"exposure={result.summary['rejected_exposure']} "
             f"texture={result.summary['rejected_texture']}")

    # Per-frame reason and how far under threshold it landed. Reasons live here,
    # in stats, never in the discard transforms.json -- the schema is frozen with
    # additionalProperties:false and would reject a `reason` field.
    per_frame = {}
    margins: dict[str, list[float]] = {"blur": [], "exposure": [], "texture": []}
    for idx in result.discarded_indices:
        m = result.metrics[idx]
        name = Path(m.file_path).name
        if m.rejection_reason == "blur":
            value, threshold = m.relative_sharpness, gate.relative_blur_threshold
        elif m.rejection_reason == "texture":
            value, threshold = m.texture_score, gate.min_texture
        else:  # exposure -- report the axis that actually tripped
            if m.saturated_fraction > gate.max_saturated_fraction:
                value, threshold = m.saturated_fraction, gate.max_saturated_fraction
            elif m.black_fraction > gate.max_black_fraction:
                value, threshold = m.black_fraction, gate.max_black_fraction
            elif m.mean_luminance < gate.dark_threshold:
                value, threshold = m.mean_luminance, gate.dark_threshold
            else:
                value, threshold = m.mean_luminance, gate.blown_threshold
        per_frame[name] = {
            "reason": m.rejection_reason,
            "value": round(float(value), 4),
            "threshold": round(float(threshold), 4),
            "margin": round(float(value - threshold), 4),
        }
        margins[m.rejection_reason].append(float(value - threshold))

    ctx.metric("discarded_frames", per_frame)
    # Save per-frame blur scores for downstream sharpness-aware keyframe selection
    ctx.metric("per_frame_sharpness", {
        Path(m.file_path).name: round(float(m.blur_score), 2)
        for m in result.metrics
    })
    for reason, values in margins.items():
        if values:
            ctx.metric(f"margin_{reason}", {
                "mean": round(float(np.mean(values)), 4),
                "worst": round(float(np.min(values)), 4),
            })
            ctx.note(f"  {reason}: mean margin {np.mean(values):+.3f}, worst {np.min(values):+.3f}")

    # The reject cap re-accepts the sharpest blur-rejects when the cull is
    # implausibly large. It mutates flags before metrics are built, so infer it:
    # an accepted frame scoring under the relative blur threshold was re-accepted.
    capped = sum(1 for i in result.accepted_indices
                 if result.metrics[i].relative_sharpness < gate.relative_blur_threshold)
    ctx.metric("reject_cap_reaccepted", capped)
    if capped:
        ctx.note(f"  reject cap fired: {capped} frame(s) re-accepted "
                 f"(cull would have exceeded {gate.max_reject_fraction:.0%})")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--workspace", default=str(DEFAULT_WORKSPACE))
    parser.add_argument("--force", action="store_true", help="Re-run even if already complete")
    args = parser.parse_args(argv)

    workspace = Path(args.workspace)
    stage_dir = workspace / STAGE_DIRNAME
    if not args.force and is_done(workspace, "filter_quality", [stage_dir / "transforms.json"]):
        print("[filter_quality] already done, skipping (use --force to re-run)")
        return 0

    if args.force:
        # Idempotence: put the previous run's rejects back before re-judging them.
        restored = merge_back(stage_dir / DISCARD_DIRNAME, stage_dir, images_root=workspace)
        if restored:
            print(f"[filter_quality] --force: restored {restored} previously discarded frame(s)")

    with StepContext("filter_quality", workspace, artifacts_dir=stage_dir) as ctx:
        filter_quality(workspace, stage_dir, ctx)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
