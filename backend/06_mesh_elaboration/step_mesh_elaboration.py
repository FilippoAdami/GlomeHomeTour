#!/usr/bin/env python3
"""Run structural extraction, fitting, regularization and level reporting in order."""
from __future__ import annotations
import argparse
from pathlib import Path
import runpy
import traceback
import sys
import time
import json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mesh-path', type=Path, required=True)
    parser.add_argument('--labels-path', type=Path, required=True)
    parser.add_argument('--structural-json', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--scene-size-file', type=Path)
    args = parser.parse_args()
    source = Path(__file__).resolve().parent
    output = args.output_dir.absolute()
    stages = [
        ('extract_structural_meshes', ['--mesh-path', str(args.mesh_path.absolute()), '--labels-path', str(args.labels_path.absolute()),
                                     '--structural-json', str(args.structural_json.absolute()), '--output-dir', str(output)]),
        ('fit_wall_planes', ['--instances-dir', str(output), '--output-dir', str(output)]),
        ('fit_floors_ceilings_planes', ['--instances-dir', str(output), '--wall-manifest', str(output/'wall_planes_manifest.json')]),
        ('recognize_and_regularize_patterns', ['--instances-dir', str(output), '--output-dir', str(output), '--scene-mesh', str(args.mesh_path.absolute())]),
        ('group_floors_ceilings_levels', ['--instances-dir', str(output), '--output-dir', str(output)]),
    ]
    if args.scene_size_file:
        stages[-1][1].extend(['--scene-size-file', str(args.scene_size_file.absolute())])
    timings = []
    for name, flags in stages:
        start = time.perf_counter()
        previous_argv = sys.argv
        sys.argv = [str(source/f'{name}.py'), *flags]
        code = 0
        try:
            runpy.run_path(sys.argv[0], run_name='__main__')
        except SystemExit as result:
            code = result.code if isinstance(result.code, int) else 0 if result.code is None else 1
        except Exception:
            traceback.print_exc()
            code = 1
        finally:
            sys.argv = previous_argv
        timings.append({'stage': name, 'wall_seconds': time.perf_counter()-start, 'exit_code': code})
        if output.exists():
            (output/'step06_timings.json').write_text(json.dumps(timings, indent=2)+'\n')
        if code:
            return code
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
