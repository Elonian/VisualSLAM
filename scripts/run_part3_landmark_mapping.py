#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts._common import add_common_args, pipeline_cfg_from_args, resolve_datasets
from visual_slam.pipeline import run_part3
from visual_slam.utils import describe_gpu


def main() -> None:
    parser = argparse.ArgumentParser(description='Run Project 3 Part 3: landmark mapping via EKF updates.')
    add_common_args(parser)
    parser.add_argument(
        '--track-file',
        type=Path,
        default=None,
        help='Optional .npz feature track file from Part 2. If omitted, dataset features are used.',
    )
    args = parser.parse_args()

    cfg = pipeline_cfg_from_args(args)
    gpu = describe_gpu(cfg.runtime.use_gpu)
    print(f'[Runtime] {gpu.message}')

    had_error = False
    for ds in resolve_datasets(args):
        try:
            out = run_part3(args.data_dir, args.output_root, ds, cfg, track_file=args.track_file)
            print(f'[Part3] Dataset {out.dataset_id} complete -> {out.output_dir / "part3_landmark_mapping.npz"}')
        except Exception as exc:
            had_error = True
            print(f'[Part3] Dataset {ds} failed: {exc}')

    if had_error:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
