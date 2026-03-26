#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts._common import add_common_args, pipeline_cfg_from_args
from visual_slam.pipeline import run_pipeline
from visual_slam.utils import describe_gpu


def main() -> None:
    parser = argparse.ArgumentParser(description='Run all implemented Project 3 parts end-to-end.')
    add_common_args(parser)
    parser.add_argument('--max-part', type=int, default=4, choices=[1, 2, 3, 4], help='Run up to this part.')
    parser.add_argument(
        '--run-part2-if-missing',
        action='store_true',
        help='If feature tracks are missing and images are available, run Part 2 automatically.',
    )
    args = parser.parse_args()

    cfg = pipeline_cfg_from_args(args)
    gpu = describe_gpu(cfg.runtime.use_gpu)
    print(f'[Runtime] {gpu.message}')

    had_error = False
    for ds in args.datasets:
        try:
            out = run_pipeline(
                data_dir=args.data_dir,
                output_root=args.output_root,
                dataset=ds,
                cfg=cfg,
                max_part=args.max_part,
                run_part2_if_missing=bool(args.run_part2_if_missing),
            )
            print(f'[All] Dataset {out.dataset_id} complete -> {out.output_dir}')
        except Exception as exc:
            had_error = True
            print(f'[All] Dataset {ds} failed: {exc}')

    if had_error:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
