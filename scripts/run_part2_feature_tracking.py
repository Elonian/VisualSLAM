#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts._common import add_common_args, add_part2_args, tracking_cfg_from_args
from visual_slam.pipeline import run_part2


def main() -> None:
    parser = argparse.ArgumentParser(description='Run Project 3 Part 2: feature tracking (optional extra credit).')
    add_common_args(parser)
    add_part2_args(parser)
    args = parser.parse_args()

    tcfg = tracking_cfg_from_args(args)
    had_error = False
    for ds in args.datasets:
        try:
            out = run_part2(args.data_dir, args.output_root, ds, cfg=tcfg)
            print(f'[Part2] Dataset {out.dataset_id} complete -> {out.output_dir / "part2_feature_tracks.npz"}')
        except Exception as exc:
            had_error = True
            print(f'[Part2] Dataset {ds} failed: {exc}')

    if had_error:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
