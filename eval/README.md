# Evaluation Tools

Use this folder to aggregate and visualize metrics after running scripts under `scripts/`.

## 1) Build metrics table

```bash
python eval/compute_metrics.py --results-root results --datasets 00 01 --output-dir eval
```

Outputs:
- `eval/metrics_summary.csv`
- `eval/metrics_summary.md`

## 2) Build report assets (plots + markdown)

```bash
python eval/build_report_assets.py \
  --metrics-csv eval/metrics_summary.csv \
  --output-dir eval \
  --output-md eval/report_assets.md
```

Outputs:
- `eval/plot_part1_path_length.png`
- `eval/plot_part3_landmarks_initialized.png`
- `eval/plot_part4_endpoint_improvement.png`
- `eval/plot_part4_pose_residual.png`
- `eval/report_assets.md`
