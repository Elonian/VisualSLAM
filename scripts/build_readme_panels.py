#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results_refined"


@dataclass(frozen=True)
class DatasetAsset:
    slug: str
    label: str


@dataclass(frozen=True)
class PartAsset:
    key: str
    title: str
    gif_name: str
    png_name: str
    out_frames: int
    frame_duration_ms: int


DATASETS = (
    DatasetAsset("dataset_00", "Dataset 00"),
    DatasetAsset("dataset_01", "Dataset 01"),
    DatasetAsset("dataset_02", "Dataset 02"),
)

PARTS = (
    PartAsset(
        key="part1",
        title="Part 1  IMU Localization".replace("  ", " "),
        gif_name="part1_imu_trajectory.gif",
        png_name="part1_imu_trajectory.png",
        out_frames=72,
        frame_duration_ms=110,
    ),
    PartAsset(
        key="part2",
        title="Part 2  Stereo Feature Tracking".replace("  ", " "),
        gif_name="part2_feature_tracking.gif",
        png_name="part2_feature_stats.png",
        out_frames=120,
        frame_duration_ms=90,
    ),
    PartAsset(
        key="part3",
        title="Part 3  Landmark Mapping".replace("  ", " "),
        gif_name="part3_landmark_mapping.gif",
        png_name="part3_landmarks_xy.png",
        out_frames=100,
        frame_duration_ms=100,
    ),
    PartAsset(
        key="part4",
        title="Part 4  Visual-Inertial SLAM".replace("  ", " "),
        gif_name="part4_vi_slam.gif",
        png_name="part4_trajectory_comparison.png",
        out_frames=96,
        frame_duration_ms=100,
    ),
)


BG = (12, 14, 18)
CARD = (20, 24, 30)
TEXT = (236, 239, 244)
MUTED = (160, 168, 180)
LINE = (48, 58, 72)
ACCENT = (74, 144, 226)

TITLE_H = 72
LABEL_H = 42
FOOT_H = 10
CELL_W = 470
CELL_H = 320
PAD = 24
CANVAS_W = PAD + (CELL_W + PAD) * len(DATASETS)
CANVAS_H = TITLE_H + LABEL_H + CELL_H + FOOT_H + 3 * PAD


def load_font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = []
    if bold:
        candidates.extend(
            [
                "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
                "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf",
            ]
        )
    else:
        candidates.extend(
            [
                "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
                "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
            ]
        )
    for path in candidates:
        if Path(path).exists():
            return ImageFont.truetype(path, size=size)
    return ImageFont.load_default()


FONT_TITLE = load_font(36, bold=True)
FONT_LABEL = load_font(24, bold=True)
FONT_FOOT = load_font(16, bold=False)


def fit_image(img: Image.Image, width: int, height: int) -> Image.Image:
    src = img.convert("RGB")
    scale = min(width / src.width, height / src.height)
    new_w = max(1, int(src.width * scale))
    new_h = max(1, int(src.height * scale))
    resized = src.resize((new_w, new_h), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (width, height), CARD)
    x = (width - new_w) // 2
    y = (height - new_h) // 2
    canvas.paste(resized, (x, y))
    return canvas


def _sample_indices(num_src: int, num_dst: int) -> list[int]:
    if num_dst <= 1 or num_src <= 1:
        return [0] * max(1, num_dst)
    return [round(i * (num_src - 1) / (num_dst - 1)) for i in range(num_dst)]


def load_sampled_gif_frames(path: Path, num_frames: int) -> list[Image.Image]:
    with Image.open(path) as gif:
        src_frames = getattr(gif, "n_frames", 1)
        indices = _sample_indices(src_frames, num_frames)
        frames: list[Image.Image] = []
        for idx in indices:
            gif.seek(idx)
            frames.append(gif.convert("RGB").copy())
        return frames


def draw_shell(title: str) -> tuple[Image.Image, ImageDraw.ImageDraw]:
    canvas = Image.new("RGB", (CANVAS_W, CANVAS_H), BG)
    draw = ImageDraw.Draw(canvas)
    draw.rounded_rectangle(
        (PAD // 2, PAD // 2, CANVAS_W - PAD // 2, CANVAS_H - PAD // 2),
        radius=28,
        fill=BG,
        outline=LINE,
        width=2,
    )
    bbox = draw.textbbox((0, 0), title, font=FONT_TITLE)
    text_w = bbox[2] - bbox[0]
    draw.text(((CANVAS_W - text_w) / 2, PAD - 4), title, fill=TEXT, font=FONT_TITLE)
    y0 = TITLE_H + PAD // 2
    for col, ds in enumerate(DATASETS):
        x0 = PAD + col * (CELL_W + PAD)
        draw.rounded_rectangle(
            (x0, y0, x0 + CELL_W, y0 + LABEL_H + CELL_H),
            radius=18,
            fill=CARD,
            outline=LINE,
            width=2,
        )
        draw.rectangle((x0, y0, x0 + CELL_W, y0 + LABEL_H), fill=(27, 33, 42))
        draw.text((x0 + 16, y0 + 8), ds.label, fill=TEXT, font=FONT_LABEL)
        draw.line((x0, y0 + LABEL_H, x0 + CELL_W, y0 + LABEL_H), fill=ACCENT, width=3)
    return canvas, draw


def build_static_panel(part: PartAsset) -> None:
    canvas, _ = draw_shell(part.title)
    y_img = TITLE_H + PAD // 2 + LABEL_H
    for col, ds in enumerate(DATASETS):
        x_img = PAD + col * (CELL_W + PAD)
        img_path = RESULTS / ds.slug / part.png_name
        img = fit_image(Image.open(img_path), CELL_W, CELL_H)
        canvas.paste(img, (x_img, y_img))
    out = RESULTS / f"readme_{part.key}_static_panel.png"
    canvas.save(out, format="PNG", optimize=True)


def build_gif_panel(part: PartAsset) -> None:
    sampled = {
        ds.slug: load_sampled_gif_frames(RESULTS / ds.slug / part.gif_name, part.out_frames)
        for ds in DATASETS
    }
    rendered: list[Image.Image] = []
    y_img = TITLE_H + PAD // 2 + LABEL_H
    for i in range(part.out_frames):
        canvas, _ = draw_shell(part.title)
        for col, ds in enumerate(DATASETS):
            x_img = PAD + col * (CELL_W + PAD)
            frame = fit_image(sampled[ds.slug][i], CELL_W, CELL_H)
            canvas.paste(frame, (x_img, y_img))
        rendered.append(canvas.quantize(colors=192, method=Image.MEDIANCUT))

    out = RESULTS / f"readme_{part.key}_animated_panel.gif"
    rendered[0].save(
        out,
        save_all=True,
        append_images=rendered[1:],
        duration=part.frame_duration_ms,
        loop=0,
        optimize=False,
        disposal=2,
    )


def main(parts: Iterable[PartAsset] = PARTS) -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    for part in parts:
        build_static_panel(part)
        build_gif_panel(part)


if __name__ == "__main__":
    main()
