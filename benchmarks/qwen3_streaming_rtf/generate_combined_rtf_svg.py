#!/usr/bin/env python3
"""Generate the combined Qwen3 streaming RTF SVG for README/model card."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from xml.sax.saxutils import escape


DIR = Path(__file__).resolve().parent
ROOT = DIR.parents[1]


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _rtf_from_summary(path: Path) -> float:
    data = _read_json(path)
    rtf = data.get("realtime_factor_total")
    if rtf is None:
        rtf = data.get("realtime_factor_mean")
    if rtf is None:
        raise SystemExit(f"{path}: missing realtime_factor_total/realtime_factor_mean")
    return float(rtf)


def _path_bar(x: float, y: float, width: float, height: float, radius: float) -> str:
    radius = min(radius, width / 2, height)
    return (
        f"M{x:.2f},{(y + height):.2f} "
        f"L{x:.2f},{(y + radius):.2f} "
        f"Q{x:.2f},{y:.2f} {(x + radius):.2f},{y:.2f} "
        f"L{(x + width - radius):.2f},{y:.2f} "
        f"Q{(x + width):.2f},{y:.2f} {(x + width):.2f},{(y + radius):.2f} "
        f"L{(x + width):.2f},{(y + height):.2f} Z"
    )


def _text(
    x: float,
    y: float,
    value: str,
    *,
    size: int = 18,
    fill: str = "#111827",
    weight: int | str = 500,
    anchor: str = "middle",
    extra: str = "",
) -> str:
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" text-anchor="{anchor}" '
        f'font-family="Arial, Helvetica, sans-serif" font-size="{size}" '
        f'font-weight="{weight}" '
        f'fill="{fill}" {extra}>{escape(value)}</text>'
    )


def _load_groups(args: argparse.Namespace) -> list[dict[str, object]]:
    streaming = _read_json(args.streaming_results)
    systems = {
        (row["family"], row["mode"]): float(row["rtf"])
        for row in streaming["systems"]
    }
    return [
        {
            "top": "Apple M5",
            "bottom": "vLLM Metal",
            "normal": systems[("metal", "normal")],
            "causal": systems[("metal", "causal")],
        },
        {
            "top": "NVIDIA H100",
            "bottom": "HF Transformers",
            "normal": systems[("h100", "normal")],
            "causal": systems[("h100", "causal")],
        },
        {
            "top": "NVIDIA A100",
            "bottom": "vLLM CUDA",
            "normal": _rtf_from_summary(args.vllm_normal_summary),
            "causal": _rtf_from_summary(args.vllm_causal_summary),
        },
    ]


def build_svg(groups: list[dict[str, object]]) -> str:
    width = 1180
    height = 680
    plot_left = 116
    plot_top = 182
    plot_width = 990
    plot_height = 332
    baseline = plot_top + plot_height
    y_max = 0.34
    normal_color = "#9CA3AF"
    causal_color = "#7C3AED"
    grid_color = "#E5E7EB"
    text_dark = "#111827"
    text_muted = "#4B5563"

    def y_for(value: float) -> float:
        return baseline - (value / y_max) * plot_height

    parts: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">',
        "<title id=\"title\">Qwen3-ASR Streaming RTF</title>",
        "<desc id=\"desc\">Normal Qwen3-ASR is shown in gray and the causal audio tower in violet. Lower real-time factor is faster.</desc>",
        '<rect width="1180" height="680" fill="#FFFFFF"/>',
        _text(width / 2, 54, "Qwen3-ASR Streaming RTF", size=34, weight=800),
        _text(width / 2, 86, "ASR compute / audio duration, excluding model load. Lower is faster.", size=17, fill=text_muted, weight=500),
    ]

    # Legend
    legend_y = 108
    parts.extend(
        [
            f'<rect x="424" y="{legend_y - 15}" width="34" height="18" rx="9" fill="{normal_color}"/>',
            _text(468, legend_y, "Qwen3-ASR normal", size=16, fill=text_dark, weight=600, anchor="start"),
            f'<rect x="644" y="{legend_y - 15}" width="34" height="18" rx="9" fill="{causal_color}"/>',
            _text(688, legend_y, "Qwen3 causal audio", size=16, fill=text_dark, weight=600, anchor="start"),
        ]
    )

    # Axes and grid.
    for tick in [0.0, 0.1, 0.2, 0.3]:
        y = y_for(tick)
        parts.append(
            f'<line x1="{plot_left}" y1="{y:.1f}" x2="{plot_left + plot_width}" y2="{y:.1f}" '
            f'stroke="{grid_color}" stroke-width="1.4"/>'
        )
        parts.append(_text(plot_left - 18, y + 6, f"{tick:.1f}", size=15, fill=text_muted, weight=500, anchor="end"))
    parts.append(
        f'<line x1="{plot_left}" y1="{baseline:.1f}" x2="{plot_left + plot_width}" y2="{baseline:.1f}" '
        'stroke="#111827" stroke-width="1.6"/>'
    )
    parts.append(
        f'<line x1="{plot_left}" y1="{plot_top}" x2="{plot_left}" y2="{baseline}" '
        'stroke="#111827" stroke-width="1.6"/>'
    )
    parts.append(
        _text(
            34,
            plot_top + plot_height / 2,
            "Streaming inference RTF",
            size=16,
            fill=text_dark,
            weight=700,
            extra=f'transform="rotate(-90 34 {plot_top + plot_height / 2:.1f})"',
        )
    )

    group_step = plot_width / len(groups)
    bar_width = 76
    bar_gap = 22
    radius = 18
    for index, group in enumerate(groups):
        center = plot_left + group_step * (index + 0.5)
        normal = float(group["normal"])
        causal = float(group["causal"])
        entries = [
            ("normal", normal, normal_color, center - bar_gap / 2 - bar_width),
            ("causal", causal, causal_color, center + bar_gap / 2),
        ]
        for _mode, value, color, x in entries:
            y = y_for(value)
            bar_h = baseline - y
            parts.append(
                f'<path d="{_path_bar(x, y, bar_width, bar_h, radius)}" fill="{color}"/>'
            )
            parts.append(
                _text(
                    x + bar_width / 2,
                    y - 12,
                    f"{value:.3f}",
                    size=17,
                    fill=text_dark,
                    weight=800,
                )
            )

        speedup = normal / causal
        pill_w = 134
        pill_h = 30
        pill_x = center - pill_w / 2
        pill_y = plot_top - 46
        parts.append(
            f'<rect x="{pill_x:.1f}" y="{pill_y:.1f}" width="{pill_w}" height="{pill_h}" '
            'rx="15" fill="#F3E8FF"/>'
        )
        parts.append(
            _text(center, pill_y + 21, f"{speedup:.2f}x faster", size=15, fill="#6D28D9", weight=800)
        )

        parts.append(_text(center, baseline + 38, str(group["top"]), size=19, fill=text_dark, weight=700))
        parts.append(_text(center, baseline + 64, str(group["bottom"]), size=17, fill=text_muted, weight=600))

    parts.extend(
        [
            _text(
                width / 2,
                height - 58,
                "Live streaming, no past rewrite, 250 ms holdback; model load excluded.",
                size=15,
                fill=text_muted,
                weight=500,
            ),
            _text(
                width / 2,
                height - 32,
                "H100 uses HF Transformers/CUDA; A100 uses vLLM CUDA; Metal uses vLLM Metal.",
                size=15,
                fill=text_muted,
                weight=500,
            ),
            "</svg>",
        ]
    )
    return "\n".join(parts) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--streaming-results",
        type=Path,
        default=DIR / "results.json",
    )
    parser.add_argument(
        "--vllm-normal-summary",
        type=Path,
        default=ROOT / "benchmarks/qwen3_vllm_h100/results/a100_normal_jfk_x2.summary.json",
    )
    parser.add_argument(
        "--vllm-causal-summary",
        type=Path,
        default=ROOT
        / "benchmarks/qwen3_vllm_h100/results/a100_causal_vllm_live_delta_mp_asyncpatch2_jfk_x2.summary.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "assets/rtf_combined.svg",
    )
    args = parser.parse_args()

    groups = _load_groups(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(build_svg(groups), encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
