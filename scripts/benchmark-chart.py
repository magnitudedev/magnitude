#!/usr/bin/env python3
"""Render the llama.cpp comparison chart as light and dark SVGs for the README.

Colors are the landing page palette from magnitude-dev-landing/src/styles/globals.css:
blue-400 #38bdf8 (accent), blue-500 #0ea5e9 (main), slate-900 #0f1826, slate-700 #334155,
slate-500 #64748b, slate-300 #cbd5e1, slate-50 #f8fafc.
"""
from pathlib import Path

# (phase, gain %, llama.cpp tok/s, Magnitude tok/s). Gains come from the unrounded measurements.
BACKENDS = [
    ("Metal", "Mac M4 Pro 48 GB", [("prefill", 9, 466, 507), ("decode", 92, 30, 57)]),
    ("CUDA", "DGX Spark", [("prefill", 23, 2033, 2507), ("decode", 19, 49, 58)]),
]

THEMES = {
    "dark": dict(bg="#0f1826", fg="#f8fafc", text="#cbd5e1", muted="#64748b", base="#334155", accent="#38bdf8"),
    "light": dict(bg="#ffffff", fg="#0f1826", text="#334155", muted="#64748b", base="#cbd5e1", accent="#0ea5e9"),
}

SANS = "-apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif"
MONO = "ui-monospace, SFMono-Regular, Menlo, Consolas, monospace"

W, H = 800, 240
MARGIN = 28
PANEL_W = 372
GAP = 28
TILE_W = 160
TILE_GAP = 32
CAPTION = "Qwen 3.6 35B A3B, 4-bit, 64k context, no speculative decoding."


def fmt(n: float) -> str:
    return f"{round(n):,}"


def render(theme: dict) -> str:
    out = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" font-family="{SANS}">',
        f'<rect width="{W}" height="{H}" rx="12" fill="{theme["bg"]}"/>',
    ]
    for i, (name, hardware, phases) in enumerate(BACKENDS):
        x0 = MARGIN + i * (PANEL_W + GAP)
        out.append(f'<text x="{x0}" y="42" font-family="{MONO}" font-size="15" font-weight="600" fill="{theme["fg"]}">{name}</text>')
        out.append(f'<text x="{x0 + (78 if name == "Metal" else 62)}" y="42" font-size="14" fill="{theme["muted"]}">{hardware}</text>')
        for j, (phase, gain, llama, mag) in enumerate(phases):
            x = x0 + j * (TILE_W + TILE_GAP)
            out.append(f'<text x="{x}" y="100" font-family="{MONO}" font-size="40" font-weight="700" letter-spacing="-1.5" fill="{theme["fg"]}">{gain}%</text>')
            out.append(f'<text x="{x}" y="126" font-size="15" fill="{theme["text"]}">faster {phase}</text>')
            base_w = round(TILE_W * 100 / (100 + gain))
            out.append(f'<rect x="{x}" y="144" width="{base_w}" height="7" rx="3.5" fill="{theme["base"]}"/>')
            out.append(f'<rect x="{x}" y="156" width="{TILE_W}" height="7" rx="3.5" fill="{theme["accent"]}"/>')
            out.append(
                f'<text x="{x}" y="186" font-size="14" fill="{theme["muted"]}">{fmt(llama)} → '
                f'<tspan font-weight="600" fill="{theme["fg"]}">{fmt(mag)}</tspan> tok/s</text>'
            )
    ly = H - 18
    out.append(f'<text x="{MARGIN}" y="{ly}" font-size="13" fill="{theme["muted"]}">{CAPTION}</text>')
    right = W - MARGIN
    out.append(f'<text x="{right}" y="{ly}" font-size="13" text-anchor="end" fill="{theme["muted"]}">Magnitude</text>')
    out.append(f'<rect x="{right - 66 - 24}" y="{ly - 7}" width="18" height="7" rx="3.5" fill="{theme["accent"]}"/>')
    out.append(f'<text x="{right - 66 - 36}" y="{ly}" font-size="13" text-anchor="end" fill="{theme["muted"]}">llama.cpp</text>')
    out.append(f'<rect x="{right - 66 - 36 - 62 - 24}" y="{ly - 7}" width="18" height="7" rx="3.5" fill="{theme["base"]}"/>')
    out.append("</svg>")
    return "\n".join(out) + "\n"


def main() -> None:
    root = Path(__file__).resolve().parent.parent / "assets" / "benchmarks"
    root.mkdir(parents=True, exist_ok=True)
    for name, theme in THEMES.items():
        (root / f"llama-cpp-{name}.svg").write_text(render(theme))
        print(root / f"llama-cpp-{name}.svg")


if __name__ == "__main__":
    main()
