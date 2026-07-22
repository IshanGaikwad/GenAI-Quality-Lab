"""Regenerate the GitHub social-preview card at docs/social-preview.png.

A small, dependency-light generator (Pillow only) so the 1280x640 preview image
is reproducible rather than a mystery binary. Fonts are resolved from a
cross-platform candidate list (Segoe UI on Windows, Arial on macOS, DejaVu on
Linux) so it runs anywhere; if none are found it falls back to Pillow's default.

    pip install pillow
    python scripts/make_social_preview.py

Note: GitHub has no API for the social preview — after regenerating, upload the
image manually via Settings -> General -> Social preview.
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter, ImageFont

W, H = 1280, 640
PAD = 92
OUT = Path(__file__).resolve().parent.parent / "docs" / "social-preview.png"

# palette (dark, indigo accent)
BG_TOP = (12, 15, 25)
BG_BOT = (20, 24, 44)
GLOW = (99, 102, 241)
WHITE = (238, 241, 249)
GRAY = (150, 158, 180)
ACCENT = (129, 145, 255)
CHIP_BORDER = (58, 64, 98)
CHIP_TEXT = (176, 184, 206)

# Cross-platform font candidates: first that loads wins.
BOLD = [
    "C:/Windows/Fonts/segoeuib.ttf",
    "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "DejaVuSans-Bold.ttf",
]
SANS = [
    "C:/Windows/Fonts/segoeuisl.ttf",
    "C:/Windows/Fonts/segoeui.ttf",
    "/System/Library/Fonts/Supplemental/Arial.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "DejaVuSans.ttf",
]
MONO = [
    "C:/Windows/Fonts/consola.ttf",
    "/System/Library/Fonts/Menlo.ttc",
    "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
    "DejaVuSansMono.ttf",
]


def load_font(candidates: list[str], size: int) -> ImageFont.FreeTypeFont:
    for path in candidates:
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    return ImageFont.load_default(size)


def build() -> Image.Image:
    f_title = load_font(BOLD, 74)
    f_eyebrow = load_font(BOLD, 20)
    f_tag = load_font(SANS, 31)
    f_chip = load_font(MONO, 21)
    f_foot = load_font(MONO, 21)

    # background: vertical gradient
    img = Image.new("RGB", (W, H), BG_TOP)
    px = img.load()
    for y in range(H):
        t = y / H
        row = tuple(int(BG_TOP[i] + (BG_BOT[i] - BG_TOP[i]) * t) for i in range(3))
        for x in range(W):
            px[x, y] = row

    # soft indigo glow, top-right
    glow = Image.new("RGB", (W, H), (0, 0, 0))
    ImageDraw.Draw(glow).ellipse([W - 520, -280, W + 220, 380], fill=GLOW)
    glow = glow.filter(ImageFilter.GaussianBlur(170))
    img = Image.composite(glow, img, Image.new("L", (W, H), 60))

    draw = ImageDraw.Draw(img)
    draw.rounded_rectangle([PAD, 96, PAD + 54, 100], radius=2, fill=ACCENT)
    draw.text((PAD, 116), "O P E N   S O U R C E   ·   M I T   ·   P Y T H O N", font=f_eyebrow, fill=ACCENT)
    draw.text((PAD - 2, 156), "GenAI Quality Lab", font=f_title, fill=WHITE)
    draw.text((PAD, 268), "Test a RAG assistant like production — groundedness,", font=f_tag, fill=GRAY)
    draw.text((PAD, 310), "hallucination, retrieval & correctness, gated in CI.", font=f_tag, fill=GRAY)

    chips = ["groundedness", "hallucination", "retrieval", "answer F1", "refusal", "semantic (NLI)"]
    x, y, ch = PAD, 402, 46
    for label in chips:
        w = int(draw.textlength(label, font=f_chip) + 40)
        draw.rounded_rectangle([x, y, x + w, y + ch], radius=12, outline=CHIP_BORDER, width=2)
        draw.text((x + 20, y + 12), label, font=f_chip, fill=CHIP_TEXT)
        x += w + 14

    draw.text((PAD, 556), "deterministic · offline · 100% coverage · lexical gate + non-blocking semantic stage", font=f_foot, fill=(110, 118, 140))
    url = "github.com/IshanGaikwad/GenAI-Quality-Lab"
    draw.text((W - PAD - draw.textlength(url, font=f_foot), 116), url, font=f_foot, fill=(120, 128, 150))
    return img


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    image = build()
    image.save(OUT)
    print(f"saved {OUT} {image.size}")


if __name__ == "__main__":
    main()
