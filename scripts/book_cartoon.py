"""Original, deterministic cartoon host artwork for narrated book lessons.

All pixels are procedurally drawn in Pillow; no copyrighted book-cover art,
stock media, or external downloads. Mouth motion is illustrative, not lip-synced.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont

CANVAS = (1080, 1920)
HOST_SIZE = (600, 820)
DESIGN_VERSION = "pip-cartoon-v1"
INK = "#13243C"
NAVY = "#10213A"
CREAM = "#FFF2D8"
MINT = "#79E1C3"
YELLOW = "#FFCF72"
WHITE = "#F8F9FF"


def _font(size: int, *, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    filename = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    return ImageFont.truetype(filename, size)


def _lines(text: str, font: Any, width: float, *, max_lines: int = 3) -> list[str]:
    words = text.strip().split()
    result: list[str] = []
    line = ""
    for word in words:
        proposed = f"{line} {word}".strip()
        if font.getlength(proposed) <= width:
            line = proposed
            continue
        if not line:
            raise ValueError("single word exceeds safe portrait text width")
        result.append(line)
        line = word
    if line:
        result.append(line)
    if len(result) > max_lines:
        raise ValueError("book artwork text would overflow the portrait safe region")
    return result


def _multiline(draw: ImageDraw.ImageDraw, text: str, x: int, y: int, size: int, *,
               width: int = 910, max_lines: int = 3, fill: str = WHITE) -> int:
    font = _font(size, bold=True)
    for line in _lines(text, font, width, max_lines=max_lines):
        draw.text((x, y), line, font=font, fill=fill, stroke_width=0)
        y += int(size * 1.25)
    return y


def _background(path: Path, brand: str, book: str, author: str, headline: str,
                index: int, total: int) -> None:
    image = Image.new("RGB", CANVAS, NAVY)
    d = ImageDraw.Draw(image)
    for y in range(CANVAS[1]):
        blend = y / CANVAS[1]
        d.line((0, y, CANVAS[0], y),
               fill=(int(16 + 18 * blend), int(33 + 11 * blend), int(58 + 20 * blend)))
    accent = ("#395681", "#356B69", "#6B5580", "#765C55", "#3C6684")[index % 5]
    d.ellipse((710, 440, 1490, 1260), fill=accent, outline=None)
    d.ellipse((-340, 1260, 330, 1930), outline="#456779", width=6)
    d.ellipse((-235, 1370, 245, 1850), outline="#456779", width=4)
    d.rounded_rectangle((59, 100, 1021, 188), radius=34, fill="#213A51")
    d.text((89, 122), brand.upper(), font=_font(32, bold=True), fill=MINT)
    d.text((832, 124), f"{index+1:02d}/{total:02d}", font=_font(31, bold=True),
           fill=WHITE)
    d.text((80, 239), "AN IDEA FROM", font=_font(34, bold=True), fill=MINT)
    y = _multiline(d, book, 80, 301, 64, width=920, max_lines=2, fill=CREAM)
    d.text((82, y + 4), f"BY {author.upper()}", font=_font(29, bold=True), fill=YELLOW)
    d.line((78, y + 67, 1000, y + 67), fill="#6A8797", width=3)
    _multiline(d, headline, 81, max(y + 97, 514), 59, width=910, max_lines=3)
    # Graphic book prop: original vector art, never a reproduction of a cover.
    d.rounded_rectangle((110, 1235, 385, 1600), radius=22, fill=CREAM, outline=INK, width=8)
    d.polygon([(135, 1263), (160, 1234), (160, 1588), (135, 1577)], fill=MINT)
    d.rounded_rectangle((187, 1310, 334, 1330), radius=9, fill=INK)
    d.rounded_rectangle((187, 1360, 325, 1380), radius=9, fill="#4D8191")
    d.ellipse((201, 1416, 325, 1540), outline="#D59355", width=10)
    d.line((240, 1475, 273, 1507, 309, 1446), fill=INK, width=9)
    d.rounded_rectangle((64, 1640, 1016, 1722), radius=24, fill="#20374D")
    d.text((91, 1658), "A BOOK IDEA. A REAL-LIFE EXPERIMENT.",
           font=_font(27, bold=True), fill=WHITE)
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, format="PNG")


def _host(path: Path, *, talking: bool) -> None:
    """Draw Pip: original bespectacled cartoon guide with two speaking poses."""
    image = Image.new("RGBA", HOST_SIZE, (0, 0, 0, 0))
    d = ImageDraw.Draw(image)
    # Shoes and legs
    d.rounded_rectangle((233, 646, 284, 760), radius=18, fill=INK)
    d.rounded_rectangle((335, 645, 384, 763), radius=18, fill=INK)
    d.ellipse((196, 728, 286, 793), fill=WHITE, outline=INK, width=7)
    d.ellipse((337, 726, 425, 791), fill=WHITE, outline=INK, width=7)
    # Sweater and arms; slight asymmetry makes the character readable at phone size
    d.rounded_rectangle((175, 422, 446, 688), radius=99, fill=MINT, outline=INK, width=13)
    d.polygon([(175, 476), (106, 563), (123, 608), (202, 546)], fill=MINT)
    d.line((175, 476, 106, 563, 123, 608), fill=INK, width=13, joint="curve")
    d.ellipse((88, 573, 142, 632), fill=CREAM, outline=INK, width=9)
    d.polygon([(435, 459), (518, 392), (554, 427), (465, 538)], fill=MINT)
    d.line((435, 459, 518, 392, 554, 427), fill=INK, width=13, joint="curve")
    d.ellipse((518, 384, 577, 437), fill=CREAM, outline=INK, width=9)
    d.polygon([(240, 464), (369, 464), (305, 544)], fill=YELLOW)
    # Head and ears
    d.ellipse((113, 187, 176, 280), fill=CREAM, outline=INK, width=12)
    d.ellipse((449, 182, 518, 278), fill=CREAM, outline=INK, width=12)
    d.rounded_rectangle((129, 80, 494, 453), radius=150, fill=CREAM, outline=INK, width=13)
    # Hair contour
    d.pieslice((124, 62, 490, 267), 183, 358, fill=INK)
    d.polygon([(195, 105), (174, 179), (269, 145), (298, 198), (343, 140),
               (438, 189), (450, 122)], fill=INK)
    d.polygon([(259, 90), (323, 61), (344, 85), (278, 120)], fill=YELLOW)
    # Glasses and eyes
    for x in (181, 346):
        d.ellipse((x, 245, x + 119, 357), fill="#D1F5EE", outline=INK, width=11)
        d.ellipse((x + 54, 282, x + 79, 318), fill=INK)
        d.ellipse((x + 61, 287, x + 70, 296), fill=WHITE)
    d.line((300, 294, 344, 294), fill=INK, width=13)
    d.ellipse((308, 327, 350, 352), fill="#E9A89E")
    d.ellipse((170, 354, 199, 370), fill="#E8A79F")
    d.ellipse((448, 354, 477, 370), fill="#E8A79F")
    if talking:
        d.ellipse((288, 370, 355, 424), fill=INK)
        d.arc((300, 392, 343, 425), 0, 180, fill="#F79A9E", width=10)
    else:
        d.arc((292, 374, 360, 418), 4, 176, fill=INK, width=11)
    # Star badge: recurring identity motif
    cx, cy = 326, 600
    star = []
    for n in range(10):
        angle = -math.pi / 2 + n * math.pi / 5
        rr = 37 if n % 2 == 0 else 16
        star.append((cx + rr * math.cos(angle), cy + rr * math.sin(angle)))
    d.polygon(star, fill=YELLOW)
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, format="PNG")


def create_assets(folder: Path, brand: str, book: str, author: str,
                  headline: str, index: int, total: int) -> tuple[Path, Path, Path]:
    """Return background and two RGBA host poses, all original and deterministic."""
    bg = folder / f"book-scene-{index:02d}.png"
    talking = folder / "pip-talking.png"
    idle = folder / "pip-idle.png"
    _background(bg, brand, book, author, headline, index, total)
    if not talking.is_file():
        _host(talking, talking=True)
    if not idle.is_file():
        _host(idle, talking=False)
    return bg, talking, idle
