#!/usr/bin/env python3
"""Draws the canvas templates for placing a picture on the ring. Needs numpy and Pillow.

The generator maps a square picture so that half its width equals 57.5 mm (imageOuterRadiusMm).
"""
import argparse
import glob
import os

import numpy as np
from PIL import Image, ImageDraw, ImageFont

SIZE = 2400
IMAGE_RADIUS_MM = 57.5
PIXELS_PER_MM = (SIZE / 2) / IMAGE_RADIUS_MM
SAFE_INNER_MM = 26.5  # keep faces and text inside this band
SAFE_OUTER_MM = 56.0  # beyond it the disc read with errors on the last 1.5 mm


def radial_coverage(inner_mm, outer_mm):
    """Antialiased coverage of the ring between two radii, 0 to 1."""
    y, x = np.mgrid[0:SIZE, 0:SIZE]
    distance = np.hypot(x - SIZE / 2 + 0.5, y - SIZE / 2 + 0.5)
    return (np.clip(distance - inner_mm * PIXELS_PER_MM + 0.5, 0, 1)
            * np.clip(outer_mm * PIXELS_PER_MM - distance + 0.5, 0, 1))


def font(size):
    found = (glob.glob("/usr/share/fonts/**/DejaVuSans.ttf", recursive=True)
             + glob.glob("/usr/share/fonts/**/*.ttf", recursive=True))
    return ImageFont.truetype(found[0], size) if found else ImageFont.load_default()


def circle(draw, radius_mm, color, width, dash=None):
    radius = radius_mm * PIXELS_PER_MM
    box = [SIZE / 2 - radius, SIZE / 2 - radius, SIZE / 2 + radius, SIZE / 2 + radius]
    if dash is None:
        draw.ellipse(box, outline=color, width=width)
    else:
        for start in range(0, 360, dash * 2):
            draw.arc(box, start, start + dash, fill=color, width=width)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inner-mm", type=float, default=24.897, help="Printed ring starts here")
    parser.add_argument("--output", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "template"))
    arguments = parser.parse_args()
    os.makedirs(arguments.output, exist_ok=True)
    ring = radial_coverage(arguments.inner_mm, IMAGE_RADIUS_MM)

    Image.fromarray(np.uint8(ring * 255)).save(os.path.join(arguments.output, "disc-ring-luma-2400.png"))

    rgba = np.zeros((SIZE, SIZE, 4), np.uint8)
    rgba[..., :3] = 190
    rgba[..., 3] = np.uint8((1 - ring) * 255)
    mask = Image.fromarray(rgba)
    mask.save(os.path.join(arguments.output, "disc-mask-2400.png"))

    guides = Image.new("RGBA", (SIZE, SIZE), (0, 0, 0, 0))
    draw = ImageDraw.Draw(guides)
    centre = SIZE / 2
    circle(draw, arguments.inner_mm, (255, 0, 0, 255), 4)
    circle(draw, IMAGE_RADIUS_MM - 2 / PIXELS_PER_MM, (255, 0, 0, 255), 4)
    circle(draw, SAFE_INNER_MM, (0, 170, 0, 255), 3, dash=2)
    circle(draw, SAFE_OUTER_MM, (0, 170, 0, 255), 3, dash=2)
    for angle in range(0, 360, 30):
        cos, sin = np.cos(np.radians(angle)), np.sin(np.radians(angle))
        low, high = arguments.inner_mm * PIXELS_PER_MM, IMAGE_RADIUS_MM * PIXELS_PER_MM
        major = angle % 90 == 0
        draw.line([centre + low * cos, centre + low * sin, centre + high * cos, centre + high * sin],
                  fill=(0, 120, 255, 200 if major else 110), width=3 if major else 2)
    draw.line([centre - 40, centre, centre + 40, centre], fill=(255, 0, 0, 255), width=3)
    draw.line([centre, centre - 40, centre, centre + 40], fill=(255, 0, 0, 255), width=3)
    labels = font(30)
    for radius_mm in range(30, 56, 5):
        radius = radius_mm * PIXELS_PER_MM
        draw.line([centre - 12, centre - radius, centre + 12, centre - radius], fill=(0, 120, 255, 255), width=3)
        draw.text((centre + 18, centre - radius - 18), f"{radius_mm} mm", fill=(0, 90, 220, 255), font=labels)
    guides.save(os.path.join(arguments.output, "disc-guides-2400.png"))

    overlay = mask.copy()
    overlay.alpha_composite(guides)
    overlay.save(os.path.join(arguments.output, "disc-overlay-2400.png"))
    print(f"Saved to {arguments.output}; {PIXELS_PER_MM:.2f} px per mm, "
          f"hole radius {arguments.inner_mm * PIXELS_PER_MM:.1f} px")


if __name__ == "__main__":
    main()
