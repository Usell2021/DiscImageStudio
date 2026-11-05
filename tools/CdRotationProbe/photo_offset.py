#!/usr/bin/env python3
"""Finds the sector offset, the one number timing cannot see, from photos of a burned cross.

Step 1, without --arm: unwraps each photo to polar PNGs (0.25 deg per column, 0.1 mm per row).
        Radial lines look vertical there; note the columns of the clean ones.
Step 2, with --arm: tracks those lines and fits their shape to the timing profile:
        arm(r) = timing residual(r) - offset * 16 mm / r + constant.
"""
import argparse
import json
import os

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from PIL import Image

from spiral import image_angle

DEGREES_PER_COLUMN = 0.25
RADII_MM = np.linspace(24, 60, 360)
FIT_RANGE = (RADII_MM > 26) & (RADII_MM < 57.3)
MIN_AGREEMENT = 0.5  # arms that track the others worse than this lost their line


def fit_rim(gray, threshold):
    """Ellipse through the first bright pixel on rays coming in from outside the disc."""
    height, width = gray.shape
    centre = np.array([width / 2, height / 2])
    points = []
    for angle in np.linspace(0, 2 * np.pi, 1440, endpoint=False):
        radii = np.arange(0.55 * height, 0.19 * height, -0.5)
        xs, ys = centre[0] + radii * np.cos(angle), centre[1] + radii * np.sin(angle)
        inside = (xs >= 2) & (xs < width - 3) & (ys >= 2) & (ys < height - 3)
        if inside.sum() < 50:
            continue
        values = gray[ys[inside].astype(int), xs[inside].astype(int)]
        first = np.argmax(values > threshold)
        if first > 0 and values[first] > threshold:
            points.append((xs[inside][first], ys[inside][first]))
    points = np.array(points)
    keep = np.ones(len(points), bool)
    for _ in range(8):
        x, y = points[keep, 0], points[keep, 1]
        mx, my, scale = x.mean(), y.mean(), x.std()
        u, v = (x - mx) / scale, (y - my) / scale
        design = np.stack([u * u, u * v, v * v, u, v, np.ones_like(u)], 1)
        a, b, c, d, e, f = np.linalg.svd(design, full_matrices=False)[2][-1]
        quadratic = np.array([[a, b / 2], [b / 2, c]])
        conic_centre = np.linalg.solve(-2 * quadratic, [d, e])
        level = f + 0.5 * (d * conic_centre[0] + e * conic_centre[1])
        values, vectors = np.linalg.eigh(quadratic / (-level))
        axes = scale / np.sqrt(values)
        ellipse_centre = conic_centre * scale + [mx, my]
        if np.linalg.det(vectors) < 0:  # a reflected basis would mirror the angle
            vectors[:, 0] *= -1
        distance = (points - ellipse_centre) @ vectors / axes
        error = np.hypot(distance[:, 0], distance[:, 1]) - 1
        keep = np.abs(error) < max(2.0 * error[keep].std(), 0.004)
    return ellipse_centre, axes, vectors, error[keep].std() * axes.mean()


def unwrap(path, threshold):
    image = np.asarray(Image.open(path).convert("RGB")).astype(float)
    centre, axes, vectors, rim_error = fit_rim(image.mean(2), threshold)
    print(f"{os.path.basename(path)}: rim error {rim_error:.2f} px, axes {np.round(axes, 1).tolist()}")
    angle, radius = np.meshgrid(np.radians(np.arange(0, 360, DEGREES_PER_COLUMN)), RADII_MM)
    plane = np.stack([np.cos(angle) * radius / 60 * axes[0], np.sin(angle) * radius / 60 * axes[1]], -1)
    positions = plane @ vectors.T + centre
    x, y = positions[..., 0], positions[..., 1]
    x0 = np.clip(np.floor(x).astype(int), 0, image.shape[1] - 2)
    y0 = np.clip(np.floor(y).astype(int), 0, image.shape[0] - 2)
    fx, fy = (x - x0)[..., None], (y - y0)[..., None]
    return (image[y0, x0] * (1 - fx) * (1 - fy) + image[y0, x0 + 1] * fx * (1 - fy)
            + image[y0 + 1, x0] * (1 - fx) * fy + image[y0 + 1, x0 + 1] * fx * fy)


def line_strength(polar):
    """Distance from the running row median; rings are horizontal, so they cancel out."""
    half = 12
    total = 0
    for channel in range(3):
        padded = np.pad(polar[:, :, channel], ((0, 0), (half, half)), mode="wrap")
        total = total + np.abs(polar[:, :, channel] - np.median(sliding_window_view(padded, 2 * half + 1, axis=1), axis=-1))
    return total


def track(strength, column, half=26, penalty=0.18):
    """Best connected path through a window around a line; returns its angle in degrees per row."""
    window = strength[:, np.arange(column - half, column + half + 1) % strength.shape[1]]
    window = window / (np.percentile(window, 99.5) + 1e-9)
    window = np.clip(window - np.median(window, axis=1, keepdims=True), 0, None)
    rows, columns = window.shape
    cost = window.copy()
    back = np.zeros_like(window, int)
    for row in range(1, rows):
        for current in range(columns):
            low, high = max(0, current - 3), min(columns, current + 4)
            candidates = cost[row - 1, low:high] - penalty * np.abs(np.arange(low, high) - current)
            best = candidates.argmax()
            cost[row, current] = window[row, current] + candidates[best]
            back[row, current] = low + best
    path = [cost[-1].argmax()]
    for row in range(rows - 1, 0, -1):
        path.append(back[row, path[-1]])
    path.reverse()
    result = []
    for row, current in enumerate(path):
        low, high = max(0, current - 2), min(columns, current + 3)
        weights = window[row, low:high] + 1e-6
        result.append((np.arange(low, high) * weights).sum() / weights.sum())
    return (column - half + np.array(result)) * DEGREES_PER_COLUMN


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("photos", nargs="*", help="Step 1: photos to unwrap")
    parser.add_argument("--polar-dir", default=".", help="Where step 1 writes PNGs")
    parser.add_argument("--arm", action="append", default=[], help="Step 2: photo.png:column[,column]")
    parser.add_argument("--phase", help="Step 2: NPZ from solve.py")
    parser.add_argument("--sidecar", help="Step 2: <track>.json of the burned cross")
    parser.add_argument("--rim-threshold", type=float, default=170)
    arguments = parser.parse_args()

    for path in arguments.photos:
        output = os.path.join(arguments.polar_dir, os.path.splitext(os.path.basename(path))[0] + "-polar.png")
        Image.fromarray(np.uint8(unwrap(path, arguments.rim_threshold))).save(output)
        print(f"Saved {output}")
    if not arguments.arm:
        return
    if not arguments.phase or not arguments.sidecar:
        parser.error("--arm needs --phase and --sidecar")

    arms, labels = [], []
    for item in arguments.arm:
        path, columns = item.rsplit(":", 1)
        strength = line_strength(unwrap(path, arguments.rim_threshold))
        for column in columns.split(","):
            line = track(strength, int(column))
            arms.append(line - line[FIT_RANGE].mean())
            labels.append(f"{os.path.basename(path)}:{column}")
    arms = np.array(arms)
    good = []
    for index, label in enumerate(labels):
        others = np.median(np.delete(arms, index, 0), axis=0)
        agreement = np.corrcoef(arms[index, FIT_RANGE], others[FIT_RANGE])[0, 1]
        print(f"  {label}: sd {arms[index, FIT_RANGE].std():.2f} deg, agreement with the others {agreement:.3f}")
        if agreement >= MIN_AGREEMENT:
            good.append(index)
    if len(good) < 2:
        raise SystemExit("Fewer than two arms agree, so no offset can be trusted. Check the columns.")

    with open(arguments.sidecar, encoding="utf-8") as stream:
        burned = json.load(stream)
    data = np.load(arguments.phase)
    sector, phi = data["s"].astype(float), data["phi"]
    angle, spiral = image_angle(burned, sector)
    timing = np.interp(RADII_MM, spiral.radius(sector), np.degrees(phi - angle))
    design = np.stack([np.ones(FIT_RANGE.sum()), np.degrees(16 / RADII_MM[FIT_RANGE])], 1)

    def fit(curve):
        solution = np.linalg.lstsq(design, (curve - timing)[FIT_RANGE], rcond=None)[0]
        return solution, ((curve - timing)[FIT_RANGE] - design @ solution).std()

    each = np.array([-fit(arms[index])[0][1] for index in good])
    median = np.median(arms[good], axis=0)
    (_, slope), remaining = fit(median)
    print(f"Arms used: {len(good)}. Median arm rms {median[FIT_RANGE].std():.2f} deg, after the fit {remaining:.2f} deg")
    print(f"Sector offset: {-slope:.3f} (per arm {each.mean():.3f} +- {each.std(ddof=1) / np.sqrt(len(each)):.3f})")


if __name__ == "__main__":
    main()
