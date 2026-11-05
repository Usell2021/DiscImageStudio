#!/usr/bin/env python3
"""Builds a CD preset from a phase profile and the sector offset found by photo_offset.py.

Timing gives phi(s), the phase when the read of sector s completes. The byte at position s lies at
phi(s - offset), so the cubic spiral is fitted to that, and what it cannot follow goes to the table.
"""
import argparse
import json

import numpy as np

from spiral import Spiral, fit_cubic

TABLE_SIZE = 181
SMOOTHING_NODES = 3  # each side; single dense nodes carry about 0.3 deg of noise


def smooth(values, half_width):
    kernel = np.hanning(2 * half_width + 3)[1:-1]
    return np.convolve(np.pad(values, half_width, mode="edge"), kernel / kernel.sum(), "valid")


def build(sector, phi, sectors, offset):
    positions = np.clip(sector + offset, 0, sectors)
    parameters, residual = fit_cubic(positions, phi, sectors)
    wander = smooth(-np.degrees(residual), SMOOTHING_NODES)
    table = np.interp(np.linspace(0, sectors, TABLE_SIZE), positions, wander)
    remaining = -np.degrees(residual) - np.interp(positions, np.linspace(0, sectors, TABLE_SIZE), table)
    print(f"Cubic alone: rms {np.degrees(residual.std()):.2f}, max {np.degrees(np.abs(residual).max()):.2f} deg")
    print(f"With the table: rms {remaining.std():.2f}, max {np.abs(remaining).max():.2f} deg "
          "(the maximum is mostly noise of single nodes)")
    return parameters, table - table.mean()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", help="NPZ from solve.py")
    parser.add_argument("--sectors", type=int, required=True)
    parser.add_argument("--offset", type=float, required=True, help="Sectors, from photo_offset.py")
    parser.add_argument("--id", required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--output", help="JSON file; printed when omitted")
    arguments = parser.parse_args()
    data = np.load(arguments.phase)
    parameters, table = build(data["s"].astype(float), data["phi"], arguments.sectors, arguments.offset)
    model = Spiral(parameters[0], parameters[1], arguments.sectors, *parameters[2:5])
    print(f"Pitch shape, um: {2 * np.pi * model.growth * 1e3 * np.array([1, 1 + sum(parameters[2:5])]).round(5)}")
    preset = {
        "id": arguments.id,
        "displayName": arguments.name,
        "description": f"Timing survey of the disc: {parameters[0]:.6f}-{parameters[1]:.6f} mm, "
                       f"{arguments.sectors} sectors, cubic pitch and phase table.",
        "sectors": arguments.sectors,
        "innerRadiusMm": float(parameters[0]),
        "outerRadiusMm": float(parameters[1]),
        "linearVelocityMmPerSecond": 1200,
        "pitchLinear": float(parameters[2]),
        "pitchQuadratic": float(parameters[3]),
        "pitchCubic": float(parameters[4]),
        "phaseCorrectionDegrees": [round(float(value), 3) for value in table],
    }
    text = json.dumps(preset, ensure_ascii=False, indent=2)
    if arguments.output:
        with open(arguments.output, "w", encoding="utf-8") as stream:
            stream.write(text + "\n")
        print(f"Saved {arguments.output}")
    else:
        print(text)


if __name__ == "__main__":
    main()
