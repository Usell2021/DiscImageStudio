#!/usr/bin/env python3
"""Answers whether a written disc matches an existing preset. Read only, about two minutes.

A disc from another stamper gives nearly random phases, about 104 deg rms.
"""
import argparse
import os
import random

import numpy as np

from sg_io import Drive, sample_period_times
from spiral import fold_period_ms, image_angle, load_preset, wrap

BUILT_IN = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..",
                        "src", "DiscImageStudio.Core", "Presets", "built-in-disc-presets.json")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset", default="noname-cd-rw-infodisc")
    parser.add_argument("--presets-file", default=BUILT_IN)
    parser.add_argument("--offset", type=float, default=0.947, help="Offset the preset was built with, sectors")
    parser.add_argument("--pairs", type=int, default=60)
    parser.add_argument("--device", default="/dev/sr0")
    parser.add_argument("--speed", type=int, default=16)
    arguments = parser.parse_args()
    preset = load_preset(arguments.presets_file, arguments.preset)
    sectors = preset["sectors"]
    drive = Drive(arguments.device)
    drive.require_disc()
    if not drive.read_sector(1000)[0]:
        raise SystemExit("Sector 1000 cannot be read. An empty disc has nothing to measure.")
    drive.set_cav(arguments.speed)
    period = fold_period_ms(sample_period_times(drive, sectors // 2, sectors))
    print(f"Rotation period {period:.5f} ms")

    def expected(sector):
        return image_angle(preset, np.float64(sector) + arguments.offset)[0] / (2 * np.pi)

    generator = random.Random(4242)
    errors = []
    for _ in range(arguments.pairs):
        a = generator.randrange(0, sectors - 30000)
        b = a + generator.randrange(3000, min(sectors - a - 1, 120000))
        drive.read_sector(b)
        differences = []
        for _ in range(6):
            time_a = drive.read_sector(a)[1]
            time_b = drive.read_sector(b)[1]
            differences.append((time_b - time_a) / 1e6)
        vector = np.exp(2j * np.pi * np.array(differences) / period).mean()
        measured = (np.angle(vector) / (2 * np.pi)) % 1
        errors.append(wrap(measured - (expected(b) - expected(a))) * 360)
    errors = np.array(errors)
    rms = np.sqrt((errors ** 2).mean())
    print(f"{len(errors)} pairs against {arguments.preset}: rms {rms:.1f}, max {np.abs(errors).max():.1f} deg")
    if rms < 5:
        print("The disc matches the preset.")
    elif rms < 30:
        print("Close, but not the same geometry. Expect visible bending; measure this disc.")
    else:
        print("The disc does NOT match the preset. Expect rings; measure this disc.")


if __name__ == "__main__":
    main()
