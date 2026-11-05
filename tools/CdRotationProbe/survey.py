#!/usr/bin/env python3
"""Measures sector angles on a fully written CD. Read only. Appends raw samples to a JSONL file.

coarse: a chain of pairs every 3000 sectors, plus ladders of offsets that let solve.py count whole
        turns, plus long pairs kept out of the solution as a check. About 15 minutes.
dense:  sweeps every 500 sectors from anchors every 20000, plus long baselines. About 17 minutes.
"""
import argparse
import json
import random
import time

from sg_io import Drive, sample_period_times

LADDER = (0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048)


class Recorder:
    def __init__(self, drive, path, sectors):
        self.drive, self.sectors = drive, sectors
        self.output = open(path, "a", buffering=1, encoding="utf-8")

    def _write(self, **record):
        self.output.write(json.dumps({**record, "t": time.time()}) + "\n")

    def period(self, lba):
        self._write(k="period", A=lba, ts=sample_period_times(self.drive, lba, self.sectors))

    def pair(self, a, b, repeats, kind, **extra):
        """Reads a and b in turn. Each read has to seek, so the drive cache never answers."""
        self.drive.read_sector(b)
        differences = []
        for _ in range(repeats):
            ok_a, time_a = self.drive.read_sector(a)
            ok_b, time_b = self.drive.read_sector(b)
            if ok_a and ok_b:
                differences.append((time_b - time_a) / 1e6)
        self._write(k=kind, A=a, B=b, dt=differences, **extra)


def coarse(recorder):
    last = recorder.sectors - 1
    sites = list(range(0, last, 3000))
    if sites[-1] != last:
        sites.append(last)
    recorder.period(2000)
    for index, site in enumerate(sites):
        if index % 30 == 15:
            recorder.period(site)
        if index + 1 < len(sites):
            recorder.pair(site, sites[index + 1], 6, "chain")
        if index % 4 == 0:
            sign = 1 if site + 3000 + LADDER[-1] < recorder.sectors else -1
            for delta in LADDER:
                recorder.pair(site, site + sign * (3000 + delta), 4, "ladder", site=site, sign=sign, delta=delta)
        print(f"{index + 1}/{len(sites)} sector {site}", flush=True)
    recorder.period(recorder.sectors // 2)
    generator = random.Random(12345)
    for _ in range(40):
        a = generator.randrange(0, recorder.sectors - 200000)
        recorder.pair(a, a + generator.randrange(20000, recorder.sectors - a - 1), 6, "closure")
    recorder.period(recorder.sectors - 10000)


def dense(recorder):
    anchors = list(range(0, recorder.sectors - 3000, 20000))
    recorder.period(2000)
    for index, anchor in enumerate(anchors):
        if index % 4 == 2:
            recorder.period(anchor)
        for lba in range(anchor + 3000, min(anchor + 23500, recorder.sectors - 1), 500):
            recorder.pair(anchor, lba, 4, "fine")
        print(f"{index + 1}/{len(anchors)} anchor {anchor}", flush=True)
    recorder.period(recorder.sectors // 2)
    for gap in (1, 2, 4, 8, 16):
        for index in range(len(anchors) - gap):
            recorder.pair(anchors[index], anchors[index + gap], 6, "long")
    recorder.pair(anchors[0], recorder.sectors - 1, 6, "long")
    recorder.pair(anchors[-1], recorder.sectors - 1, 6, "long")
    recorder.period(recorder.sectors - 10000)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("mode", choices=("coarse", "dense"))
    parser.add_argument("output", help="JSONL file, appended to")
    parser.add_argument("--sectors", type=int, required=True, help="Lead-out LBA from `cdrecord -toc`")
    parser.add_argument("--device", default="/dev/sr0")
    parser.add_argument("--speed", type=int, default=16, help="Requested CAV speed, x")
    arguments = parser.parse_args()
    drive = Drive(arguments.device)
    drive.require_disc()
    if not drive.read_sector(1000)[0]:
        raise SystemExit("Sector 1000 cannot be read. The survey needs a fully written disc.")
    drive.set_cav(arguments.speed)
    recorder = Recorder(drive, arguments.output, arguments.sectors)
    (coarse if arguments.mode == "coarse" else dense)(recorder)
    print("done")


if __name__ == "__main__":
    main()
