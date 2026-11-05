#!/usr/bin/env python3
"""Turns survey samples into a phase profile: unwrapped sector angle over the whole disc.

The coarse survey counts whole turns: ladders give the local slope, which predicts each chain link
well enough to round it to whole turns. The dense survey then measures against that prediction and
is solved by least squares over all pairs. Output: NPZ with `s` (sector) and `phi` (radians).
"""
import argparse
import json

import numpy as np

from spiral import Spiral, fit_cubic, fold_period_ms, wrap

MAX_READ_DEVIATION = np.radians(25)  # rare single reads land a whole sector away


def load(path):
    with open(path, encoding="utf-8") as stream:
        records = [json.loads(line) for line in stream]
    periods = [(r["t"], fold_period_ms(r["ts"])) for r in records if r["k"] == "period"]
    print(f"{path}: period ms {np.round([p[1] for p in periods], 5).tolist()}")
    if max(p[1] for p in periods) / min(p[1] for p in periods) > 1.001:
        raise SystemExit("The rotation period drifted by more than 0.1%, so the drive was not in CAV.")
    return records, np.array([p[0] for p in periods]), np.array([p[1] for p in periods])


def phase(record, period_times, period_values):
    """Returns (phase in turns, number of reads kept), or None without enough consistent reads."""
    period = float(np.interp(record["t"], period_times, period_values))
    vectors = np.exp(2j * np.pi * np.array(record["dt"]) / period)
    centre = np.angle(vectors.mean())
    keep = np.abs(np.angle(vectors * np.exp(-1j * centre))) < MAX_READ_DEVIATION
    if keep.sum() < 2:
        return None
    return (np.angle(vectors[keep].mean()) / (2 * np.pi)) % 1, int(keep.sum())


def unwrap_coarse(path, sectors, r0, r1):
    records, times, values = load(path)
    prior = Spiral(r0, r1, sectors)

    def prior_turns(s):
        return prior.theta(s) / (2 * np.pi)

    ladders = {}
    for record in records:
        if record["k"] == "ladder":
            ladders.setdefault(record["site"], {})[record["delta"]] = record
    positions, slopes = [], []
    for site, rungs in sorted(ladders.items()):
        base = rungs[0]
        base_phase = phase(base, times, values)[0]
        slope = 0.0
        x, y = [], []
        for delta in sorted(rungs):
            if delta == 0:
                continue
            measured = phase(rungs[delta], times, values)
            if measured is None:
                continue
            span = rungs[delta]["B"] - base["B"]
            error = wrap(measured[0] - base_phase - (prior_turns(rungs[delta]["B"]) - prior_turns(base["B"])))
            x.append(span)
            y.append(slope * span + wrap(error - slope * span))
            slope = (np.array(x) * np.array(y)).sum() / (np.array(x) ** 2).sum()
        positions.append(base["B"] + base["sign"] * 1024)
        slopes.append(slope)
    order = np.argsort(positions)
    positions, slopes = np.array(positions)[order], np.array(slopes)[order]

    sector, error = [], []
    worst = 0.0
    for record in sorted((r for r in records if r["k"] == "chain"), key=lambda r: r["A"]):
        a, b = record["A"], record["B"]
        if not sector:
            sector, error = [a], [0.0]
        expected = float(np.interp(0.5 * (a + b), positions, slopes)) * (b - a)
        step = expected + wrap(phase(record, times, values)[0] - (prior_turns(b) - prior_turns(a)) - expected)
        worst = max(worst, abs(step - expected))
        sector.append(b)
        error.append(error[-1] + step)
    sector, error = np.array(sector, float), np.array(error)
    print(f"Chain: {len(sector) - 1} links, largest deviation from prediction {worst:.3f} turn")
    if worst > 0.25:
        raise SystemExit("A chain link is too far from its prediction to count whole turns reliably.")
    angle = (prior_turns(sector) + error) * 2 * np.pi
    return sector, angle - angle[0], records, times, values


def solve_dense(path, sector, angle, sectors):
    """Least squares over all pairs, measured against the coarse profile."""
    parameters, _ = fit_cubic(sector, angle, sectors)
    model = Spiral(parameters[0], parameters[1], sectors, *parameters[2:5])
    rest = angle / (2 * np.pi) - model.theta(sector) / (2 * np.pi)

    def predicted(s):
        return model.theta(s) / (2 * np.pi) + np.interp(s, sector, rest)

    records, times, values = load(path)
    observations = []
    for record in records:
        if record["k"] not in ("fine", "long"):
            continue
        measured = phase(record, times, values)
        if measured is not None:
            a, b = record["A"], record["B"]
            observations.append((a, b, wrap(measured[0] - (predicted(b) - predicted(a))), measured[1]))
    a, b, difference, weight = (np.array(column, float) for column in zip(*observations))
    print(f"Dense: {len(a)} pairs, disagreement with the coarse profile "
          f"rms {np.degrees(np.sqrt((difference ** 2).mean()) * 2 * np.pi):.2f} deg")
    nodes = np.unique(np.r_[a, b])
    index = {node: i for i, node in enumerate(nodes)}
    design = np.zeros((len(a), len(nodes)))
    for row, (first, second) in enumerate(zip(a, b)):
        design[row, index[second]], design[row, index[first]] = 1, -1
    root = np.sqrt(weight)
    weighted = design * root[:, None]
    correction = np.linalg.solve(weighted.T @ weighted + 1e-9 * np.eye(len(nodes)), weighted.T @ (difference * root))
    phi = (predicted(nodes) + correction) * 2 * np.pi
    return nodes, phi - phi[0]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("coarse", help="JSONL from `survey.py coarse`")
    parser.add_argument("dense", help="JSONL from `survey.py dense`")
    parser.add_argument("output", help="Output NPZ")
    parser.add_argument("--sectors", type=int, required=True)
    parser.add_argument("--prior-r0", type=float, default=25.0, help="Starting guess, mm")
    parser.add_argument("--prior-r1", type=float, default=58.0, help="Starting guess, mm")
    arguments = parser.parse_args()
    sector, angle, records, times, values = unwrap_coarse(
        arguments.coarse, arguments.sectors, arguments.prior_r0, arguments.prior_r1)
    print(f"Coarse total: {angle[-1] / (2 * np.pi):.3f} turns")
    parameters, residual = fit_cubic(sector, angle, arguments.sectors)
    model = Spiral(parameters[0], parameters[1], arguments.sectors, *parameters[2:5])
    print(f"Coarse cubic fit: rms {np.degrees(residual.std()):.2f}, max {np.degrees(np.abs(residual).max()):.2f} deg")
    closure = []
    for record in records:
        if record["k"] == "closure":
            measured = phase(record, times, values)
            if measured is not None:
                closure.append(wrap(measured[0] - (model.theta(record["B"]) - model.theta(record["A"])) / (2 * np.pi)) * 360)
    if closure:
        print(f"Withheld long pairs: {len(closure)}, rms {np.sqrt(np.mean(np.square(closure))):.2f}, "
              f"max {np.abs(closure).max():.2f} deg")
    nodes, phi = solve_dense(arguments.dense, sector, angle, arguments.sectors)
    print(f"Dense total: {phi[-1] / (2 * np.pi):.3f} turns, {len(nodes)} nodes")
    np.savez(arguments.output, s=nodes, phi=phi)
    print(f"Saved {arguments.output}")


if __name__ == "__main__":
    main()
