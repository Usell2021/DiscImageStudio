"""Spiral model shared by the probe scripts; it matches DvdTrackGeometry and CdDiscParameters.

dr/dtheta = b * P(x), P(x) = 1 + c1 x + c2 x^2 + c3 x^3, x = (r - r0) / (r1 - r0), and b comes
from the track length. Angles are radians, positions are fractional sectors from LBA 0.
"""
import json
import numpy as np

SECTOR_LENGTH_MM = 16.0  # 1200 mm/s / 75 sectors per second


class Spiral:
    def __init__(self, r0, r1, sectors, c1=0.0, c2=0.0, c3=0.0, grid=6001):
        self.r0, self.r1, self.sectors = r0, r1, sectors
        self.coefficients = (c1, c2, c3)
        radius = np.linspace(r0, r1, grid)
        step = radius[1] - radius[0]
        middle = 0.5 * (radius[:-1] + radius[1:])

        def shape(values):
            x = (values - r0) / (r1 - r0)
            return 1 + c1 * x + c2 * x * x + c3 * x ** 3

        pitch, pitch_middle = shape(radius), shape(middle)
        if (pitch <= 0).any():
            raise ValueError("The pitch shape must stay positive.")
        length = sectors * SECTOR_LENGTH_MM
        growth = np.sum(middle / pitch_middle) * step / length
        for _ in range(4):  # the exact arc depends weakly on b, so a few passes settle it
            per_radius = np.sqrt(radius ** 2 + (growth * pitch) ** 2) / pitch
            per_radius_middle = np.sqrt(middle ** 2 + (growth * pitch_middle) ** 2) / pitch_middle
            integral = np.concatenate(([0], np.cumsum(
                (per_radius[:-1] + 4 * per_radius_middle + per_radius[1:]) / 6 * step)))
            growth = integral[-1] / length
        angle = np.concatenate(([0], np.cumsum(
            (1 / pitch[:-1] + 4 / pitch_middle + 1 / pitch[1:]) / 6 * step))) / growth
        self.growth, self._radius, self._arc, self._angle = growth, radius, integral / growth, angle
        self._pitch, self._arc_rate = pitch, per_radius / growth

    def theta(self, sector):
        arc = np.asarray(sector, float) * SECTOR_LENGTH_MM
        index = np.clip(np.searchsorted(self._arc, arc) - 1, 0, len(self._arc) - 2)
        step = self._radius[1] - self._radius[0]
        rest = arc - self._arc[index]
        rate0, rate1 = self._arc_rate[index], self._arc_rate[index + 1]
        t = rest / (0.5 * (rate0 + rate1) * step)
        for _ in range(4):
            t = t - (step * (rate0 * t + 0.5 * (rate1 - rate0) * t * t) - rest) / (step * (rate0 + (rate1 - rate0) * t))
        slope0 = 1 / (self.growth * self._pitch[index])
        slope1 = 1 / (self.growth * self._pitch[index + 1])
        return self._angle[index] + step * (slope0 * t + 0.5 * (slope1 - slope0) * t * t)

    def radius(self, sector):
        return np.interp(np.asarray(sector, float) * SECTOR_LENGTH_MM, self._arc, self._radius)


def fit_cubic(sectors, angles, count, start=(24.9, 57.8, 0.0, 0.0, 0.0, 0.0)):
    """Levenberg-Marquardt on (r0, r1, c1, c2, c3, constant). Returns (parameters, residual rad)."""
    def residual(p):
        return Spiral(p[0], p[1], count, p[2], p[3], p[4]).theta(sectors) + p[5] - angles

    parameters = np.array(start, float)
    current = residual(parameters)
    cost = (current ** 2).sum()
    damping = 1e-3
    for _ in range(100):
        jacobian = np.stack(
            [(residual(parameters + 1e-6 * np.eye(6)[i]) - current) / 1e-6 for i in range(6)], 1)
        normal, gradient = jacobian.T @ jacobian, jacobian.T @ current
        for _ in range(25):
            trial = parameters + np.linalg.solve(normal + damping * np.diag(np.diag(normal)), -gradient)
            try:
                candidate = residual(trial)
                trial_cost = (candidate ** 2).sum()
            except ValueError:
                trial_cost = np.inf
            if trial_cost < cost:
                parameters, current, cost = trial, candidate, trial_cost
                damping = max(damping / 5, 1e-13)
                break
            damping *= 5
        else:
            break
    return parameters, current


def load_preset(path, preset_id):
    with open(path, encoding="utf-8") as stream:
        document = json.load(stream)
    return next(item for item in document["cdPresets"] if item["id"] == preset_id)


def image_angle(geometry, sector):
    """Angle the generator uses for a sector: spiral plus the optional phase table.

    `geometry` is a preset or a track sidecar. Both use the same field names.
    """
    spiral = Spiral(
        geometry["innerRadiusMm"], geometry["outerRadiusMm"], geometry["sectors"],
        geometry.get("pitchLinear", 0.0), geometry.get("pitchQuadratic", 0.0),
        geometry.get("pitchCubic", 0.0))
    angle = spiral.theta(sector)
    table = geometry.get("phaseCorrectionDegrees")
    if table:
        nodes = np.linspace(0, geometry["sectors"], len(table))
        angle = angle + np.radians(np.interp(sector, nodes, table))
    return angle, spiral


def fold_period_ms(times):
    """True rotation period from completion times, between 5 and 100 ms."""
    t = np.asarray(times, float)
    t = t - t[0]

    def scan(low, high, count):
        periods = np.linspace(low, high, count)
        strength = np.abs(np.exp(2j * np.pi * t[None, :] / periods[:, None]).mean(1))
        return periods, strength

    periods, strength = scan(0.005, 0.100, 60000)
    # Sub-multiples of the period concentrate as fully as the period; multiples do not because the
    # pauses are random. Side lobes reach about 0.85, so the threshold has to stay above them.
    period = periods[strength > 0.97 * strength.max()].max()
    for width, count in ((1e-2, 8000), (1e-3, 4000), (1e-4, 4000)):
        periods, strength = scan(period * (1 - width), period * (1 + width), count)
        period = periods[strength.argmax()]
    return period * 1e3


def wrap(turns):
    return (turns + 0.5) % 1 - 0.5
