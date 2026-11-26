"""Synthetic |a| signal generator for the ball-centre accelerometer.

In free flight a sensor at the ball's centre reads ~0 g (no gravity in free fall,
no centripetal term at the spin axis), so contacts stand out as pulses.
Pulse shapes are modelling assumptions, to be replaced by recorded data (HTC-08).
"""
import numpy as np

FS = 1600.0


def half_sine(dur_ms, peak_g):
    n = max(2, int(round(dur_ms * FS / 1000)))
    return peak_g * np.sin(np.pi * np.arange(n) / (n - 1))


def plateau(dur_ms, level_g, ramp_ms=8):
    """Held ball: fast rise, long low plateau, fast release."""
    n = int(round(dur_ms * FS / 1000))
    r = int(round(ramp_ms * FS / 1000))
    x = np.full(n, level_g)
    x[:r] = np.linspace(0, level_g * 1.4, r)
    x[-r:] = np.linspace(level_g, 0, r)
    return x


def double_pulse(gap_ms, peak1, peak2, dur_ms=8):
    """Two hands one after another, the ball almost leaves between them."""
    gap = np.zeros(int(round(gap_ms * FS / 1000)))
    return np.concatenate([half_sine(dur_ms, peak1), gap, half_sine(dur_ms, peak2)])


def legal_two_hand_set(dur_ms=20, peak_g=18, skew_ms=3):
    """Two hands slightly out of sync but overlapping: one action, legal."""
    a = half_sine(dur_ms, peak_g * 0.6)
    s = int(round(skew_ms * FS / 1000))
    out = np.zeros(len(a) + s)
    out[:len(a)] += a
    out[s:] += a
    return out


class Timeline:
    def __init__(self, seed=0, noise_g=0.3):
        self.rng = np.random.default_rng(seed)
        self.noise = noise_g
        self.parts = []
        self.truth = []  # (start_s, end_s)

    @property
    def t(self):
        return sum(len(p) for p in self.parts) / FS

    def flight(self, ms):
        self.parts.append(np.zeros(int(round(ms * FS / 1000))))
        return self

    def contact(self, pulse):
        t0 = self.t
        self.parts.append(np.asarray(pulse, float))
        self.truth.append((t0, self.t))
        return self

    def signal(self):
        x = np.concatenate(self.parts + [np.zeros(int(0.05 * FS))])
        return np.abs(x + self.rng.normal(0, self.noise, len(x)))
