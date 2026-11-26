"""Reference fault-detection algorithm for the smart volleyball (Ball Unit firmware model).

This Python model is the executable specification of the firmware logic
(SW-02 .. SW-07, SW-15). The C firmware on the nRF52840 must match its outputs
on the same input data.
"""
from dataclasses import dataclass, field
from enum import Enum
import numpy as np


class Fault(str, Enum):
    LEGAL = "LEGAL"
    DOUBLE_CONTACT = "DOUBLE_CONTACT"
    CATCH = "CATCH"
    FOUR_HITS = "FOUR_HITS"


@dataclass
class Params:  # SW-15: every threshold is a calibration parameter
    fs: float = 1600.0               # sample rate, Hz
    start_g: float = 2.5             # contact threshold on 2 ms-smoothed |a|, g (SW-02)
    end_quiet_ms: float = 15.0       # below threshold this long = contact end (SW-02)
    catch_min_ms: float = 40.0       # catch: longer than this ... (SW-05)
    catch_peak_ratio: float = 2.5    # ... and peak/mean below this (SW-05)
    double_gap_ms: float = 300.0     # separate contacts closer than this = double (SW-06)
    valley_ratio: float = 0.25       # in-contact valley < this x smaller peak = double (SW-06)
    min_peak_g: float = 10.0         # a sub-peak must exceed this to count


@dataclass
class Contact:
    start: float      # s
    end: float        # s
    peak_g: float
    mean_g: float
    n_peaks: int

    @property
    def duration_ms(self):
        return (self.end - self.start) * 1000.0


def segment(acc_g: np.ndarray, p: Params) -> list[Contact]:
    """Split |a| samples into contacts (SW-02) and extract features (SW-03)."""
    k = max(1, int(round(0.002 * p.fs)))
    smooth = np.convolve(acc_g, np.ones(k) / k, mode="same")
    above = smooth > p.start_g
    quiet = max(1, int(round(p.end_quiet_ms * p.fs / 1000.0)))
    contacts, i, n = [], 0, len(acc_g)
    while i < n:
        if not above[i]:
            i += 1
            continue
        s = i
        last_above = i
        j = i
        while j < n and j - last_above < quiet:
            if above[j]:
                last_above = j
            j += 1
        e = last_above + 1
        seg = acc_g[s:e]
        contacts.append(Contact(s / p.fs, e / p.fs, float(seg.max()), float(seg.mean()),
                                _count_peaks(seg, p)))
        i = j
    return contacts


def _count_peaks(seg: np.ndarray, p: Params) -> int:
    """Count sub-impacts separated by a deep valley (two hands hitting one after the other)."""
    k = max(3, int(p.fs * 0.002))  # 2 ms smoothing
    sm = np.convolve(seg, np.ones(k) / k, mode="same")
    peaks = [i for i in range(1, len(sm) - 1)
             if sm[i] >= sm[i - 1] and sm[i] > sm[i + 1] and sm[i] > p.min_peak_g]
    if not peaks:
        return 1
    count, last = 1, peaks[0]
    for pk in peaks[1:]:
        valley = sm[last:pk + 1].min()
        if valley < p.valley_ratio * min(sm[last], sm[pk]):
            count += 1
            last = pk
        elif sm[pk] > sm[last]:
            last = pk
    return count


@dataclass
class Decision:
    contact: Contact
    fault: Fault
    decided_at: float        # s, when the firmware can send the alert
    hit_no: int              # team hit number in the possession


@dataclass
class RuleEngine:
    """Possession tracking and FIVB 9.2-9.3 rules (SW-05..SW-08)."""
    p: Params = field(default_factory=Params)
    hits: int = 0
    prev_hit: Contact | None = None
    prev_was_block: bool = False

    def possession_change(self):          # wristband button or net-crossing estimate
        self.hits, self.prev_hit, self.prev_was_block = 0, None, False

    def on_contact(self, c: Contact, is_block: bool = False) -> Decision:
        decided_at = c.end + self.p.end_quiet_ms / 1000.0
        if is_block:                      # FIVB 9.4.1: block is not a team hit
            self.prev_hit, self.prev_was_block = c, True
            return Decision(c, Fault.LEGAL, decided_at, 0)
        gap_ms = (c.start - self.prev_hit.end) * 1000.0 if self.prev_hit else float("inf")
        quick = gap_ms < self.p.double_gap_ms and not self.prev_was_block
        if self.hits == 1 and quick:
            first = True                  # FIVB 9.2.3.2: consecutive contacts in the first team contact
        else:
            self.hits += 1
            first = self.hits == 1
        fault = Fault.LEGAL
        if c.duration_ms > self.p.catch_min_ms and c.peak_g / max(c.mean_g, 1e-6) < self.p.catch_peak_ratio:
            fault = Fault.CATCH
        elif not first and (c.n_peaks >= 2 or quick):
            fault = Fault.DOUBLE_CONTACT
        elif self.hits >= 4:
            fault = Fault.FOUR_HITS
        self.prev_hit, self.prev_was_block = c, False
        return Decision(c, fault, decided_at, self.hits)
