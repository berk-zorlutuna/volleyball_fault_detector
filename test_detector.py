"""Software test cases STC-01 .. STC-16. Run: python -m pytest -v"""
import os
import numpy as np
import pytest
from detector import Params, RuleEngine, Fault, segment
from simulator import Timeline, half_sine, plateau, double_pulse, legal_two_hand_set, FS
from radio import Link, Receiver, send_with_retry

P = Params()


def classify(tl, blocks=(), changes=(), p=P):
    """Segment the signal, feed the rule engine. blocks/changes = contact indexes."""
    eng = RuleEngine(p)
    out = []
    for i, c in enumerate(segment(tl.signal(), p)):
        if i in changes:
            eng.possession_change()
        out.append(eng.on_contact(c, is_block=i in blocks))
    return out


def faults(ds):
    return [d.fault for d in ds]


# --- segmentation -------------------------------------------------------
def test_stc01_single_hit_segmented():
    tl = Timeline(1).flight(200).contact(half_sine(10, 80)).flight(300)
    cs = segment(tl.signal(), P)
    assert len(cs) == 1
    t0, t1 = tl.truth[0]
    # threshold crossing trims the low tails of the pulse; allow 3 ms
    assert abs(cs[0].start - t0) * 1000 < 3 and abs(cs[0].end - t1) * 1000 < 3


def test_stc02_noise_only_no_contacts():
    tl = Timeline(2, noise_g=0.5).flight(60_000)   # 60 s of flight noise
    assert segment(tl.signal(), P) == []


# --- single-contact classification ---------------------------------------
def test_stc03_spike_legal():
    tl = Timeline(3).flight(100).contact(half_sine(9, 140)).flight(500)
    assert faults(classify(tl)) == [Fault.LEGAL]


def test_stc04_two_hand_set_legal():
    tl = Timeline(4).flight(800).contact(half_sine(10, 40)).flight(900).contact(legal_two_hand_set()).flight(500)
    assert faults(classify(tl)) == [Fault.LEGAL, Fault.LEGAL]


def test_stc05_catch_detected():
    tl = Timeline(5).flight(800).contact(plateau(90, 5)).flight(500)
    assert faults(classify(tl)) == [Fault.CATCH]


def test_stc06_double_within_contact():
    tl = (Timeline(6).flight(500).contact(half_sine(10, 40)).flight(900)
          .contact(double_pulse(4, 20, 18)).flight(500))
    assert faults(classify(tl)) == [Fault.LEGAL, Fault.DOUBLE_CONTACT]


def test_stc07_double_separate_contacts():
    tl = (Timeline(7).flight(500).contact(half_sine(10, 40)).flight(900)
          .contact(half_sine(12, 20)).flight(150).contact(half_sine(10, 15)).flight(500))
    assert faults(classify(tl)) == [Fault.LEGAL, Fault.LEGAL, Fault.DOUBLE_CONTACT]


# --- rule exemptions -------------------------------------------------------
def test_stc08_first_contact_exempt():
    tl = (Timeline(8).flight(500).contact(half_sine(10, 60)).flight(120).contact(half_sine(10, 25))
          .flight(900).contact(double_pulse(4, 20, 18)).flight(500))
    # reception double (contacts 0,1) is legal; a later double by the setter is not
    tl2 = Timeline(9).flight(500).contact(double_pulse(4, 30, 25)).flight(500)
    assert faults(classify(tl))[:2] == [Fault.LEGAL, Fault.LEGAL]
    assert faults(classify(tl2)) == [Fault.LEGAL]


def test_stc09_four_hits():
    tl = Timeline(10).flight(300)
    for _ in range(4):
        tl.contact(half_sine(10, 40)).flight(900)
    assert faults(classify(tl)) == [Fault.LEGAL] * 3 + [Fault.FOUR_HITS]


def test_stc10_block_not_counted():
    tl = Timeline(11).flight(300).contact(half_sine(8, 60)).flight(250)  # block, then quick dig
    for _ in range(3):
        tl.contact(half_sine(10, 40)).flight(900)
    assert faults(classify(tl, blocks={0})) == [Fault.LEGAL] * 4


def test_stc11_possession_change_resets_count():
    tl = Timeline(12).flight(300)
    for _ in range(5):
        tl.contact(half_sine(10, 40)).flight(900)
    assert faults(classify(tl, changes={3})) == [Fault.LEGAL] * 5


# --- radio -----------------------------------------------------------------
KEY = bytes(range(16))


def test_stc12_packet_roundtrip_and_security():
    tx, rx = Link(0xBA11, KEY), Receiver(0xBA11, KEY)
    pkt = tx.encode("CATCH", 17, 92, 123456)
    assert rx.decode(pkt) == ("CATCH", 17, 92, 123456)
    assert rx.decode(pkt) is None                           # replay rejected
    bad = bytearray(tx.encode("CATCH", 18, 92, 1)); bad[-1] ^= 1
    assert rx.decode(bytes(bad)) is None                    # tampered rejected
    assert Receiver(0xBA11, KEY).decode(Link(0xBEEF, KEY).encode("CATCH", 1, 1, 1)) is None  # foreign ball
    assert Receiver(0xBA11, KEY).decode(Link(0xBA11, bytes(16)).encode("CATCH", 1, 1, 1)) is None  # wrong key


def test_stc13_retry_delivers():
    tx, rx = Link(1, KEY), Receiver(1, KEY)
    drops = iter([True, True, False])
    assert send_with_retry(tx.encode("FOUR_HITS", 1, 99, 5), lambda p: None if next(drops) else p, rx) == 3
    assert send_with_retry(tx.encode("FOUR_HITS", 2, 99, 6), lambda p: None, rx) is None


# --- timing ----------------------------------------------------------------
def test_stc14_decision_latency():
    tl = (Timeline(13).flight(500).contact(half_sine(10, 40)).flight(900).contact(plateau(90, 5))
          .flight(900).contact(double_pulse(4, 20, 18)).flight(500))
    for d in classify(tl):
        assert (d.decided_at - d.contact.end) * 1000 <= 60


# --- statistics (SYS-04 / SYS-05 on simulated data) -------------------------
def monte_carlo(n, seed=42):
    rng = np.random.default_rng(seed)
    res = {k: [0, 0] for k in ("legal", "CATCH", "DOUBLE_CONTACT")}
    for i in range(n):
        kind = ("legal", "CATCH", "DOUBLE_CONTACT")[i % 3]
        tl = Timeline(int(rng.integers(1e9)), noise_g=rng.uniform(0.2, 0.8)).flight(500)
        tl.contact(half_sine(rng.uniform(8, 14), rng.uniform(25, 120))).flight(rng.uniform(600, 1200))
        if kind == "legal":
            pick = rng.integers(3)
            pulse = [half_sine(rng.uniform(7, 14), rng.uniform(20, 150)),
                     legal_two_hand_set(rng.uniform(15, 30), rng.uniform(12, 30), rng.uniform(0, 5)),
                     half_sine(rng.uniform(25, 38), rng.uniform(10, 20))][pick]
            want = Fault.LEGAL
        elif kind == "CATCH":
            pulse, want = plateau(rng.uniform(55, 150), rng.uniform(3.5, 9)), Fault.CATCH
        else:
            pulse, want = double_pulse(rng.uniform(2, 15), rng.uniform(14, 40), rng.uniform(14, 40)), Fault.DOUBLE_CONTACT
        tl.contact(pulse).flight(500)
        got = classify(tl)
        fs = faults(got)
        # a slow double touch (gap > end_quiet_ms) is segmented as two contacts; either way
        # exactly one alert of the right type must follow the legal first contact
        alerts = [f for f in fs if f != Fault.LEGAL]
        ok = fs[0] == Fault.LEGAL and (alerts == [] if want == Fault.LEGAL else alerts == [want])
        res[kind][0] += ok
        res[kind][1] += 1
    return res


def test_stc15_monte_carlo_accuracy():
    n = int(os.environ.get("MC_N", "3000"))
    res = monte_carlo(n)
    for k, (ok, tot) in res.items():
        print(f"{k}: {ok}/{tot} = {100 * ok / tot:.2f} %")
    assert res["CATCH"][0] / res["CATCH"][1] >= 0.95
    assert res["DOUBLE_CONTACT"][0] / res["DOUBLE_CONTACT"][1] >= 0.95
    assert res["legal"][0] / res["legal"][1] >= 0.98


def test_stc16_thresholds_configurable():
    tl = Timeline(14).flight(800).contact(plateau(80, 5)).flight(500)
    assert faults(classify(tl)) == [Fault.CATCH]
    tl = Timeline(14).flight(800).contact(plateau(80, 5)).flight(500)
    assert faults(classify(tl, p=Params(catch_min_ms=100))) == [Fault.LEGAL]
