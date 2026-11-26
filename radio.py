"""Alert packet format and secure link model (SW-09, SW-11)."""
import struct
from cryptography.hazmat.primitives.ciphers.aead import AESCCM
from cryptography.exceptions import InvalidTag

FAULT_CODES = {"DOUBLE_CONTACT": 1, "CATCH": 2, "FOUR_HITS": 3}
CODES_FAULT = {v: k for k, v in FAULT_CODES.items()}
# payload: fault(u8) contact_id(u16) confidence(u8, %) timestamp_ms(u32)
FMT = "<BHBI"


class Link:
    def __init__(self, ball_id: int, key: bytes):
        self.ball_id, self.aead, self.seq = ball_id, AESCCM(key, tag_length=8), 0

    def _nonce(self, ball_id, seq):
        return struct.pack("<IQ", ball_id, seq)[:13].ljust(13, b"\0")

    def encode(self, fault, contact_id, confidence, ts_ms) -> bytes:
        self.seq += 1
        hdr = struct.pack("<IQ", self.ball_id, self.seq)
        body = struct.pack(FMT, FAULT_CODES[fault], contact_id, confidence, ts_ms)
        return hdr + self.aead.encrypt(self._nonce(self.ball_id, self.seq), body, hdr)


class Receiver:
    def __init__(self, paired_ball_id: int, key: bytes):
        self.ball_id, self.aead, self.last_seq = paired_ball_id, AESCCM(key, tag_length=8), 0

    def decode(self, pkt: bytes):
        """Return (fault, contact_id, confidence, ts_ms) or None if rejected."""
        if len(pkt) < 12:
            return None
        ball_id, seq = struct.unpack("<IQ", pkt[:12])
        if ball_id != self.ball_id or seq <= self.last_seq:   # foreign ball or replay
            return None
        try:
            body = self.aead.decrypt(struct.pack("<IQ", ball_id, seq)[:13].ljust(13, b"\0"), pkt[12:], pkt[:12])
        except InvalidTag:
            return None
        self.last_seq = seq
        code, cid, conf, ts = struct.unpack(FMT, body)
        return CODES_FAULT[code], cid, conf, ts


def send_with_retry(pkt, channel, receiver, max_tries=4):
    """Send, resend up to 3 times until ACK (SW-09). Returns tries used or None."""
    for attempt in range(1, max_tries + 1):
        delivered = channel(pkt)
        if delivered is not None and receiver.decode(delivered) is not None:
            return attempt
    return None
