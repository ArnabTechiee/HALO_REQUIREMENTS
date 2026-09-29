"""
HALO status LEDs: one file, two roles
======================================
Pi (owns the GPIO pins): run as the LED server and leave it running:
    python led_status.py
ServerApp (laptop): imported; it only sends small HTTP updates to the Pi:
    from pytorchexample import led_status

Node LEDs 1-4 = partition 0-3. An LED goes off when that laptop's SuperNode
disconnects and comes back on when it reconnects. The big LED shows phase/round.
gpiozero is imported only in server mode, and the client half never raises.
"""
from __future__ import annotations

import json
import os
import queue
import threading
import time
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

NUM_SLOTS = 4
NODE_PINS = [17, 27, 22, 23]   # physical pins 11, 13, 15, 16
STATUS_PIN = 18                # physical pin 12 (big LED)
PORT = int(os.environ.get("HALO_LED_PORT", "5050"))
LED_URL = os.environ.get("HALO_LED_URL", f"http://100.118.174.94:{PORT}/led")
PHASES = ("idle", "waiting", "train", "evaluate", "done")
STALE_AFTER = 20.0             # Pi: no update this long -> ServerApp gone -> idle

# ═══════════════════ client half (ServerApp on the laptop) ═══════════════════
POLL_SECONDS = 2.0
KEEPALIVE_SECONDS = 5.0

_lock = threading.Lock()
_phase = {"phase": "idle", "round": 0, "total": 0}
_last_flags = None
_monitor_started = False
_sender_started = False
_q: queue.Queue = queue.Queue(maxsize=100)


def _post(payload: dict) -> None:
    try:
        req = urllib.request.Request(
            LED_URL, data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"}, method="POST")
        urllib.request.urlopen(req, timeout=1.0).close()
    except Exception:  # noqa: BLE001 - LEDs must never break training
        pass


def _sender() -> None:
    while True:
        _post(_q.get())


def _send(payload: dict) -> None:
    global _sender_started
    with _lock:
        if not _sender_started:
            _sender_started = True
            threading.Thread(target=_sender, daemon=True, name="halo-led-sender").start()
    try:
        _q.put_nowait(payload)
    except queue.Full:
        pass


def set_phase(phase: str, server_round: int | None = None,
              total_rounds: int | None = None) -> None:
    """Tell the big LED what training is doing."""
    with _lock:
        _phase["phase"] = phase
        if server_round is not None:
            _phase["round"] = int(server_round)
        if total_rounds is not None:
            _phase["total"] = int(total_rounds)
        payload = dict(_phase)
    _send(payload)


def _flags_for(online_ids, strategy) -> list[bool]:
    """LED i is on if the device that owns partition i is online.

    Identity comes from AdaptiveFedAvg (node_device, device_partition), so a
    laptop keeps its own LED across reconnects. A node not identified yet
    (e.g. just reconnected, before the next round's probe) takes the lowest
    free LED, which is normally its own, since that one just went off."""
    node_dev = dict(getattr(strategy, "node_device", None) or {})
    dev_part = dict(getattr(strategy, "device_partition", None) or {})
    flags = [False] * NUM_SLOTS
    unknown = 0
    for nid in online_ids:
        part = dev_part.get(node_dev.get(nid))
        if part is not None and 0 <= int(part) < NUM_SLOTS and not flags[int(part)]:
            flags[int(part)] = True
        else:
            unknown += 1
    for _ in range(unknown):
        free = next((i for i, on in enumerate(flags) if not on), None)
        if free is None:
            break
        flags[free] = True
    return flags


def start_monitor(grid, strategy=None) -> None:
    """Poll the SuperLink every 2s for connected nodes; push LED state to the Pi."""
    global _monitor_started
    with _lock:
        if _monitor_started:
            return
        _monitor_started = True

    def loop():
        global _last_flags
        last_keepalive = 0.0
        while True:
            try:
                flags = _flags_for(list(grid.get_node_ids()), strategy)
            except Exception:  # noqa: BLE001 - skip this poll, try again in 2s
                flags = None
            now = time.time()
            with _lock:
                changed = flags is not None and flags != _last_flags
                if flags is not None:
                    _last_flags = flags
                due = now - last_keepalive >= KEEPALIVE_SECONDS
                payload = dict(_phase)
                if _last_flags is not None:
                    payload["nodes"] = _last_flags
            if changed or due:
                _send(payload)
                last_keepalive = now
            time.sleep(POLL_SECONDS)

    threading.Thread(target=loop, daemon=True, name="halo-led-monitor").start()


# ═══════════════════ server half (Raspberry Pi) ═══════════════════
class _Board:
    def __init__(self):
        from gpiozero import LED, PWMLED  # Pi only
        self.node_leds = [LED(p) for p in NODE_PINS]
        self.big = PWMLED(STATUS_PIN)
        self.lock = threading.Lock()
        self.wake = threading.Event()
        self.state = {"nodes": [False] * NUM_SLOTS, "phase": "idle",
                      "round": 0, "total": 0, "t": 0.0, "alert_until": 0.0}

    def update(self, data: dict, alert: bool = True) -> None:
        wake = False
        with self.lock:
            s = self.state
            s["t"] = time.time()
            if isinstance(data.get("nodes"), list):
                new = [bool(x) for x in data["nodes"]][:NUM_SLOTS]
                new += [False] * (NUM_SLOTS - len(new))
                if alert and any(old and not n for old, n in zip(s["nodes"], new)):
                    s["alert_until"] = time.time() + 1.5   # a laptop dropped
                    wake = True
                s["nodes"] = new
            if data.get("phase") in PHASES and data["phase"] != s["phase"]:
                s["phase"] = data["phase"]
                wake = True
            for k in ("round", "total"):
                if isinstance(data.get(k), int) and data[k] != s[k]:
                    s[k] = data[k]
                    wake = True
            nodes = list(s["nodes"])
        for led, on in zip(self.node_leds, nodes):
            led.on() if on else led.off()
        if wake:
            self.wake.set()

    # ── big LED patterns ──
    def _wait(self, sec: float) -> bool:
        """Sleep; return True early if the state changed."""
        if self.wake.wait(sec):
            self.wake.clear()
            return True
        return False

    def _flash(self, on: float, off: float, level: float = 1.0) -> bool:
        self.big.value = level
        if self._wait(on):
            return True
        self.big.value = 0
        return self._wait(off)

    def _breathe(self, period: float, peak: float) -> bool:
        steps = 40
        for i in range(steps * 2):
            x = i / steps if i < steps else 2 - i / steps
            self.big.value = peak * x
            if self._wait(period / (steps * 2)):
                return True
        return False

    def _signature(self, rnd: int) -> bool:
        """Round number as flashes: long = 10, short = 1 (round 12 = 1 long + 2 short)."""
        tens, units = divmod(max(rnd, 0), 10)
        for _ in range(tens):
            if self._flash(0.9, 0.3):
                return True
        for _ in range(units):
            if self._flash(0.15, 0.25):
                return True
        return self._wait(0.8)

    def run(self) -> None:
        while True:
            with self.lock:
                s = dict(self.state)
            now = time.time()
            if (s["phase"] != "idle" or any(s["nodes"])) and now - s["t"] > STALE_AFTER:
                self.update({"phase": "idle", "nodes": [False] * NUM_SLOTS}, alert=False)
                continue
            if now < s["alert_until"]:
                self._flash(0.05, 0.05)                       # strobe: node dropped
                continue
            ph = s["phase"]
            if ph == "idle":
                self._breathe(4.0, 0.25)                      # slow dim breathing
            elif ph == "waiting":
                self._flash(0.2, 1.8, 0.6)                    # blink every 2s
            elif ph == "train":
                if self._signature(s["round"]):
                    continue
                self.big.value = min(1.0, 0.1 + 0.9 * s["round"] / max(s["total"], 1))
                self._wait(3.0)                               # brighter = further along
            elif ph == "evaluate":
                if not self._flash(0.1, 0.15):
                    self._flash(0.1, 0.9)                     # double blink
            elif ph == "done":
                if not any(self._flash(0.8, 0.4) for _ in range(3)):
                    self.big.value = 1.0                      # solid 8s, then idle
                    if not self._wait(8.0):
                        self.update({"phase": "idle"}, alert=False)
            else:
                self._wait(0.5)


def serve() -> None:
    board = _Board()

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            if self.path != "/led":
                self.send_error(404)
                return
            try:
                length = int(self.headers.get("Content-Length", 0))
                data = json.loads(self.rfile.read(length) or b"{}")
                if not isinstance(data, dict):
                    raise ValueError
            except ValueError:
                self.send_error(400)
                return
            board.update(data)
            self.send_response(204)
            self.end_headers()

        def do_GET(self):
            with board.lock:
                body = json.dumps(board.state).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    threading.Thread(target=board.run, daemon=True, name="halo-led-patterns").start()
    server = ThreadingHTTPServer(("0.0.0.0", PORT), Handler)
    print(f"[HALO] LED server on :{PORT}; node pins {NODE_PINS}, big LED GPIO{STATUS_PIN}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        board.big.off()
        for led in board.node_leds:
            led.off()


if __name__ == "__main__":
    serve()
