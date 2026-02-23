#!/usr/bin/env python3
"""
Raspberry Pi Camera Module 3 - Live Stream Server
Starte mit: python3 camera_server.py
Zugriff:    http://<raspberry-pi-ip>:8080
"""

import io
import time
import json
import threading
import logging
import numpy as np
import cv2
from http.server import BaseHTTPRequestHandler, HTTPServer
from socketserver import ThreadingMixIn
from picamera2 import Picamera2
from picamera2.encoders import MJPEGEncoder
from picamera2.outputs import FileOutput

# ── Konfiguration ──────────────────────────────────────────────────────────────
HOST = "0.0.0.0"   # alle Netzwerk-Interfaces (für LAN-Zugriff wichtig)
PORT = 8080
RESOLUTION = (1280, 720)   # 720p – änder auf (1920, 1080) für Full-HD
FRAMERATE = 30
# ──────────────────────────────────────────────────────────────────────────────

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# ── Gesture Detection State ────────────────────────────────────────────────────
gesture_lock = threading.Lock()
gesture_state = {
    "count": 0,
    "last_gesture": None,
    "detected": False,
    # Arm-Tracking: je [{hand: [x,y], elbow: [x,y], active: bool}, ...]
    # Koordinaten normiert auf 0.0–1.0 relativ zur Bildgröße
    "arms": [
        {"hand": None, "elbow": None, "active": False},
        {"hand": None, "elbow": None, "active": False},
    ],
}

def _skin_mask(frame):
    """Hautfarben-Maske im HSV-Farbraum, robust für verschiedene Hauttöne."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # Bereich 1: helle Hauttöne
    m1 = cv2.inRange(hsv, np.array([0,  20, 70],  dtype=np.uint8),
                          np.array([20, 255, 255], dtype=np.uint8))
    # Bereich 2: dunklere/rötliche Hauttöne
    m2 = cv2.inRange(hsv, np.array([160, 20, 70],  dtype=np.uint8),
                          np.array([180, 255, 255], dtype=np.uint8))
    mask = cv2.bitwise_or(m1, m2)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    return mask

def _largest_blob_center(mask, min_area=2000):
    """Gibt (cx, cy, area, contour) des größten Hautflecks zurück oder None."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    c = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(c)
    if area < min_area:
        return None
    M = cv2.moments(c)
    if M["m00"] == 0:
        return None
    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])
    return cx, cy, area, c

def _arm_axis(contour, img_w, img_h):
    """
    Berechnet die Arm-Achse via PCA auf den Kontur-Punkten.
    Gibt (hand_pt, elbow_pt) als normierte (x,y) Tupel zurück — oder None.

    Funktionsweise:
    - PCA liefert die Hauptachse des Hautblobs (= Arm-Richtung)
    - Entlang dieser Achse werden die zwei äußersten Punkte der Kontur gesucht
    - Der höhere Punkt (kleinere Y) = Hand, der tiefere = Ellenbogen/Unterarm
    - Durch ein Mindest-Aspektverhältnis wird sichergestellt dass der Blob
      tatsächlich länglich ist (Arm) und kein runder Fleck (Gesicht/Torso)
    """
    pts = contour.reshape(-1, 2).astype(np.float32)
    if len(pts) < 6:
        return None

    # Bounding Box Aspektverhältnis prüfen — Arm ist länger als breit
    x, y, w, h = cv2.boundingRect(contour)
    aspect = max(w, h) / max(min(w, h), 1)
    if aspect < 1.4:
        # Blob zu rund → wahrscheinlich Gesicht oder Torso, nicht Arm
        return None

    # PCA: Hauptachse des Blobs bestimmen
    mean, eigenvectors = cv2.PCACompute(pts, mean=None)
    main_axis = eigenvectors[0]  # Richtungsvektor der Hauptachse

    # Alle Konturpunkte auf die Hauptachse projizieren
    centered = pts - mean
    projections = centered @ main_axis

    # Extrempunkte entlang der Achse
    idx_min = np.argmin(projections)
    idx_max = np.argmax(projections)
    pt_a = pts[idx_min]
    pt_b = pts[idx_max]

    # Höherer Punkt (kleinere Y) = Hand, tieferer = Ellenbogen
    if pt_a[1] < pt_b[1]:
        hand_pt, elbow_pt = pt_a, pt_b
    else:
        hand_pt, elbow_pt = pt_b, pt_a

    return (
        (float(hand_pt[0]) / img_w,  float(hand_pt[1]) / img_h),
        (float(elbow_pt[0]) / img_w, float(elbow_pt[1]) / img_h),
    )

def gesture_detection_thread():
    """
    Erkennt die 6/7-Geste: abwechselndes Heben der linken und rechten Hand.
    Gibt zusätzlich Arm-Tracking-Daten für das visuelle Overlay zurück.
    """
    WINDOW      = 8
    RISE_PX     = 18
    MIN_AREA    = 1800
    COOLDOWN_F  = 12

    from collections import deque
    history = {0: deque(maxlen=WINDOW), 1: deque(maxlen=WINDOW)}
    last_side = None
    cooldown  = 0

    log.info("Gestenerkennung gestartet — abwechselndes Heben (6/7-Bewegung)")

    while True:
        with output.condition:
            output.condition.wait()
            frame_bytes = output.frame

        if frame_bytes is None:
            continue

        arr = np.frombuffer(frame_bytes, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            continue

        small = cv2.resize(img, (640, 360))
        h_img, w_img = small.shape[:2]
        mid = w_img // 2

        mask = _skin_mask(small)

        mask_l = mask.copy(); mask_l[:, mid:] = 0
        mask_r = mask.copy(); mask_r[:, :mid] = 0

        # Arm-Tracking-Daten sammeln
        arm_data = [
            {"hand": None, "elbow": None, "active": False},
            {"hand": None, "elbow": None, "active": False},
        ]

        for side, m in enumerate([mask_l, mask_r]):
            blob = _largest_blob_center(m, min_area=MIN_AREA)
            if blob:
                cx, cy, area, contour = blob
                history[side].append(cy)
                result = _arm_axis(contour, w_img, h_img)
                if result:
                    hand_pt, elbow_pt = result
                    arm_data[side]["hand"]  = list(hand_pt)
                    arm_data[side]["elbow"] = list(elbow_pt)
                else:
                    # Blob zu rund (Gesicht/Torso) → kein Arm-Overlay
                    arm_data[side]["hand"]  = None
                    arm_data[side]["elbow"] = None

        if cooldown > 0:
            cooldown -= 1
            with gesture_lock:
                gesture_state["detected"] = False
                gesture_state["arms"] = arm_data
            continue

        detected_side = None
        for side in [0, 1]:
            buf = history[side]
            if len(buf) < WINDOW:
                continue
            recent_avg = np.mean(list(buf)[WINDOW//2:])
            old_avg    = np.mean(list(buf)[:WINDOW//2])
            if old_avg - recent_avg >= RISE_PX:
                detected_side = side
                break

        if detected_side is not None:
            if last_side is None or detected_side != last_side:
                last_side = detected_side
                arm_data[detected_side]["active"] = True
                side_name = "Links" if detected_side == 0 else "Rechts"

                with gesture_lock:
                    gesture_state["count"] += 1
                    gesture_state["detected"] = True
                    gesture_state["last_gesture"] = time.strftime("%H:%M:%S")
                    gesture_state["arms"] = arm_data
                log.info(f"6/7 erkannt ({side_name})! Gesamt: {gesture_state['count']}")

                history[detected_side].clear()
                cooldown = COOLDOWN_F
            else:
                with gesture_lock:
                    gesture_state["detected"] = False
                    gesture_state["arms"] = arm_data
        else:
            with gesture_lock:
                gesture_state["detected"] = False
                gesture_state["arms"] = arm_data

# ──────────────────────────────────────────────────────────────────────────────


class ThreadingHTTPServer(ThreadingMixIn, HTTPServer):
    """HTTP-Server der jede Verbindung in einem eigenen Thread behandelt."""
    daemon_threads = True


class StreamOutput(io.BufferedIOBase):
    """Thread-sicherer Buffer für den MJPEG-Stream."""

    def __init__(self):
        self.frame = None
        self.condition = threading.Condition()

    def write(self, buf):
        with self.condition:
            self.frame = buf
            self.condition.notify_all()


output = StreamOutput()


class StreamHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        pass  # HTTP-Log unterdrücken – bei Bedarf entfernen

    def do_GET(self):
        if self.path == "/":
            self._serve_html()
        elif self.path == "/stream.mjpg":
            self._serve_stream()
        elif self.path == "/snapshot.jpg":
            self._serve_snapshot()
        elif self.path == "/gesture":
            self._serve_gesture()
        else:
            self.send_error(404)

    # ── HTML-Seite ─────────────────────────────────────────────────────────────
    def _serve_html(self):
        html = HTML_PAGE.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", len(html))
        self.end_headers()
        self.wfile.write(html)

    # ── MJPEG-Stream ───────────────────────────────────────────────────────────
    def _serve_stream(self):
        self.send_response(200)
        self.send_header("Age", "0")
        self.send_header("Cache-Control", "no-cache, private")
        self.send_header("Pragma", "no-cache")
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=FRAME")
        self.end_headers()
        try:
            while True:
                with output.condition:
                    output.condition.wait()
                    frame = output.frame
                self.wfile.write(b"--FRAME\r\n")
                self.send_header("Content-Type", "image/jpeg")
                self.send_header("Content-Length", len(frame))
                self.end_headers()
                self.wfile.write(frame)
                self.wfile.write(b"\r\n")
        except Exception:
            pass  # Client getrennt

    # ── Snapshot ───────────────────────────────────────────────────────────────
    def _serve_snapshot(self):
        with output.condition:
            output.condition.wait()
            frame = output.frame
        self.send_response(200)
        self.send_header("Content-Type", "image/jpeg")
        self.send_header("Content-Length", len(frame))
        self.send_header("Content-Disposition", 'attachment; filename="snapshot.jpg"')
        self.end_headers()
        self.wfile.write(frame)


    # ── Gesture API ────────────────────────────────────────────────────────────
    def _serve_gesture(self):
        with gesture_lock:
            data = dict(gesture_state)
        body = json.dumps(data).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", len(body))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)


# ── HTML-Seite (eingebettet) ───────────────────────────────────────────────────
HTML_PAGE = """<!DOCTYPE html>
<html lang="de">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>6/7 Challenge</title>
<style>
  @import url('https://fonts.googleapis.com/css2?family=Geist+Mono:wght@400;600;700&family=Geist:wght@300;400;500&display=swap');

  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

  body {
    background: #09090b;
    color: #fafafa;
    font-family: 'Geist', sans-serif;
    min-height: 100vh;
    display: flex;
    align-items: center;
    justify-content: center;
    -webkit-font-smoothing: antialiased;
  }

  /* ── Video Wrapper ── */
  .video-wrap {
    position: relative;
    width: min(90vw, calc(90vh * 16 / 9));
    aspect-ratio: 16 / 9;
    border-radius: 20px;
    overflow: hidden;
    box-shadow: 0 0 0 1px rgba(255,255,255,.08), 0 32px 80px rgba(0,0,0,.7);
  }

  .video-wrap img {
    width: 100%;
    height: 100%;
    object-fit: cover;
    display: block;
  }

  /* ── Zähler oben mittig ── */
  .counter {
    position: absolute;
    top: 24px;
    left: 50%;
    transform: translateX(-50%);
    font-family: 'Geist Mono', monospace;
    font-size: clamp(3rem, 10vw, 7rem);
    font-weight: 700;
    color: #ef4444;
    line-height: 1;
    text-shadow: 0 2px 20px rgba(0,0,0,.8);
    pointer-events: none;
    transition: transform 0.08s ease;
    user-select: none;
  }

  .counter.pop {
    transform: translateX(-50%) scale(1.25);
  }

  /* ── Timer oben links ── */
  .timer-box {
    position: absolute;
    top: 20px;
    left: 20px;
    background: rgba(0,0,0,.55);
    backdrop-filter: blur(8px);
    border: 1px solid rgba(255,255,255,.12);
    border-radius: 12px;
    padding: 10px 18px;
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 2px;
    pointer-events: none;
  }

  .timer-label {
    font-family: 'Geist Mono', monospace;
    font-size: 10px;
    font-weight: 600;
    letter-spacing: 0.12em;
    color: rgba(255,255,255,.45);
    text-transform: uppercase;
  }

  .timer-val {
    font-family: 'Geist Mono', monospace;
    font-size: 2rem;
    font-weight: 700;
    color: #fafafa;
    line-height: 1;
    transition: color 0.3s;
  }

  .timer-val.urgent { color: #ef4444; }

  /* Timer-Ring */
  .timer-ring {
    position: absolute;
    top: 14px;
    left: 14px;
    width: 80px;
    height: 80px;
    pointer-events: none;
  }

  .timer-ring circle {
    fill: none;
    stroke: rgba(255,255,255,.12);
    stroke-width: 3;
  }

  .timer-ring .progress {
    stroke: #fafafa;
    stroke-linecap: round;
    stroke-dasharray: 220;
    stroke-dashoffset: 0;
    transform: rotate(-90deg);
    transform-origin: center;
    transition: stroke-dashoffset 0.5s linear, stroke 0.3s;
  }

  .timer-ring .progress.urgent { stroke: #ef4444; }

  /* ── LIVE badge unten rechts ── */
  .live-badge {
    position: absolute;
    bottom: 18px;
    right: 18px;
    display: flex;
    align-items: center;
    gap: 6px;
    background: rgba(0,0,0,.5);
    backdrop-filter: blur(6px);
    border: 1px solid rgba(255,255,255,.1);
    border-radius: 9999px;
    padding: 5px 12px;
    font-family: 'Geist Mono', monospace;
    font-size: 11px;
    font-weight: 600;
    color: rgba(255,255,255,.7);
    letter-spacing: 0.08em;
    pointer-events: none;
  }

  .live-dot {
    width: 6px;
    height: 6px;
    border-radius: 50%;
    background: #22c55e;
    animation: blink 2s ease-in-out infinite;
  }

  @keyframes blink {
    0%, 100% { opacity: 1; }
    50%       { opacity: 0.25; }
  }

  /* ── Fertig-Overlay ── */
  .done-overlay {
    position: absolute;
    inset: 0;
    background: rgba(0,0,0,.7);
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    gap: 12px;
    opacity: 0;
    pointer-events: none;
    transition: opacity 0.4s;
    border-radius: 20px;
  }

  .done-overlay.show {
    opacity: 1;
    pointer-events: auto;
  }

  .done-score {
    font-family: 'Geist Mono', monospace;
    font-size: clamp(4rem, 14vw, 9rem);
    font-weight: 700;
    color: #ef4444;
    line-height: 1;
  }

  .done-label {
    font-size: 1.1rem;
    color: rgba(255,255,255,.6);
    letter-spacing: 0.04em;
  }

  .done-btn {
    margin-top: 12px;
    padding: 10px 28px;
    background: #fafafa;
    color: #09090b;
    border: none;
    border-radius: 9999px;
    font-family: 'Geist', sans-serif;
    font-size: 14px;
    font-weight: 600;
    cursor: pointer;
    letter-spacing: -0.01em;
    transition: opacity 0.15s;
  }

  .done-btn:hover { opacity: 0.85; }

  /* ── Arm-Canvas ── */
  #arm-canvas {
    position: absolute;
    inset: 0;
    width: 100%;
    height: 100%;
    pointer-events: none;
    border-radius: 20px;
  }
</style>
</head>
<body>

<div class="video-wrap">
  <img id="stream-img" src="/stream.mjpg" alt="Live">

  <!-- Arm-Tracking Canvas (liegt über dem Video) -->
  <canvas id="arm-canvas"></canvas>

  <!-- Zähler -->
  <div class="counter" id="counter">0</div>

  <!-- Timer -->
  <svg class="timer-ring" viewBox="0 0 80 80">
    <circle cx="40" cy="40" r="35"/>
    <circle class="progress" id="timer-ring-progress" cx="40" cy="40" r="35"/>
  </svg>
  <div class="timer-box">
    <span class="timer-label">Zeit</span>
    <span class="timer-val" id="timer-val">20</span>
  </div>

  <!-- LIVE -->
  <div class="live-badge">
    <div class="live-dot"></div>
    LIVE
  </div>

  <!-- Fertig-Overlay -->
  <div class="done-overlay" id="done-overlay">
    <div class="done-score" id="done-score">0</div>
    <div class="done-label">6/7 in 20 Sekunden</div>
    <button class="done-btn" onclick="resetGame()">Nochmal</button>
  </div>
</div>

<script>
  const TIMER_DURATION = 20;
  const CIRCUMFERENCE  = 2 * Math.PI * 35;

  let gestureOffset   = 0;
  let localCount      = 0;
  let timerActive     = false;
  let timerEnd        = null;
  let gameOver        = false;
  let lastServerCount = 0;
  let lastArms        = null;

  const counterEl   = document.getElementById('counter');
  const timerValEl  = document.getElementById('timer-val');
  const ringEl      = document.getElementById('timer-ring-progress');
  const doneOverlay = document.getElementById('done-overlay');
  const doneScore   = document.getElementById('done-score');
  const canvas      = document.getElementById('arm-canvas');
  const ctx         = canvas.getContext('2d');

  ringEl.style.strokeDasharray  = CIRCUMFERENCE;
  ringEl.style.strokeDashoffset = 0;

  // ── Canvas-Größe mit Video synchron halten ────────────────────────────────
  const wrap = document.querySelector('.video-wrap');
  function resizeCanvas() {
    canvas.width  = wrap.clientWidth;
    canvas.height = wrap.clientHeight;
  }
  resizeCanvas();
  new ResizeObserver(resizeCanvas).observe(wrap);

  // ── Arm-Overlay zeichnen ──────────────────────────────────────────────────
  function drawArms(arms) {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    if (!arms) return;

    arms.forEach(arm => {
      if (!arm.hand || !arm.elbow) return;

      const hx = arm.hand[0]  * canvas.width;
      const hy = arm.hand[1]  * canvas.height;
      const ex = arm.elbow[0] * canvas.width;
      const ey = arm.elbow[1] * canvas.height;

      const color  = arm.active ? '#ff2222' : 'rgba(255,60,60,0.75)';
      const lwidth = arm.active ? 5 : 3;

      // Unterarm-Linie (Ellenbogen → Hand)
      ctx.beginPath();
      ctx.moveTo(ex, ey);
      ctx.lineTo(hx, hy);
      ctx.strokeStyle = color;
      ctx.lineWidth   = lwidth;
      ctx.lineCap     = 'round';
      ctx.shadowColor = '#ff0000';
      ctx.shadowBlur  = arm.active ? 14 : 6;
      ctx.stroke();

      // Ellenbogen-Punkt
      ctx.beginPath();
      ctx.arc(ex, ey, arm.active ? 8 : 5, 0, Math.PI * 2);
      ctx.fillStyle   = color;
      ctx.shadowBlur  = arm.active ? 18 : 8;
      ctx.fill();

      // Hand-Punkt
      ctx.beginPath();
      ctx.arc(hx, hy, arm.active ? 6 : 4, 0, Math.PI * 2);
      ctx.fillStyle  = color;
      ctx.shadowBlur = arm.active ? 14 : 6;
      ctx.fill();

      ctx.shadowBlur = 0;
    });
  }

  // ── Gesture Polling ───────────────────────────────────────────────────────
  async function pollGesture() {
    try {
      const res  = await fetch('/gesture');
      const data = await res.json();

      // Arm-Overlay immer aktualisieren (auch während gameOver)
      lastArms = data.arms || null;
      drawArms(lastArms);

      if (gameOver) return;

      const serverCount = data.count;
      const newCount    = Math.max(0, serverCount - gestureOffset);

      // Erste Geste → Timer starten
      if (!timerActive && newCount > 0) {
        timerActive = true;
        timerEnd    = Date.now() + TIMER_DURATION * 1000;
        requestAnimationFrame(timerTick);
      }

      // Zähler aktualisieren & Pop-Animation
      if (newCount !== localCount) {
        localCount = newCount;
        counterEl.textContent = localCount;
        counterEl.classList.remove('pop');
        void counterEl.offsetWidth;
        counterEl.classList.add('pop');
        setTimeout(() => counterEl.classList.remove('pop'), 150);
      }

      lastServerCount = serverCount;
    } catch(e) {}
  }

  // ── Timer Loop ────────────────────────────────────────────────────────────
  function timerTick() {
    if (!timerActive || gameOver) return;

    const remaining = Math.max(0, (timerEnd - Date.now()) / 1000);
    timerValEl.textContent = Math.ceil(remaining);

    const fraction = remaining / TIMER_DURATION;
    ringEl.style.strokeDashoffset = CIRCUMFERENCE * (1 - fraction);

    if (remaining <= 5) {
      timerValEl.classList.add('urgent');
      ringEl.classList.add('urgent');
    }

    if (remaining <= 0) {
      endGame();
      return;
    }
    requestAnimationFrame(timerTick);
  }

  function endGame() {
    gameOver = true;
    doneScore.textContent = localCount;
    doneOverlay.classList.add('show');
  }

  function resetGame() {
    gestureOffset = lastServerCount;
    localCount    = 0;
    timerActive   = false;
    timerEnd      = null;
    gameOver      = false;

    counterEl.textContent  = '0';
    timerValEl.textContent = TIMER_DURATION;
    timerValEl.classList.remove('urgent');
    ringEl.classList.remove('urgent');
    ringEl.style.strokeDashoffset = 0;
    doneOverlay.classList.remove('show');
  }

  setInterval(pollGesture, 150);
</script>
</body>
</html>
"""


# ── Server starten ─────────────────────────────────────────────────────────────
def main():
    picam2 = Picamera2()
    config = picam2.create_video_configuration(
        main={"size": RESOLUTION},
        controls={"FrameRate": FRAMERATE}
    )
    picam2.configure(config)
    picam2.start_recording(MJPEGEncoder(), FileOutput(output))

    # Gestenerkennung im Hintergrund starten
    t = threading.Thread(target=gesture_detection_thread, daemon=True)
    t.start()

    server = ThreadingHTTPServer((HOST, PORT), StreamHandler)
    log.info(f"Server läuft auf http://0.0.0.0:{PORT}")
    log.info(f"Lokaler Zugriff:    http://localhost:{PORT}")
    log.info(f"Netzwerk-Zugriff:   http://<Raspberry-Pi-IP>:{PORT}")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        log.info("Server gestoppt.")
    finally:
        picam2.stop_recording()


if __name__ == "__main__":
    main()
