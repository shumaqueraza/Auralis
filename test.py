"""Auralis live sound dashboard: run it, the browser opens, sounds get classified."""

from __future__ import annotations

import argparse
import json
import queue
import random
import sys
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from auralis import Detector

IGNORED = {"background"}


class Hub:
    """Latest detection plus fan-out to every connected browser."""

    def __init__(self, classes: list[str], mode: str, threshold: float) -> None:
        self.classes = classes
        self.mode = mode
        self.threshold = threshold
        self.latest: dict | None = None
        self.clients: set[queue.Queue] = set()
        self.lock = threading.Lock()

    def publish(self, class_name: str, confidence: float) -> None:
        if class_name in IGNORED:
            return
        event = {
            "class_name": class_name,
            "confidence": round(float(confidence), 3),
            "ts": round(time.time(), 3),
        }
        with self.lock:
            self.latest = event
            for client in list(self.clients):
                try:
                    client.get_nowait()
                except queue.Empty:
                    pass
                client.put_nowait(event)

    def subscribe(self) -> tuple[queue.Queue, dict]:
        client: queue.Queue = queue.Queue(maxsize=1)
        with self.lock:
            self.clients.add(client)
            return client, {
                "type": "hello",
                "classes": self.classes,
                "latest": self.latest,
                "mode": self.mode,
                "threshold": self.threshold,
            }

    def unsubscribe(self, client: queue.Queue) -> None:
        with self.lock:
            self.clients.discard(client)


PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="color-scheme" content="dark">
<title>Auralis</title>
<style>
@property --h { syntax: "<number>"; inherits: true; initial-value: 210; }

:root {
  --h: 210;
  --ink: #f4f3f0;
  --dim: #7b818a;
  --line: rgba(255, 255, 255, .09);
  --accent: hsl(var(--h) 80% 62%);
  --mono: ui-monospace, "Cascadia Mono", "SF Mono", Consolas, monospace;
  --sans: "Segoe UI Variable Display", "Segoe UI", system-ui, -apple-system, sans-serif;
  transition: --h .9s ease;
}

* { box-sizing: border-box; margin: 0; }

body {
  min-height: 100vh;
  color: var(--ink);
  font-family: var(--sans);
  background:
    radial-gradient(130% 80% at 50% -20%, hsl(var(--h) 70% 34% / .2), transparent 62%),
    #08090a;
  transition: background 1s ease;
}

.grid, .grain { position: fixed; inset: 0; pointer-events: none; }

.grid {
  background-image:
    linear-gradient(to right, var(--line) 1px, transparent 1px),
    linear-gradient(to bottom, var(--line) 1px, transparent 1px);
  background-size: 76px 76px;
  -webkit-mask-image: radial-gradient(130% 90% at 50% 0%, #000 25%, transparent 78%);
  mask-image: radial-gradient(130% 90% at 50% 0%, #000 25%, transparent 78%);
  opacity: .55;
}

.grain {
  opacity: .04;
  background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='160' height='160'%3E%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='.85' numOctaves='3'/%3E%3C/filter%3E%3Crect width='160' height='160' filter='url(%23n)'/%3E%3C/svg%3E");
}

main {
  position: relative;
  display: flex;
  flex-direction: column;
  gap: clamp(22px, 4vh, 40px);
  max-width: 1140px;
  min-height: 100vh;
  margin: 0 auto;
  padding: clamp(20px, 4vw, 46px);
}

header {
  display: flex;
  align-items: baseline;
  justify-content: space-between;
  gap: 16px;
  padding-bottom: 14px;
  border-bottom: 1px solid var(--line);
  font: 500 11px/1 var(--mono);
  letter-spacing: .2em;
  text-transform: uppercase;
  color: var(--dim);
}

header b { color: var(--ink); font-weight: 600; letter-spacing: .24em; }

.live { display: flex; align-items: center; gap: 8px; }

.live i {
  width: 6px;
  height: 6px;
  border-radius: 50%;
  background: #d8a13a;
  box-shadow: 0 0 10px #d8a13a;
}

.live.on i { background: var(--accent); box-shadow: 0 0 10px var(--accent); transition: background .9s; }

.stage {
  display: grid;
  grid-template-columns: minmax(0, 1.1fr) minmax(0, 1fr);
  gap: clamp(24px, 4vw, 54px);
  align-items: end;
}

.label {
  display: flex;
  align-items: center;
  gap: 8px;
  font: 500 10px/1 var(--mono);
  letter-spacing: .2em;
  text-transform: uppercase;
  color: var(--dim);
}

.label::before {
  content: "";
  width: 6px;
  height: 6px;
  background: var(--accent);
  transition: background .9s ease;
}

h1 {
  margin-top: 16px;
  font-size: clamp(34px, 6.6vw, 78px);
  font-weight: 700;
  line-height: .94;
  letter-spacing: -.035em;
  overflow-wrap: anywhere;
  opacity: 1;
  transition: opacity .22s ease;
}

.sub {
  margin-top: 12px;
  font: 500 11px/1 var(--mono);
  letter-spacing: .18em;
  text-transform: uppercase;
  color: var(--dim);
}

.meter {
  margin-top: 26px;
  height: 3px;
  background: rgba(255, 255, 255, .1);
}

.meter i {
  display: block;
  width: 0;
  height: 100%;
  background: var(--accent);
  box-shadow: 0 0 14px var(--accent);
  transition: width .45s cubic-bezier(.2, .8, .2, 1), background .9s ease;
}

.stats {
  display: flex;
  gap: 20px;
  margin-top: 12px;
  font: 500 11px/1 var(--mono);
  letter-spacing: .14em;
  text-transform: uppercase;
  color: var(--dim);
}

.stats b { color: var(--ink); font-weight: 600; }

.scope { border: 1px solid var(--line); padding: 16px 16px 12px; }

.bars {
  display: flex;
  align-items: flex-end;
  gap: 2px;
  height: clamp(84px, 17vh, 150px);
  margin-top: 16px;
}

.bars i {
  flex: 1;
  min-width: 1px;
  height: 3%;
  background: hsl(var(--h) 80% 62% / .85);
  transition: height .3s ease, background .9s ease;
}

.axis {
  display: flex;
  justify-content: space-between;
  margin-top: 10px;
  font: 500 9px/1 var(--mono);
  letter-spacing: .18em;
  text-transform: uppercase;
  color: var(--dim);
  opacity: .7;
}

.log { margin-top: 14px; border-top: 1px solid var(--line); }

.row {
  display: grid;
  grid-template-columns: 74px 8px minmax(0, 1fr) auto;
  gap: 14px;
  align-items: center;
  padding: 10px 2px;
  border-bottom: 1px solid rgba(255, 255, 255, .05);
  font: 500 11px/1 var(--mono);
  letter-spacing: .12em;
  text-transform: uppercase;
  animation: row-in .45s cubic-bezier(.2, .8, .2, 1) both;
}

.row .t, .row .c { color: var(--dim); }
.row .s { width: 8px; height: 8px; background: hsl(var(--h) 80% 62%); }
.row .n { color: var(--ink); overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }

.empty { padding: 14px 2px; font: 500 11px/1 var(--mono); letter-spacing: .12em; text-transform: uppercase; color: var(--dim); opacity: .6; }

@keyframes row-in { from { opacity: 0; transform: translateY(-10px); } }

@media (max-width: 880px) {
  .stage { grid-template-columns: minmax(0, 1fr); align-items: start; }
}

@media (max-width: 520px) {
  .row { grid-template-columns: 8px minmax(0, 1fr) auto; }
  .row .t { display: none; }
}
</style>
</head>
<body>
<div class="grid"></div>
<div class="grain"></div>
<main>
  <header>
    <div><b>Auralis</b> sound classifier</div>
    <div class="live" id="live"><i></i><span id="status">connecting</span></div>
  </header>

  <section class="stage">
    <div>
      <div class="label">detected class</div>
      <h1 id="class">&mdash;</h1>
      <div class="sub" id="sub">waiting for sound</div>
      <div class="meter"><i id="meter"></i></div>
      <div class="stats">
        <div>conf <b id="conf">0.00</b></div>
        <div>hits <b id="hits">0</b></div>
        <div>mode <b id="mode">&mdash;</b></div>
      </div>
    </div>

    <div class="scope">
      <div class="label">confidence trace</div>
      <div class="bars" id="bars"></div>
      <div class="axis"><span>live</span><span>history</span></div>
    </div>
  </section>

  <section>
    <div class="label">log</div>
    <div class="log" id="log"><div class="empty">no detections yet</div></div>
  </section>
</main>
<script>
const HUE = {
  clap: 348,
  finger_tap: 276,
  keyboard_mouse: 196,
  keys: 44,
  knock: 16,
  pen_click: 158,
  snaps: 306,
  speech: 254,
  whistle: 92,
};

const BARS = 56;
const el = (id) => document.getElementById(id);
const ui = {
  live: el("live"), status: el("status"), name: el("class"), sub: el("sub"),
  meter: el("meter"), conf: el("conf"), hits: el("hits"), mode: el("mode"),
  bars: el("bars"), log: el("log"),
};

const barPool = [];
for (let i = 0; i < BARS; i++) {
  const bar = document.createElement("i");
  ui.bars.append(bar);
  barPool.push(bar);
}

const trace = [];
let hits = 0;
let shown = null;

const hueOf = (name) => HUE[name] ?? 210;
const pretty = (name) => name.replace(/_/g, " ");

function drawTrace(confidence, hue) {
  trace.push({ confidence, hue });
  if (trace.length > BARS) trace.shift();
  const start = trace.length - BARS;
  barPool.forEach((bar, i) => {
    const point = trace[start + i];
    bar.style.height = point ? Math.max(4, point.confidence * 100) + "%" : "3%";
    bar.style.opacity = point ? ".9" : ".18";
    if (point) bar.style.setProperty("--h", point.hue);
  });
}

function addRow(event) {
  const empty = ui.log.querySelector(".empty");
  if (empty) empty.remove();

  const row = document.createElement("div");
  row.className = "row";
  row.style.setProperty("--h", hueOf(event.class_name));
  row.innerHTML =
    "<span class='t'>" + new Date(event.ts * 1000).toLocaleTimeString() + "</span>" +
    "<span class='s'></span>" +
    "<span class='n'>" + pretty(event.class_name) + "</span>" +
    "<span class='c'>" + event.confidence.toFixed(2) + "</span>";
  ui.log.prepend(row);

  while (ui.log.children.length > 7) ui.log.lastElementChild.remove();
}

function render(event) {
  if (!event || !event.class_name) return;

  const hue = hueOf(event.class_name);
  document.documentElement.style.setProperty("--h", hue);
  hits += 1;

  if (shown !== event.class_name) {
    ui.name.style.opacity = ".2";
    setTimeout(() => {
      ui.name.textContent = pretty(event.class_name);
      ui.name.style.opacity = "1";
    }, 90);
    ui.name.animate(
      [{ textShadow: "0 0 44px hsl(" + hue + " 90% 60% / .5)" }, { textShadow: "0 0 0 transparent" }],
      { duration: 750, easing: "ease-out" }
    );
    const total = (window.classes || []).length;
    const index = (window.classes || []).indexOf(event.class_name) + 1;
    ui.sub.textContent = "class " + String(index).padStart(2, "0") + " / " + total;
    shown = event.class_name;
  }

  ui.conf.textContent = event.confidence.toFixed(2);
  ui.hits.textContent = String(hits);
  ui.meter.style.width = Math.round(Math.min(1, Math.max(0, event.confidence)) * 100) + "%";

  drawTrace(event.confidence, hue);
  addRow(event);
}

function status(text, online) {
  ui.status.textContent = text;
  ui.live.classList.toggle("on", online);
}

const stream = new EventSource("/stream");
stream.onopen = () => status("streaming", true);
stream.onerror = () => status("reconnecting", false);
stream.onmessage = (message) => {
  const data = JSON.parse(message.data);
  if (data.type === "hello") {
    window.classes = data.classes;
    ui.mode.textContent = data.mode + " " + data.threshold.toFixed(2);
    status("streaming", true);
    if (data.latest) render(data.latest);
    return;
  }
  render(data);
};
</script>
</body>
</html>
"""


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "Auralis"
    sys_version = ""
    timeout = 30

    def do_GET(self) -> None:
        route = self.path.split("?")[0]
        if route == "/":
            self.respond(200, "text/html; charset=utf-8", PAGE.encode("utf-8"))
        elif route == "/stream":
            self.stream()
        elif route == "/favicon.ico":
            self.respond(204, "image/x-icon", b"")
        else:
            self.respond(404, "text/plain", b"not found")

    def stream(self) -> None:
        client, hello = self.server.hub.subscribe()
        try:
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream; charset=utf-8")
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            self.send_event(hello)
            while True:
                try:
                    event = client.get(timeout=10)
                except queue.Empty:
                    self.wfile.write(b": ping\n\n")
                    self.wfile.flush()
                    continue
                self.send_event(event)
        except OSError:
            pass
        finally:
            self.server.hub.unsubscribe(client)

    def send_event(self, payload: dict) -> None:
        self.wfile.write(f"data: {json.dumps(payload)}\n\n".encode("utf-8"))
        self.wfile.flush()

    def respond(self, code: int, mime: str, body: bytes) -> None:
        self.send_response(code)
        self.send_header("Content-Type", mime)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args) -> None:
        pass


class Server(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True
    hub: Hub

    def handle_error(self, request, client_address) -> None:
        error = sys.exc_info()[0]
        if error is not None and issubclass(error, OSError):
            return
        super().handle_error(request, client_address)


def demo_loop(stop: threading.Event, hub: Hub) -> None:
    rng = random.Random()
    sounds = [name for name in hub.classes if name not in IGNORED]
    while not stop.wait(rng.uniform(0.8, 2.2)):
        hub.publish(rng.choice(sounds), rng.uniform(0.5, 0.99))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Auralis live sound dashboard.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--confidence", type=float, default=0.5, help="minimum confidence to report")
    parser.add_argument("--no-browser", action="store_true")
    parser.add_argument("--demo", action="store_true", help="fake detections, no microphone")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    detector = Detector(confidence=args.confidence)

    try:
        server = Server((args.host, args.port), Handler)
    except OSError as exc:
        sys.exit(f"port {args.port} is busy: {exc}")

    server.hub = Hub(list(detector.classes), "demo" if args.demo else "mic", args.confidence)
    stop = threading.Event()

    if args.demo:
        threading.Thread(target=demo_loop, args=(stop, server.hub), daemon=True).start()
    else:
        try:
            detector.start(lambda result: server.hub.publish(result.class_name, result.confidence))
        except Exception as exc:
            server.server_close()
            sys.exit(f"cannot open the microphone: {exc}")

    threading.Thread(target=server.serve_forever, daemon=True).start()

    url = f"http://{args.host}:{args.port}/"
    print(f"Auralis live on {url}  (ctrl+c to stop)")
    if not args.no_browser:
        webbrowser.open(url)

    try:
        threading.Event().wait()
    except KeyboardInterrupt:
        print("\nstopping")
    finally:
        stop.set()
        detector.stop()
        server.shutdown()
        server.server_close()


if __name__ == "__main__":
    main()
