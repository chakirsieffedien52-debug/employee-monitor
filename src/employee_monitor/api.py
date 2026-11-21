"""FastAPI status and daily-report dashboard."""

import threading
from collections.abc import Mapping, Sequence
from datetime import date
from typing import Dict

from employee_monitor.reporting import ReportExporter

_DASHBOARD = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>Employee Monitor</title>
<style>
:root{color-scheme:light;--ink:#17212b;--muted:#607182;--line:#dce4ec;--panel:#fff;--bg:#f4f7fa}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.5 system-ui,sans-serif}
header{background:#18324b;color:#fff;padding:1.2rem max(1rem,calc((100vw - 1100px)/2))}
main{max-width:1100px;margin:2rem auto;padding:0 1rem}.toolbar{display:flex;gap:.7rem;align-items:center;margin-bottom:1rem}
button,a{border:1px solid var(--line);border-radius:8px;background:#fff;color:var(--ink);padding:.55rem .8rem;text-decoration:none}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:1rem}
.camera{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:1rem}
.desk{display:flex;justify-content:space-between;border-top:1px solid var(--line);padding:.55rem 0}
.status{font-weight:700}.WORKING{color:#16803c}.ABSENT,.PHONE_USAGE{color:#bd3030}
small{color:var(--muted)}table{width:100%;border-collapse:collapse;background:white}td,th{text-align:left;padding:.6rem;border-bottom:1px solid var(--line)}
</style></head>
<body><header><strong>Employee Monitor</strong><br><small>Local camera status dashboard</small></header>
<main><div class="toolbar"><button id="refresh">Refresh</button>
<a href="/api/report/export?format=csv">Export CSV</a>
<a href="/api/report/export?format=html">Export HTML</a>
<a href="/api/report/export?format=pdf">Export PDF</a>
<small id="updated"></small></div><section id="cameras" class="grid"></section>
<h2>Today's status events</h2><table><thead><tr><th>Time</th><th>Camera</th><th>Desk</th><th>Status</th><th>Duration (s)</th></tr></thead>
<tbody id="events"></tbody></table></main>
<script>
async function refresh(){const [status,report]=await Promise.all([
fetch('/api/status').then(r=>r.json()),fetch('/api/report/daily').then(r=>r.json())]);
const escapeHTML=value=>String(value).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
document.querySelector('#updated').textContent='Updated '+new Date().toLocaleTimeString();
document.querySelector('#cameras').innerHTML=Object.entries(status.cameras).map(([name,c])=>
`<article class="camera"><h2>${escapeHTML(name)}</h2><small>${c.connected?'Connected':'Reconnecting'} · ${c.fps.toFixed(1)} FPS</small>
${Object.entries(c.statuses).map(([desk,value])=>`<div class="desk"><span>${escapeHTML(desk)}</span><span class="status ${escapeHTML(value).replace(' ','_')}">${escapeHTML(value)} · ${Number((c.durations[desk]||{})[value]||0).toFixed(1)}s</span></div>`).join('')}</article>`).join('');
document.querySelector('#events').innerHTML=report.events.slice(-30).reverse().map(e=>
`<tr><td>${escapeHTML(e.timestamp)}</td><td>${escapeHTML(e.camera)}</td><td>${escapeHTML(e.desk)}</td><td>${escapeHTML(e.status)}</td><td>${escapeHTML(e.duration_seconds||'')}</td></tr>`).join('');
}
document.querySelector('#refresh').addEventListener('click',refresh);refresh();setInterval(refresh,3000);
</script></body></html>"""


class LiveState:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._cameras: Dict[str, dict] = {}

    def update(
        self,
        camera: str,
        statuses: Mapping[str, str],
        durations: Mapping[str, Mapping[str, float]],
        fps: float,
        connected: bool,
    ) -> None:
        with self._lock:
            self._cameras[camera] = {
                "statuses": dict(statuses),
                "durations": {desk: dict(values) for desk, values in durations.items()},
                "fps": float(fps),
                "connected": bool(connected),
            }

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "cameras": {name: dict(value) for name, value in self._cameras.items()}
            }


def create_app(
    state: LiveState,
    report_exporter: ReportExporter,
    events_by_camera: Mapping[str, Sequence[dict]],
):
    try:
        from fastapi import FastAPI, HTTPException, Query
        from fastapi.responses import FileResponse, HTMLResponse
    except ImportError as exc:
        raise RuntimeError(
            "The dashboard requires FastAPI. Install the project dependencies."
        ) from exc

    app = FastAPI(title="Employee Monitor", docs_url=None, redoc_url=None)

    @app.get("/", response_class=HTMLResponse)
    def dashboard():
        return _DASHBOARD

    @app.get("/api/status")
    def status():
        return state.snapshot()

    @app.get("/api/report/daily")
    def daily_report():
        return {
            "date": date.today().isoformat(),
            "cameras": state.snapshot()["cameras"],
            "events": [event for rows in events_by_camera.values() for event in rows],
        }

    @app.get("/api/report/export")
    def export_report(format: str = Query("csv", pattern="^(csv|html|pdf)$")):
        snapshot = state.snapshot()["cameras"]
        durations = {
            camera: values.get("durations", {}) for camera, values in snapshot.items()
        }
        paths = report_exporter.export_daily(events_by_camera, durations)
        path = paths[format]
        media_types = {
            "csv": "text/csv",
            "html": "text/html",
            "pdf": "application/pdf",
        }
        if not path.is_file():
            raise HTTPException(
                status_code=500, detail="Report export was not created."
            )
        return FileResponse(path, media_type=media_types[format], filename=path.name)

    return app
