# Usage

## Install and start

From the project directory on Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
employee-monitor --config configs/config.example.yaml
```

The example config opens webcam index `0` in detect mode. On first inference, Ultralytics downloads the configured model weights if they are absent. The weights and runtime files are not tracked by Git.

## Camera sources

Configure one or more sources in YAML. A source may be a webcam index, a video file, or an RTSP/HTTP URL:

```yaml
cameras:
  - name: office
    source: 0
  - name: training-video
    source: ../videos/consented-demo.mp4
```

The example path above is relative to a config file under `configs/`. Video paths are resolved relative to the selected config file's directory. Check a source before running inference:

```powershell
employee-monitor --test-camera 0 --config configs/config.example.yaml
employee-monitor --test-camera ../videos/consented-demo.mp4 --config configs/config.example.yaml
```

For private RTSP credentials, use an environment variable rather than committing a URL:

```yaml
cameras:
  - name: office
    source: ${CAMERA_RTSP_URL}
```

Create a local `.env` next to the selected configuration file and set `CAMERA_RTSP_URL`. The provided `.env.example` contains only an empty variable placeholder. Do not commit the real `.env`.

## CLI options

- `--headless`: process without an OpenCV window.
- `--grid`: show all configured camera previews in one window.
- `--dashboard`: serve the dashboard (loopback by default).
- `--clear-desks`: remove remembered-desk JSON files for configured cameras before starting.
- `--no-desk-memory`: disable desk-memory loading, learning, and absence alerts for the run.
- `--max-frames N`: stop after N processed frames per camera; useful for smoke checks.
- `--test-camera SOURCE`: inspect a source and exit.
- `select-zones --camera NAME`: open the optional interactive selector for the legacy `desks` mode.

For a headless local-video run, create a temporary config whose camera source points to a short, consented video, then use:

```powershell
employee-monitor --headless --max-frames 30 --config path\to\local-test.yaml
```

## Desk memory and absence alerts

In detect mode, desk memory is enabled by default. A nearly stationary tracked person can establish a remembered area after `desks_memory.min_still_seconds`. If an established area is empty, the app logs an absence after `absent_after_seconds` and escalates after `urgent_after_seconds`. Never-occupied areas do not raise absence alerts. Desk data is stored per camera under the configured desk directory (default `desks/`).

Tune thresholds in YAML for a controlled, consented environment. `forget_after_minutes: 0` retains a desk indefinitely. `--clear-desks` deletes configured remembered desk files at startup; review the target config before using it. An optional webhook is configured with `desks_memory.webhook_url`; keep private webhook URLs out of source control.

## Dashboard and output files

The dashboard is optional and binds to `127.0.0.1:8000` by default. It has no authentication; do not bind it to a public or untrusted network. When reports are generated, daily CSV, HTML, and PDF files are written under `logs/reports/`. Event logs are under `logs/`; desk state is under `desks/`; annotated video output is under `outputs/` only when explicitly enabled.

Treat all these files as potentially sensitive. Protect them with operating-system permissions and use a retention policy consistent with local requirements.

## Optional legacy zone mode

Set `mode: desks` to use fixed zones. This mode may discover furniture-based zones or load a manual selection. `employee-monitor select-zones --config ... --camera office` opens the interactive selector. Detect mode does not require the selector, zone files, or a calibration phase.
