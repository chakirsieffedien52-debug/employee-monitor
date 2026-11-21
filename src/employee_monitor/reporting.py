"""Per-camera event logs and daily report exports."""

import csv
import html
import logging
import threading
from collections.abc import Iterable, Mapping, Sequence
from datetime import date, datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

LOGGER = logging.getLogger(__name__)
EVENT_FIELDS = ("timestamp", "camera", "desk", "status", "duration_seconds")


class EventLogger:
    def __init__(self, directory: Path, camera: str = "camera") -> None:
        directory.mkdir(parents=True, exist_ok=True)
        self.camera = camera
        self.events_path = directory / "events.csv"
        needs_header = (
            not self.events_path.exists() or self.events_path.stat().st_size == 0
        )
        if not needs_header:
            _upgrade_event_file(self.events_path)
            needs_header = (
                not self.events_path.exists() or self.events_path.stat().st_size == 0
            )
        self._lock = threading.Lock()
        self._event_file = self.events_path.open("a", newline="", encoding="utf-8")
        self._event_writer = csv.writer(self._event_file)
        if needs_header:
            self._event_writer.writerow(EVENT_FIELDS)
            self._event_file.flush()

    def log_change(
        self,
        desk: str,
        status: str,
        camera: Optional[str] = None,
        duration_seconds: Optional[float] = None,
    ) -> None:
        timestamp = datetime.now().astimezone().isoformat(timespec="seconds")
        with self._lock:
            self._event_writer.writerow(
                (
                    timestamp,
                    camera or self.camera,
                    desk,
                    status,
                    "" if duration_seconds is None else f"{duration_seconds:.3f}",
                )
            )
            self._event_file.flush()

    def save_summary(self, rows: Iterable[Tuple[str, str, float]]) -> Path:
        summary_path = self.events_path.parent / "summary.csv"
        with summary_path.open("w", newline="", encoding="utf-8") as summary_file:
            writer = csv.writer(summary_file)
            writer.writerow(("desk", "status", "seconds"))
            for desk, status, seconds in rows:
                writer.writerow((desk, status, f"{seconds:.3f}"))
        LOGGER.info("Saved status summary for %s to %s", self.camera, summary_path)
        return summary_path

    def close(self) -> None:
        with self._lock:
            if not self._event_file.closed:
                self._event_file.close()


def _upgrade_event_file(path: Path) -> None:
    try:
        with path.open("r", newline="", encoding="utf-8") as stream:
            header = next(csv.reader(stream), None)
    except (csv.Error, UnicodeError):
        header = None
    if header and tuple(header) == EVENT_FIELDS:
        return
    timestamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S_%f")
    backup = path.with_name(f"events_{timestamp}.bak.csv")
    suffix = 1
    while backup.exists():
        backup = path.with_name(f"events_{timestamp}_{suffix}.bak.csv")
        suffix += 1
    path.replace(backup)
    LOGGER.warning(
        "Archived event log with an unsupported format to %s; starting a new log",
        backup,
    )


class ReportExporter:
    def __init__(self, output_dir: Path) -> None:
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def export_daily(
        self,
        events_by_camera: Mapping[str, Sequence[Mapping[str, str]]],
        durations_by_camera: Optional[
            Mapping[str, Mapping[str, Mapping[str, float]]]
        ] = None,
        report_date: Optional[date] = None,
    ) -> Dict[str, Path]:
        day = report_date or date.today()
        rows = []
        for camera, events in events_by_camera.items():
            rows.extend(
                {
                    "timestamp": event.get("timestamp", ""),
                    "camera": camera,
                    "desk": event.get("desk", ""),
                    "status": event.get("status", ""),
                    "duration_seconds": event.get("duration_seconds", ""),
                }
                for event in events
            )
        summaries = [
            {
                "camera": camera,
                "desk": desk,
                "status": status,
                "seconds": seconds,
            }
            for camera, desks in (durations_by_camera or {}).items()
            for desk, statuses in desks.items()
            for status, seconds in statuses.items()
        ]
        csv_path = self.output_dir / f"daily-{day.isoformat()}.csv"
        with csv_path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(
                stream,
                fieldnames=(
                    "record_type",
                    "timestamp",
                    "camera",
                    "desk",
                    "status",
                    "duration_seconds",
                    "seconds",
                ),
            )
            writer.writeheader()
            writer.writerows({"record_type": "status_event", **row} for row in rows)
            writer.writerows(
                {
                    "record_type": "status_duration",
                    "timestamp": "",
                    **summary,
                }
                for summary in summaries
            )

        html_path = self.output_dir / f"daily-{day.isoformat()}.html"
        html_path.write_text(_render_html(day, rows, summaries), encoding="utf-8")
        pdf_path = self.output_dir / f"daily-{day.isoformat()}.pdf"
        _render_pdf(pdf_path, day, rows, summaries)
        return {"csv": csv_path, "html": html_path, "pdf": pdf_path}


def read_daily_events(
    directory: Path, report_date: Optional[date] = None
) -> List[dict]:
    path = directory / "events.csv"
    if not path.is_file():
        return []
    day = report_date or date.today()
    events = []
    with path.open("r", newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            if not row.get("timestamp", "").startswith(day.isoformat()):
                continue
            events.append(
                {
                    "timestamp": row.get("timestamp", ""),
                    "camera": row.get("camera", directory.name),
                    "desk": row.get("desk", ""),
                    "status": row.get("status", ""),
                    "duration_seconds": row.get("duration_seconds", ""),
                }
            )
    return events


def _render_html(
    day: date,
    rows: Sequence[Mapping[str, str]],
    summaries: Sequence[Mapping[str, object]],
) -> str:
    body = "\n".join(
        "<tr><td>{}</td><td>{}</td><td>{}</td><td>{}</td><td>{}</td></tr>".format(
            *(
                html.escape(row[key])
                for key in (
                    "timestamp",
                    "camera",
                    "desk",
                    "status",
                    "duration_seconds",
                )
            )
        )
        for row in rows
    )
    duration_body = "\n".join(
        "<tr><td>{}</td><td>{}</td><td>{}</td><td>{:.1f}s</td></tr>".format(
            html.escape(str(row["camera"])),
            html.escape(str(row["desk"])),
            html.escape(str(row["status"])),
            float(row["seconds"]),
        )
        for row in summaries
    )
    return f"""<!doctype html>
<html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>Daily monitoring report</title>
<style>
body{{font:16px/1.5 system-ui,sans-serif;max-width:1000px;margin:3rem auto;padding:0 1rem;color:#17212b}}
h1{{font-size:1.8rem}}table{{border-collapse:collapse;width:100%}}
th,td{{border-bottom:1px solid #d8e0e8;padding:.65rem;text-align:left}}
th{{background:#f2f6fa}}
</style>
<h1>Daily status events — {day.isoformat()}</h1>
<h2>Session time by status</h2>
<table><thead><tr><th>Camera</th><th>Desk</th><th>Status</th><th>Time</th></tr></thead>
<tbody>{duration_body}</tbody></table>
<h2>Status changes</h2>
<table><thead><tr><th>Timestamp</th><th>Camera</th><th>Desk</th><th>Status</th><th>Duration (s)</th></tr></thead>
<tbody>{body}</tbody></table>
</html>"""


def _render_pdf(
    path: Path,
    day: date,
    rows: Sequence[Mapping[str, str]],
    summaries: Sequence[Mapping[str, object]],
) -> None:
    try:
        from reportlab.lib import colors
        from reportlab.lib.pagesizes import A4, landscape
        from reportlab.lib.styles import getSampleStyleSheet
        from reportlab.platypus import (
            Paragraph,
            SimpleDocTemplate,
            Spacer,
            Table,
            TableStyle,
        )
    except ImportError as exc:
        raise RuntimeError(
            "PDF export requires reportlab. Install the project dependencies."
        ) from exc

    document = SimpleDocTemplate(str(path), pagesize=landscape(A4))
    summary_data = [["Camera", "Desk", "Status", "Seconds"]]
    summary_data.extend(
        [
            row.get("camera", ""),
            row.get("desk", ""),
            row.get("status", ""),
            f"{float(row.get('seconds', 0.0)):.1f}",
        ]
        for row in summaries
    )
    event_data = [["Timestamp", "Camera", "Desk", "Status", "Duration (s)"]]
    event_data.extend(
        [
            row.get("timestamp", ""),
            row.get("camera", ""),
            row.get("desk", ""),
            row.get("status", ""),
            row.get("duration_seconds", ""),
        ]
        for row in rows
    )
    table_style = TableStyle(
        [
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#18324b")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#ccd5df")),
            (
                "ROWBACKGROUNDS",
                (0, 1),
                (-1, -1),
                [colors.white, colors.HexColor("#f2f6fa")],
            ),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ]
    )
    summary_table = Table(summary_data, repeatRows=1)
    summary_table.setStyle(table_style)
    event_table = Table(event_data, repeatRows=1)
    event_table.setStyle(table_style)
    document.build(
        [
            Paragraph(
                f"Daily monitoring report — {day.isoformat()}",
                getSampleStyleSheet()["Title"],
            ),
            Spacer(1, 16),
            Paragraph("Session time by status", getSampleStyleSheet()["Heading2"]),
            summary_table,
            Spacer(1, 16),
            Paragraph("Status changes", getSampleStyleSheet()["Heading2"]),
            event_table,
        ]
    )
