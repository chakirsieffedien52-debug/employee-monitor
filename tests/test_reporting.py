import csv
from datetime import date, datetime

from employee_monitor.reporting import EventLogger, ReportExporter, read_daily_events


def test_event_logger_writes_camera_scoped_events_and_summary(tmp_path):
    logger = EventLogger(tmp_path / "office", "office")
    logger.log_change("Desk 1", "WORKING")
    summary_path = logger.save_summary([("Desk 1", "WORKING", 4.5)])
    logger.close()

    events = read_daily_events(tmp_path / "office")
    assert events[0]["camera"] == "office"
    assert events[0]["desk"] == "Desk 1"
    assert events[0]["duration_seconds"] == ""
    with summary_path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    assert rows == [{"desk": "Desk 1", "status": "WORKING", "seconds": "4.500"}]


def test_event_logger_archives_legacy_format_and_persists_absence_duration(tmp_path):
    directory = tmp_path / "office"
    directory.mkdir()
    events_path = directory / "events.csv"
    with events_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(("timestamp", "camera", "desk", "status"))
        writer.writerow(
            (
                datetime.now().astimezone().isoformat(timespec="seconds"),
                "office",
                "Desk 1",
                "WORKING",
            )
        )
    legacy_contents = events_path.read_text(encoding="utf-8")

    logger = EventLogger(directory, "office")
    logger.log_change("Desk 1", "ABSENT END", duration_seconds=42.5)
    logger.close()

    backups = list(directory.glob("events_*.bak.csv"))
    assert len(backups) == 1
    assert backups[0].read_text(encoding="utf-8") == legacy_contents
    events = read_daily_events(directory)
    assert len(events) == 1
    assert events[0]["status"] == "ABSENT END"
    assert events[0]["duration_seconds"] == "42.500"


def test_event_logger_archives_unknown_format_instead_of_failing(tmp_path):
    directory = tmp_path / "office"
    directory.mkdir()
    events_path = directory / "events.csv"
    events_path.write_text("unrecognized,columns\nnot a status log\n", encoding="utf-8")

    logger = EventLogger(directory, "office")
    logger.log_change("Desk 1", "WORKING")
    logger.close()

    backups = list(directory.glob("events_*.bak.csv"))
    assert len(backups) == 1
    assert backups[0].read_text(encoding="utf-8") == (
        "unrecognized,columns\nnot a status log\n"
    )
    assert events_path.read_text(encoding="utf-8").startswith(
        "timestamp,camera,desk,status,duration_seconds\n"
    )


def test_daily_report_exports_events_and_time_summaries(tmp_path):
    exporter = ReportExporter(tmp_path / "reports")
    paths = exporter.export_daily(
        {
            "office": [
                {
                    "timestamp": "2026-01-02T10:00:00+00:00",
                    "desk": "Desk 1",
                    "status": "WORKING",
                }
            ]
        },
        {"office": {"Desk 1": {"WORKING": 12.5}}},
        report_date=date(2026, 1, 2),
    )

    assert set(paths) == {"csv", "html", "pdf"}
    assert all(path.is_file() and path.stat().st_size > 0 for path in paths.values())
    report_csv = paths["csv"].read_text(encoding="utf-8")
    assert "status_event" in report_csv
    assert "status_duration" in report_csv
    assert "12.5" in report_csv
    assert "Session time by status" in paths["html"].read_text(encoding="utf-8")
