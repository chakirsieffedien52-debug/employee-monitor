from fastapi.testclient import TestClient

from employee_monitor.api import LiveState, create_app
from employee_monitor.reporting import ReportExporter


def test_dashboard_status_api_and_daily_report_exports(tmp_path):
    state = LiveState()
    state.update(
        "office",
        {"Desk 1": "WORKING"},
        {"Desk 1": {"WORKING": 2.0}},
        1.5,
        True,
    )
    app = create_app(
        state,
        ReportExporter(tmp_path / "reports"),
        {
            "office": [
                {
                    "timestamp": "2026-01-02T10:00:00+00:00",
                    "camera": "office",
                    "desk": "Desk 1",
                    "status": "ABSENT END",
                    "duration_seconds": "42.000",
                }
            ]
        },
    )
    client = TestClient(app)

    dashboard = client.get("/")
    assert dashboard.status_code == 200
    assert "Duration (s)" in dashboard.text
    status = client.get("/api/status").json()
    assert status["cameras"]["office"]["statuses"]["Desk 1"] == "WORKING"
    report = client.get("/api/report/daily").json()
    assert report["cameras"]["office"]["durations"]["Desk 1"]["WORKING"] == 2.0
    assert report["events"][0]["duration_seconds"] == "42.000"
    assert client.get("/api/report/export?format=csv").status_code == 200
