import pytest

from employee_monitor.config import ConfigurationError, load_config


def test_loads_multiple_cameras_and_legacy_zones(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
cameras:
  - name: front
    source: 0
  - name: back
    source: rtsp://camera.local/live
auto_discovery:
  calibration_seconds: 12
zones:
  - name: Desk A
    x1: 1
    y1: 2
    x2: 20
    y2: 30
zone_frame_size: [100, 100]
""",
        encoding="utf-8",
    )

    config = load_config(config_path)

    assert [camera.name for camera in config.cameras] == ["front", "back"]
    assert config.cameras[0].source == 0
    assert config.cameras[0].calibration_seconds == 12
    assert config.cameras[1].manual_zones[0].name == "Desk A"
    assert config.cameras[1].zones_file == tmp_path / "zones" / "back.json"


@pytest.mark.parametrize(
    "content, message",
    [
        ("cameras: []\n", "non-empty list"),
        (
            "cameras:\n  - name: same\n    source: 0\n  - name: same\n    source: 1\n",
            "unique",
        ),
        ("cameras:\n  - name: ../bad\n    source: 0\n", "path separators"),
        ("cameras:\n  - name: office\n    source: -1\n", "negative"),
        (
            "cameras:\n  - name: office\n    source: 0\nmodel:\n  phone_conf: 1.5\n",
            "between 0 and 1",
        ),
    ],
)
def test_invalid_config_has_clear_validation_error(tmp_path, content, message):
    config_path = tmp_path / "invalid.yaml"
    config_path.write_text(content, encoding="utf-8")

    with pytest.raises(ConfigurationError, match=message):
        load_config(config_path)


def test_supports_legacy_single_camera_input(tmp_path):
    config_path = tmp_path / "legacy.yaml"
    config_path.write_text("input:\n  source: office.mp4\n", encoding="utf-8")

    config = load_config(config_path)

    assert config.cameras[0].source == "office.mp4"


def test_default_config_uses_detection_mode_without_zone_setup(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "cameras:\n  - name: office\n    source: 0\n", encoding="utf-8"
    )

    config = load_config(config_path)

    assert config.mode == "detect"
    assert config.cameras[0].manual_zones == ()
    assert config.model.imgsz == 960
    assert config.model.phone_confidence == 0.15
    assert config.clahe_enabled is False
    assert config.desks_memory.enabled is True
    assert config.desks_memory.min_still_seconds == 5.0
    assert config.desks_memory.absent_after_seconds == 3.0
    assert config.desks_memory.urgent_after_seconds == 30.0
    assert config.cameras[0].desks_file == tmp_path / "desks" / "office.json"


def test_desk_mode_and_clahe_can_be_enabled(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "mode: desks\npreprocessing:\n  clahe: true\n"
        "cameras:\n  - name: office\n    source: 0\n",
        encoding="utf-8",
    )

    config = load_config(config_path)

    assert config.mode == "desks"
    assert config.clahe_enabled is True


def test_desk_memory_rejects_urgent_timeout_before_absence_timeout(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "desks_memory:\n  absent_after_seconds: 10\n  urgent_after_seconds: 5\n"
        "cameras:\n  - name: office\n    source: 0\n",
        encoding="utf-8",
    )

    with pytest.raises(ConfigurationError, match="at least absent_after_seconds"):
        load_config(config_path)


def test_loads_camera_source_from_local_dotenv_file(tmp_path, monkeypatch):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "cameras:\n  - name: office\n    source: ${CAMERA_RTSP_URL}\n",
        encoding="utf-8",
    )
    (tmp_path / ".env").write_text(
        "CAMERA_RTSP_URL=rtsp://camera.local/live\n", encoding="utf-8"
    )
    monkeypatch.delenv("CAMERA_RTSP_URL", raising=False)

    config = load_config(config_path)

    assert config.cameras[0].source == "rtsp://camera.local/live"


def test_missing_camera_source_environment_variable_is_clear(tmp_path, monkeypatch):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "cameras:\n  - name: office\n    source: ${CAMERA_RTSP_URL}\n",
        encoding="utf-8",
    )
    monkeypatch.delenv("CAMERA_RTSP_URL", raising=False)

    with pytest.raises(ConfigurationError, match="CAMERA_RTSP_URL"):
        load_config(config_path)
