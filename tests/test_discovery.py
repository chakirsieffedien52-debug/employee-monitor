from employee_monitor.detection import Detection
from employee_monitor.zones import (
    AutoZoneCalibrator,
    DeskZone,
    load_zone_document,
    merge_overlapping_zones,
    save_zone_document,
)


def test_auto_calibration_clusters_supported_desk_objects():
    calibrator = AutoZoneCalibrator(cluster_distance_ratio=0.1)
    detections = [
        Detection((90, 90, 120, 110), 0.9, 63),
        Detection((125, 95, 145, 110), 0.9, 66),
        Detection((420, 290, 450, 310), 0.9, 56),
    ]

    zones = calibrator.observe(detections, (600, 400), 1.0)
    zones = calibrator.discover()

    assert len(zones) == 2
    assert all(zone.origin == "auto" for zone in zones)
    assert zones[0].x1 < zones[0].x2


def test_stationary_person_creates_a_zone_after_configured_duration():
    calibrator = AutoZoneCalibrator(still_seconds=30)
    person = Detection((100, 100, 160, 240), 0.9, 0, track_id=7)

    calibrator.observe([person], (400, 300), 10.0)
    assert calibrator.stationary_zones() == []
    calibrator.observe([person], (400, 300), 39.9)
    assert calibrator.stationary_zones() == []
    calibrator.observe([person], (400, 300), 40.0)

    zones = calibrator.stationary_zones()
    assert len(zones) == 1
    assert zones[0].x1 <= 100 and zones[0].x2 >= 160


def test_overlapping_zones_merge_above_iou_threshold():
    zones = [
        DeskZone("A", 0, 0, 100, 100),
        DeskZone("B", 10, 10, 110, 110),
        DeskZone("C", 250, 250, 300, 300),
    ]

    merged = merge_overlapping_zones(zones, iou_threshold=0.4)

    assert len(merged) == 2
    assert (merged[0].x1, merged[0].y1, merged[0].x2, merged[0].y2) == (
        0,
        0,
        110,
        110,
    )


def test_zone_document_round_trip_preserves_origin_and_frame_size(tmp_path):
    path = tmp_path / "zones" / "office.json"
    zones = [DeskZone("Desk 1", 10, 20, 100, 120, "manual")]

    save_zone_document(path, "office", zones, (640, 480), "manual")
    loaded, frame_size, origin = load_zone_document(path)

    assert loaded == zones
    assert frame_size == (640, 480)
    assert origin == "manual"
