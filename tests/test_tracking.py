from employee_monitor.detection import Detection
from employee_monitor.tracking import PersonTracker


def test_person_tracker_persists_a_box_for_configured_missed_frames():
    tracker = PersonTracker(persistence_frames=2)
    first = tracker.update([Detection((0, 0, 100, 200), 0.9, 0)])[0]

    stale = tracker.update([])[0]
    assert stale.track_id == first.track_id
    assert stale.observed is False

    refreshed = tracker.update([Detection((2, 0, 102, 200), 0.8, 0)])[0]
    assert refreshed.track_id == first.track_id
    assert refreshed.observed is True


def test_tracker_releases_people_after_persistence_window():
    tracker = PersonTracker(persistence_frames=1)
    tracker.update([Detection((0, 0, 100, 200), 0.9, 0)])

    assert len(tracker.update([])) == 1
    assert tracker.update([]) == ()


def test_tracker_expires_a_track_after_its_persistence_window():
    tracker = PersonTracker(persistence_frames=2)
    tracker.update([Detection((0, 0, 100, 200), 0.9, 0)])

    assert len(tracker.update([])) == 1
    assert len(tracker.update([])) == 1
    assert tracker.update([]) == ()
