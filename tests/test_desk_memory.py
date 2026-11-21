import json

from employee_monitor.desk_memory import DeskMemory
from employee_monitor.detection import Detection


def _person(box=(100, 100, 200, 300), track_id=1):
    return Detection(box, 0.9, 0, track_id=track_id)


def _remember(memory, person=None):
    person = person or _person()
    memory.update([person], 0.0, (640, 480))
    memory.update([person], 5.0, (640, 480))
    return person


def test_remembers_still_person_area_and_nearby_furniture(tmp_path):
    memory = DeskMemory(tmp_path / "desks" / "office.json")
    person = _person()
    chair = Detection((110, 280, 190, 360), 0.8, 56)

    memory.update([person, chair], 0.0, (640, 480))
    assert memory.desks == ()
    memory.update([person, chair], 5.0, (640, 480))

    assert memory.desks[0][0] == "Desk 1"
    x1, y1, x2, y2 = memory.desks[0][1]
    assert x1 <= 75 and y1 <= 50
    assert x2 >= 215 and y2 >= 380
    saved = json.loads((tmp_path / "desks" / "office.json").read_text())
    assert saved["desks"][0]["id"] == "Desk 1"
    assert DeskMemory(tmp_path / "desks" / "office.json").desks == memory.desks


def test_absent_alert_waits_for_threshold_and_ends_on_return(tmp_path):
    memory = DeskMemory(tmp_path / "office.json")
    person = _remember(memory)

    assert memory.update([], 7.9, (640, 480)) == ()
    assert memory.absent_desks(7.9) == ()
    start = memory.update([], 8.0, (640, 480))
    assert [(event.status, event.duration_seconds) for event in start] == [
        ("ABSENT START", 3.0)
    ]
    assert memory.absent_desks(8.0)[0].urgent is False

    returned = memory.update([person], 9.0, (640, 480))

    assert [(event.status, event.duration_seconds) for event in returned] == [
        ("ABSENT END", 4.0)
    ]
    assert memory.absent_desks(9.0) == ()


def test_urgent_escalation_is_emitted_once(tmp_path):
    memory = DeskMemory(tmp_path / "office.json")
    _remember(memory)
    memory.update([], 8.0, (640, 480))

    events = memory.update([], 35.1, (640, 480))

    assert [event.status for event in events] == ["URGENT"]
    assert memory.absent_desks(35.1)[0].urgent is True
    assert memory.update([], 36.0, (640, 480)) == ()


def test_absence_state_survives_restart_without_duplicate_start_event(tmp_path):
    path = tmp_path / "office.json"
    memory = DeskMemory(path)
    person = _remember(memory)
    memory.update([], 8.0, (640, 480))

    restarted = DeskMemory(path)

    assert restarted.update([], 9.0, (640, 480)) == ()
    ended = restarted.update([person], 10.0, (640, 480))
    assert [(event.status, event.duration_seconds) for event in ended] == [
        ("ABSENT END", 5.0)
    ]


def test_forget_expiry_removes_only_previously_remembered_desks(tmp_path):
    memory = DeskMemory(
        tmp_path / "office.json",
        forget_after_minutes=0.1,
    )
    assert memory.absent_desks(100.0) == ()
    _remember(memory)

    events = memory.update([], 11.1, (640, 480))

    assert [event.status for event in events] == ["ABSENT START", "ABSENT END"]
    assert memory.desks == ()
    assert memory.absent_desks(11.1) == ()


def test_overlapping_remembered_areas_keep_the_existing_desk_id(tmp_path):
    memory = DeskMemory(tmp_path / "office.json", min_still_seconds=0.0)
    first = _person()
    second = _person((110, 100, 210, 300), track_id=2)

    memory.update([first], 0.0, (640, 480))
    memory.update([first, second], 1.0, (640, 480))

    assert [name for name, _ in memory.desks] == ["Desk 1"]
