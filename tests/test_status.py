import unittest

from employee_monitor.detection import Detection
from employee_monitor.status import (
    ABSENT,
    PHONE_USAGE,
    WORKING,
    PersonStatusTracker,
    StatusTracker,
    assign_person_phone_statuses,
)


class StatusTrackerTests(unittest.TestCase):
    def test_status_changes_after_three_consecutive_observations_at_low_fps(self):
        tracker = StatusTracker(["Desk 1"], stable_frames=3, fps=1.2)

        for _ in range(2):
            self.assertEqual(tracker.update({"Desk 1": WORKING}), [])
        self.assertEqual(tracker.statuses["Desk 1"], ABSENT)

        changes = tracker.update({"Desk 1": WORKING})
        self.assertEqual(
            [(change.desk, change.status) for change in changes], [("Desk 1", WORKING)]
        )

    def test_reverting_candidate_resets_smoothing_counter(self):
        tracker = StatusTracker(["Desk 1"], stable_frames=3, fps=4.0)
        for _ in range(2):
            tracker.update({"Desk 1": PHONE_USAGE})
        tracker.update({"Desk 1": ABSENT})
        for _ in range(2):
            self.assertEqual(tracker.update({"Desk 1": PHONE_USAGE}), [])
        changes = tracker.update({"Desk 1": PHONE_USAGE})
        self.assertEqual([change.status for change in changes], [PHONE_USAGE])

    def test_elapsed_time_does_not_replace_consecutive_frame_count(self):
        tracker = StatusTracker(["Desk 1"], stable_frames=3, fps=1.2)
        self.assertEqual(tracker.update({"Desk 1": WORKING}, elapsed_frames=10), [])
        self.assertEqual(tracker.update({"Desk 1": WORKING}, elapsed_frames=10), [])
        changes = tracker.update({"Desk 1": WORKING}, elapsed_frames=10)
        self.assertEqual([change.status for change in changes], [WORKING])

    def test_skipped_frames_advance_duration_without_counting_as_observations(self):
        tracker = StatusTracker(["Desk 1"], stable_frames=3, fps=2.0)
        tracker.update({"Desk 1": WORKING})
        tracker.advance(elapsed_frames=4)
        self.assertEqual(tracker.update({"Desk 1": WORKING}), [])
        changes = tracker.update({"Desk 1": WORKING})

        self.assertEqual([change.status for change in changes], [WORKING])
        durations = {
            (desk, status): seconds for desk, status, seconds in tracker.summary()
        }
        self.assertEqual(durations[("Desk 1", ABSENT)], 3.5)

    def test_desk_durations_are_independent(self):
        tracker = StatusTracker(["Desk 1", "Desk 2"], stable_frames=2, fps=2.0)
        tracker.update({"Desk 1": WORKING, "Desk 2": PHONE_USAGE})
        changes = tracker.update({"Desk 1": WORKING, "Desk 2": PHONE_USAGE})
        self.assertEqual(
            {(change.desk, change.status) for change in changes},
            {("Desk 1", WORKING), ("Desk 2", PHONE_USAGE)},
        )
        tracker.update({"Desk 1": WORKING, "Desk 2": PHONE_USAGE})
        durations = {
            (desk, status): seconds for desk, status, seconds in tracker.summary()
        }
        self.assertEqual(durations[("Desk 1", WORKING)], 0.5)
        self.assertEqual(durations[("Desk 2", PHONE_USAGE)], 0.5)


class PersonStatusTests(unittest.TestCase):
    def test_phone_is_assigned_to_the_nearest_person_and_status_is_smoothed(self):
        people = [
            Detection((0, 0, 100, 200), 0.9, 0, track_id=1),
            Detection((120, 0, 220, 200), 0.9, 0, track_id=2),
        ]
        phone = Detection((55, 55, 70, 80), 0.8, 67)
        observations = assign_person_phone_statuses(people, [phone], 0.45)

        self.assertEqual(observations, {1: PHONE_USAGE, 2: WORKING})
        tracker = PersonStatusTracker(stable_frames=2)
        self.assertEqual(tracker.update(observations, (1, 2)), [])
        changes = tracker.update(observations, (1, 2))
        self.assertEqual(
            [(change.desk, change.status) for change in changes],
            [("ID 1", PHONE_USAGE), ("ID 2", WORKING)],
        )
        self.assertEqual(tracker.statuses, {1: PHONE_USAGE, 2: WORKING})

    def test_phone_near_a_wrist_is_phone_usage_even_outside_upper_body_box(self):
        keypoints = [(0.0, 0.0, 0.0)] * 9 + [(105.0, 100.0, 0.9), (0.0, 0.0, 0.0)]
        person = Detection(
            (0, 0, 100, 200), 0.9, 0, track_id=1, keypoints=tuple(keypoints)
        )
        phone = Detection((110, 100, 115, 105), 0.8, 67)

        statuses = assign_person_phone_statuses([person], [phone], 0.0)

        self.assertEqual(statuses, {1: PHONE_USAGE})

    def test_person_status_state_expires_with_the_visible_track(self):
        tracker = PersonStatusTracker(stable_frames=1)
        tracker.update({7: PHONE_USAGE}, (7,))

        tracker.update({}, ())

        self.assertEqual(tracker.statuses, {})


if __name__ == "__main__":
    unittest.main()
