import unittest

from employee_monitor.detection import Detection
from employee_monitor.zones import (
    DeskZone,
    assign_people_to_zones,
    phone_is_near_person,
    scale_zones_to_frame,
)


class ZoneTests(unittest.TestCase):
    def test_person_is_assigned_to_only_one_best_matching_zone(self):
        first = DeskZone("Desk 1", 0, 0, 100, 100)
        second = DeskZone("Desk 2", 50, 0, 150, 100)
        person = Detection((70.0, 10.0, 120.0, 90.0), 0.9, 0, 1)

        assignments = assign_people_to_zones([person], [first, second])

        self.assertEqual(assignments["Desk 1"], tuple())
        self.assertEqual(assignments["Desk 2"], (person,))

    def test_zone_match_accepts_center_or_over_30_percent_person_overlap(self):
        center_person = Detection((0.0, 0.0, 100.0, 100.0), 0.9, 0, 1)
        overlap_person = Detection((0.0, 0.0, 100.0, 100.0), 0.9, 0, 2)
        center_zone = DeskZone("Center", 45, 45, 55, 55)
        overlap_zone = DeskZone("Overlap", 0, 0, 35, 100)
        small_overlap_zone = DeskZone("Small", 80, 0, 100, 100)

        center_assignment = assign_people_to_zones([center_person], [center_zone])
        overlap_assignment = assign_people_to_zones([overlap_person], [overlap_zone])
        rejected_assignment = assign_people_to_zones(
            [overlap_person], [small_overlap_zone]
        )

        self.assertEqual(center_assignment["Center"], (center_person,))
        self.assertEqual(overlap_assignment["Overlap"], (overlap_person,))
        self.assertEqual(rejected_assignment["Small"], tuple())

    def test_zones_scale_from_selector_size_to_video_size(self):
        zone = DeskZone("Desk 1", 80, 40, 160, 100)

        scaled = scale_zones_to_frame([zone], (832, 464), (416, 232))

        self.assertEqual(scaled, [DeskZone("Desk 1", 40, 20, 80, 50)])

    def test_phone_proximity_is_limited_to_person_hand_area(self):
        person = (100.0, 100.0, 200.0, 300.0)

        self.assertTrue(
            phone_is_near_person((215.0, 305.0, 225.0, 315.0), person, 0.25)
        )
        self.assertFalse(
            phone_is_near_person((230.0, 305.0, 240.0, 315.0), person, 0.25)
        )

    def test_phone_at_desk_edge_is_associated_with_nearby_person(self):
        person = (566.0, 245.0, 701.0, 378.0)
        phone = (663.0, 387.0, 728.0, 419.0)

        self.assertFalse(phone_is_near_person(phone, person, 0.25))
        self.assertTrue(phone_is_near_person(phone, person, 0.45))


if __name__ == "__main__":
    unittest.main()
