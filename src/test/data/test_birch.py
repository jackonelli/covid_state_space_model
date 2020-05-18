"""Testing angle measure"""
import unittest
from data.common import Serie
from data.birch import into_birch_dict


class TestBirch(unittest.TestCase):
    def test_single_observed_state_component(self):
        serie = Serie("i", [0, 1, 2], [1, 8, 3])
        start_state = {"i": 1, "d": 0}
        prior_params = {"p": 0.5}
        birch_dict = into_birch_dict([serie], start_state, prior_params)
        true_birch_dict = {
            "theta": {
                "p": 0.5
            },
            "x": [{
                "i": 1,
                "d": 0
            }, {
                "i": 1
            }, {
                "i": 8
            }, {
                "i": 3
            }]
        }
        self.assertEqual(birch_dict, true_birch_dict)

    def test_two_observed_state_components(self):
        series = [
            Serie("i", [0, 1, 2], [1, 8, 3]),
            Serie("d", [0, 1, 2], [2, 3, 4])
        ]
        start_state = {"i": 1, "d": 0, "r": 1}
        prior_params = {"p": 0.5}
        birch_dict = into_birch_dict(series, start_state, prior_params)
        true_birch_dict = {
            "theta": {
                "p": 0.5
            },
            "x": [{
                "i": 1,
                "d": 0,
                "r": 1,
            }, {
                "i": 1,
                "d": 2,
            }, {
                "i": 8,
                "d": 3,
            }, {
                "i": 3,
                "d": 4,
            }]
        }
        self.assertEqual(birch_dict, true_birch_dict)
