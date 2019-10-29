import unittest

import numpy as np
import pandas as pd
from pytest import approx

import evaluation


class TestEvaluation(unittest.TestCase):
    def test_calculates_nwrmsle_for_perfect_match(self):
        estimate = np.array([1, 2, 3])
        actual = np.array([1, 2, 3])
        weights = np.array([1, 1, 1])
        calculated_nwrmsle = evaluation.nwrmsle(estimate, actual, weights)

        assert calculated_nwrmsle == 0.0

    def test_calculates_nwrmsle_for_perfect_match_for_series(self):
        estimate = pd.Series(data=[1, 2, 3])
        actual = np.array([1, 2, 3])
        weights = np.array([1, 1, 1])
        calculated_nwrmsle = evaluation.nwrmsle(estimate, actual, weights)

        assert calculated_nwrmsle == 0.0

    def test_calculates_nwrmsle_for_imperfect_match_nan(self):
        estimate = np.array([-1, 2, 3]).tolist()
        actual = np.array([-1, 2, 3])
        weights = np.array([1, 1, 1])
        calculated_nwrmsle = evaluation.nwrmsle(estimate, actual, weights)

        assert calculated_nwrmsle != calculated_nwrmsle

    def test_calculates_nwrmsle_for_imperfect_match(self):
        estimate = np.array([0, 0, 0])
        actual = np.array([1, 1, 1])
        weights = np.array([1, 1, 1])
        calculated_nwrmsle = evaluation.nwrmsle(estimate, actual, weights)

        # Assert by-hand calculation of nwrmsle is reasonably close to python calculation
        assert approx(calculated_nwrmsle, 0.69314718)


if __name__ == "__main__":
    unittest.main()
