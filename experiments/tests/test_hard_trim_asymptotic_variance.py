import unittest

from experiments.scripts.hard_trim_asymptotic_variance import (
    calculate_variance_benchmarks,
)


class HardTrimAsymptoticVarianceTest(unittest.TestCase):
    def test_known_gaussian_variance_constants(self):
        result = calculate_variance_benchmarks()
        self.assertAlmostEqual(
            result["oracle_evaluation_only"]["threshold_asymptotic_variance"],
            3.5595679587,
            places=8,
        )
        self.assertAlmostEqual(
            result["full_sample"]["threshold_asymptotic_variance"],
            41.5199392563,
            places=7,
        )
        self.assertAlmostEqual(
            result["honest_split"]["threshold_asymptotic_variance"],
            183.4781791184,
            places=7,
        )

    def test_honest_split_is_less_efficient(self):
        result = calculate_variance_benchmarks()
        self.assertGreater(
            result["honest_split"]["threshold_asymptotic_variance"],
            4.0 * result["full_sample"]["threshold_asymptotic_variance"],
        )


if __name__ == "__main__":
    unittest.main()
