import unittest

from experiments.scripts.hard_trim_spline_density_variance import (
    calculate_limiting_variance,
    population_sieve_target,
)


class HardTrimSplineDensityVarianceTest(unittest.TestCase):
    def test_known_limiting_variance_constants(self):
        result = calculate_limiting_variance()
        self.assertAlmostEqual(
            result["spline_density_and_boundary"]["density_score_variance"],
            0.0201594543,
            places=8,
        )
        self.assertAlmostEqual(
            result["full_sample"]["threshold_asymptotic_variance"],
            43.32611164,
            places=7,
        )
        self.assertAlmostEqual(
            result["honest_split"]["threshold_asymptotic_variance"],
            205.83554404,
            places=7,
        )

    def test_population_projection_target_approaches_truth(self):
        coarse = population_sieve_target(8)
        fine = population_sieve_target(20)
        self.assertLess(
            abs(fine["projection_bias"]), abs(coarse["projection_bias"])
        )
        self.assertLess(abs(fine["projection_bias"]), 1e-4)


if __name__ == "__main__":
    unittest.main()
