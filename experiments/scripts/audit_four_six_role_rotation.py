"""Check four/shared-gamma versus six/separate-gamma role rotation algebra.

This checks the proposed influence expansion, not fitted splines or the full
hard-trimmed estimator. O=(Z1,Z2,Z3) has independent standard normal entries.
The three oracle contributions are Z1, Z1+Z2, Z2+Z3. The OLS-like
two-dimensional influence vector is (Z1+Z3, Z2-Z3), with loadings
(1,0), (0,1), (-1/2,1/2). These deliberately correlated contributions
test all cross-role covariances, including those with the first-stage term.

Run: python3 experiments/scripts/audit_four_six_role_rotation.py

For K equal folds of size N/K and all K cyclic role assignments, the
average contribution for any given role j is exactly P_N psi_j, because
(1/K)*(K/N)=1/N. The same identity holds when three gamma contributions
are combined into one role before averaging. Finitely many remainders
o_p((N/K)^(-1/2)) average to o_p(N^(-1/2)).
"""

import json

import numpy as np


def main():
    n, reps, seed = 1200, 20000, 20261002
    six = np.array([
        [1., 0., 0.], [1., 1., 0.], [0., 1., 1.],
        [1., 0., 1.], [0., 1., -1.], [-0.5, 0.5, -1.],
    ])
    four = np.vstack((six[:3], six[3:].sum(axis=0)))
    # Twelve microfold means allow exactly the same observations to be grouped
    # into either four or six equal folds. Gaussian means are simulated exactly.
    rng = np.random.default_rng(seed)
    micro = rng.normal(size=(reps, 12, 3)) * np.sqrt(12/n)
    common = micro.mean(axis=1) @ six.sum(axis=0)
    results = {}
    for k, roles in ((4, four), (6, six)):
        means = micro.reshape(reps, k, 12//k, 3).mean(axis=2)
        rotations = np.stack([
            np.einsum('rkd,kd->r', np.roll(means, r, axis=1), roles)
            for r in range(k)
        ], axis=1)
        averaged = rotations.mean(axis=1)
        error = float(np.max(np.abs(averaged-common)))
        assert error < 1e-12, (k, error)
        results[str(k)] = {
            'max_rotation_vs_pooled_error': error,
            'analytic_fixed_assignment_variance': float(k*np.sum(roles**2)),
            'analytic_rotated_variance': float(np.sum(roles.sum(axis=0)**2)),
            'monte_carlo_fixed_assignment_variance': float(n*np.var(rotations[:,0], ddof=1)),
            'monte_carlo_rotated_variance': float(n*np.var(averaged, ddof=1)),
        }
    print(json.dumps({
        'settings': {'N': n, 'replicates': reps, 'seed': seed},
        'normalization': 'sqrt(total N)',
        'common_influence_coefficients_on_Z': six.sum(axis=0).tolist(),
        'incorrect_sum_of_four_marginal_variances': float(np.sum(four**2)),
        'results': results,
    }, indent=2))


if __name__ == '__main__':
    main()
