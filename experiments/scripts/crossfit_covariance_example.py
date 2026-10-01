"""Analytic covariance example, with reproducible Monte Carlo checks.

Run: python3 experiments/scripts/crossfit_covariance_example.py

Z, eta, epsilon are independent N(0,1); X=1+Z, T=X-1, Q=T+eta,
D=1{Q>0}, W=-1+X, Y=D*W+epsilon. Treat gamma=(-1,1) as known.
The utility is U(phi)=E[W 1{Q>phi}]=f_Q(phi), maximized at phi=0.

This is a restricted parametric illustration, not a validation of the general
spline/EIV/boundary theorem. Fit Y on (1,X,D,DX), equivalently separate
intercept/slope regressions in treatment and control. Alpha is constant.
The outcome comparison holds distribution weights fixed at their true values.

Writing c=1/sqrt(pi), v=1-1/pi, E[X|D=d]=1+(2d-1)c and
Var(X|D=d)=v. Independent homoskedastic noise gives covariance of the
sqrt(N) treatment-minus-control coefficient errors
    Sigma = (4/v) [[2,-1],[-1,1]].
At phi=0 the weights on both coefficients equal f_Q(0)=1/(2sqrt(pi)).
Hence the weighted covariance matrix is [[2,-1],[-1,1]]/(pi-1).

For the distribution block, let k(z) be the standard normal density and
f=f_Q(0). The g and p_X influence contributions to -U'(0) are
    psi_g=-(k(Z)-f), psi_p=(1+Z)*k(Z)-f.
Their sum is Z*k(Z). Put a=1/(2*pi*sqrt(3))-1/(4*pi),
b=1/(6*pi*sqrt(3)); their covariance matrix is [[a,-a],[-a,a+b]].
The distribution Monte Carlo uses these exact influence functions, not a
fitted spline. The outcome Monte Carlo uses actual OLS estimators.

For a two-fold role rotation, each observation enters both component scores
once. Averaging therefore restores their covariance. This conclusion requires
the same influence functions in both methods; it is not a universal efficiency
claim for different fitting procedures or an independent-rotations assumption.
"""

import argparse
import json
import math

import numpy as np


def contrast(x, d, y):
    """OLS treatment minus control intercept and slope."""
    fits = []
    for treatment in (0, 1):
        xx, yy = x[d == treatment], y[d == treatment]
        xc = xx - xx.mean()
        slope = xc @ (yy - yy.mean()) / (xc @ xc)
        fits.append(np.array([yy.mean() - slope * xx.mean(), slope]))
    return fits[1] - fits[0]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reps', type=int, default=5000)
    parser.add_argument('--sample-size', type=int, default=2000)
    parser.add_argument('--seed', type=int, default=20261001)
    args = parser.parse_args()
    if args.sample_size < 100 or args.sample_size % 2 or args.reps < 2:
        parser.error('Use an even sample size >=100 and at least two replicates.')
    rng = np.random.default_rng(args.seed)
    n = args.sample_size
    f = 1 / (2 * math.sqrt(math.pi))
    outcome_cov = np.array([[2., -1.], [-1., 1.]]) / (math.pi - 1)
    a = 1 / (2 * math.pi * math.sqrt(3)) - 1 / (4 * math.pi)
    b = 1 / (6 * math.pi * math.sqrt(3))
    distribution_cov = np.array([[a, -a], [-a, a + b]])
    outcome = np.empty((args.reps, 3))
    distribution = np.empty_like(outcome)
    components = np.empty((args.reps, 2))
    for r in range(args.reps):
        z, eta, eps = rng.normal(size=(3, n))
        x = 1 + z
        d = (z + eta > 0).astype(int)
        y = d * z + eps
        pooled = contrast(x, d, y)
        ca = contrast(x[:n//2], d[:n//2], y[:n//2])
        cb = contrast(x[n//2:], d[n//2:], y[n//2:])
        first = f * (ca[0] + cb[1])
        second = f * (cb[0] + ca[1])
        outcome[r] = (f * pooled.sum(), first, (first + second) / 2)
        components[r] = f * (pooled - np.array([-1., 1.]))
        k = np.exp(-z*z/2) / math.sqrt(2*math.pi)
        sg, sp = -(k-f), x*k-f
        first = sg[:n//2].mean() + sp[n//2:].mean()
        second = sg[n//2:].mean() + sp[:n//2].mean()
        distribution[r] = ((sg+sp).mean(), first, (first+second)/2)
    result = {'settings': vars(args), 'normalization': 'sqrt(total sample size N)'}
    for name, covariance, estimates in (
        ('outcome_actual_ols', outcome_cov, outcome),
        ('distribution_influence_average', distribution_cov, distribution),
    ):
        joint = covariance.sum()
        exact = (joint, 2*np.trace(covariance), joint)
        empirical = n * np.var(estimates, axis=0, ddof=1)
        result[name] = {
            'analytic_component_covariance': covariance.tolist(),
            'variance_comparison': {
                method: {'analytic': float(target), 'monte_carlo': float(value)}
                for method, target, value in zip(
                    ('together', 'fully_split_no_rotation', 'fully_split_rotated'),
                    exact, empirical,
                )
            },
        }
    result['outcome_actual_ols']['monte_carlo_component_covariance'] = (
        n * np.cov(components, rowvar=False)
    ).tolist()
    result['distribution_rotation_identity_max_abs_error'] = float(
        np.max(np.abs(distribution[:, 0] - distribution[:, 2]))
    )
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
