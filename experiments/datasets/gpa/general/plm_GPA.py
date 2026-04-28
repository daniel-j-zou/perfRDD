import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
from data import Dataset
import pandas as pd


# def TSplm_multi(y, x, q, kn, bwf):
#     """
#     Multivariate version of the Partial Linear Model: y = x @ b + h(q) + e
#     """
#     y, x, q = np.asarray(y), np.asarray(x), np.asarray(q)
#     if x.ndim == 1: x = x[:, None]
#     n, d = x.shape
#
#     # Basis Setup
#     ql, qr = np.min(q), np.ptp(q)
#     if qr == 0: qr = 1.0
#     mu = ql + (np.arange(1, kn + 1) - 0.5) * qr / kn
#     bw = bwf * qr / kn
#
#     # Step 1: Initial Regression
#     Phi = norm.pdf((q[:, None] - mu[None, :]), scale=bw)
#     H = np.column_stack((x, Phi))
#
#     # Variance masking to handle collinearity in basis functions
#     v = H.var(axis=0, ddof=1)
#     thresh1 = 0.03 * np.max(v[d:]) if len(v) > d else 0.0
#     mask1 = v > thresh1
#     be1 = np.zeros(H.shape[1])
#     if mask1.any():
#         be1_sel, *_ = np.linalg.lstsq(H[:, mask1], y, rcond=None)
#         be1[mask1] = be1_sel
#
#     # Step 2: Robinson-style shift correction
#     # Perturb the score slightly to separate X from the nonparametric part
#     x_pert = 0.01 * x.sum(axis=1)
#     Phi_shift = norm.pdf((q[:, None] + x_pert[:, None] - mu[None, :]), scale=bw)
#     Hs = np.column_stack((x, Phi_shift))
#     diff = (Hs - H) @ be1
#     H2 = np.column_stack((x, diff, H[:, d:]))
#
#     v2 = H2.var(axis=0, ddof=1)
#     thresh2 = 0.03 * np.max(v2[d + 1:]) if len(v2) > d + 1 else 0.0
#     mask2 = v2 > thresh2
#     be = np.zeros(H2.shape[1])
#     if mask2.any():
#         be_sel, *_ = np.linalg.lstsq(H2[:, mask2], y, rcond=None)
#         be[mask2] = be_sel
#
#     return be[0:d], be[d + 1:], mu, bw

import numpy as np
from scipy.stats import norm


def TSplm_multi(y, x, q, kn, bwf):
    """
    Multivariate Partial linear model:
        y = x @ b + h(q) + error
    where x can be a vector or a matrix.
    """
    y = np.asarray(y).flatten()
    q = np.asarray(q).flatten()
    x = np.asarray(x)

    # Ensure x is 2D: (n, k)
    if x.ndim == 1:
        x = x[:, np.newaxis]

    n, k = x.shape

    # Range and centers for q
    ql = np.min(q)
    qr = np.ptp(q)
    if qr == 0: qr = 1.0

    idx = np.arange(1, kn + 1) - 0.5
    mu = ql + idx * qr / kn
    bw = bwf * qr / kn

    # Gaussian basis: Phi shape (n, kn)
    Phi = norm.pdf((q[:, np.newaxis] - mu[np.newaxis, :]), scale=bw)

    # First regression: y ~ x + Phi
    # H shape: (n, k + kn)
    H = np.column_stack((x, Phi))

    # Variance masking to prevent collinearity
    v = np.var(H, axis=0, ddof=1)
    # Threshold based on the basis functions (columns indices k to end)
    thresh1 = 0.03 * np.max(v[k:]) if len(v) > k else 0.0
    mask1 = v > thresh1

    be1 = np.zeros(H.shape[1])
    H_sel = H[:, mask1]
    if H_sel.shape[1] > 0:
        be1_sel, *_ = np.linalg.lstsq(H_sel, y, rcond=None)
        be1[mask1] = be1_sel

    # Second regression:
    # Shift q by a small fraction of the linear prediction from x
    # x @ be1[:k] gives the linear component prediction
    x_effect = x @ be1[:k]

    # Phi_shift shape (n, kn)
    Phi_shift = norm.pdf((q[:, np.newaxis] + 0.01 * x_effect[:, np.newaxis] - mu[np.newaxis, :]), scale=bw)

    # Hs is the "shifted" version of H
    Hs = np.column_stack((x, Phi_shift))

    # diff = (Hs - H) @ be1
    # This captures the change in the basis approximation relative to the shift
    diff = (Hs - H) @ be1

    # H2 shape: (n, k + 1 + kn)
    # [x (k cols), diff (1 col), Phi (kn cols)]
    H2 = np.column_stack((x, diff, Phi))

    v2 = np.var(H2, axis=0, ddof=1)
    # Threshold based on the basis functions (columns indices k+1 to end)
    thresh2 = 0.03 * np.max(v2[k + 1:]) if len(v2) > (k + 1) else 0.0
    mask2 = v2 > thresh2

    be = np.zeros(H2.shape[1])
    H2_sel = H2[:, mask2]
    if H2_sel.shape[1] > 0:
        be_sel, *_ = np.linalg.lstsq(H2_sel, y, rcond=None)
        be[mask2] = be_sel

    # Results extraction
    b = be[:k]  # Coefficients for multivariate x
    h = be[k + 1:]  # Coefficients for nonparametric h(q)

    return h, b, mu, bw

filepath_dta = '/Users/lukat/PycharmProjects/GPA_STATA_Data/Dep_Data/AEJApp2008-0202_data/data_for_analysis.dta'
df = pd.read_stata(filepath_dta, convert_categoricals=False)

x_cols = [
    'hsgrade_pct', 'totcredits_year1', 'loc_campus1', 'loc_campus2',
    'male', 'bpl_north_america', 'age_at_entry', 'english'
]

all_cols = ['nextGPA', 'dist_from_cut'] + x_cols
df_clean = df[all_cols].dropna()

dataset = Dataset(
    Y=df_clean['nextGPA'].values.astype(float),
    Q=df_clean['dist_from_cut'].values.astype(float),
    X=df_clean[x_cols].values.astype(float),
    W=np.zeros(len(df_clean)),
    eta=np.zeros(len(df_clean)),
    eps=np.zeros(len(df_clean)),
    params={'phi': 0.0}
)

y, q, x = dataset.Y, dataset.Q, dataset.X

bandwidth = 0.6
mask_bw = (q >= -bandwidth) & (q <= bandwidth)

y, q, x = y[mask_bw], q[mask_bw], x[mask_bw]
n = len(y)
phi = dataset.params.get('phi', 0.0)
KN = 20 # Number of knots from plm.py

idx_all = np.arange(n)
iTr = idx_all[q < phi]
iCon = idx_all[q >= phi]

X_design = np.column_stack((np.ones(n), x))
coef_q, *_ = np.linalg.lstsq(X_design, q, rcond=None)
b0_hat = coef_q[0]
b1_hat = coef_q[1:]
etaHat = q - b0_hat - x @ b1_hat

etaCon = etaHat[iCon]
etaTr = etaHat[iTr]

# Partial linear fits on controls and treated
hCon, bCon, muCon, bwCon = TSplm_multi(y[iCon], x[iCon], etaCon, KN, 1.0)
hTr, bTr, muTr, bwTr = TSplm_multi(y[iTr], x[iTr], etaTr, KN, 1.0)

# Construct alpha(eta) estimates for treated sample
dmt = norm.pdf((etaTr[:, None] - muTr[None, :]), scale=bwTr)
dmc = norm.pdf((etaTr[:, None] - muCon[None, :]), scale=bwCon)
insupp = (dmt.sum(axis=1) > 1.0) & (dmc.sum(axis=1) > 1.0)
if insupp.any():
    ewA = np.mean(dmt[insupp] @ hTr - dmc[insupp] @ hCon)
else:
    ewA= np.nan

print(f"Treatment Effect (Alpha): {ewA:.4f}")

# 1. Generate basis expansions for the WHOLE sample (everyone)
# We use the centers (mu) and bandwidths (bw) estimated from the split samples
dm_treated_all = norm.pdf((etaHat[:, None] - muTr[None, :]), scale=bwTr)
dm_control_all = norm.pdf((etaHat[:, None] - muCon[None, :]), scale=bwCon)

# 2. Check "Support" (ensure everyone is within the range of both models)
# This is a key step in semiparametric models: we only average over people
# who 'look' like they could have been in either group based on their eta.
insupp_all = (dm_treated_all.sum(axis=1) > 1.0) & (dm_control_all.sum(axis=1) > 1.0)
is_treated = (q < phi)

# 3. Predict outcomes for everyone under both scenarios
# We include the linear part (x @ b) AND the nonparametric part (dm @ h)
y1_hat = dm_treated_all @ hTr
y0_hat = dm_control_all @ hCon

# 4. Calculate Individual Treatment Effects and average them
individual_effects = y1_hat - y0_hat
att = np.mean(individual_effects[is_treated & insupp_all])
ate = np.mean(individual_effects[insupp_all])

import matplotlib.pyplot as plt

# 1. Create a grid of eta values for the plot
# We use the range of etaHat that exists in both treated and control samples
eta_min = min(etaTr.min(), etaCon.min())
eta_max = max(etaTr.max(), etaCon.max())
eta_grid = np.linspace(eta_min, eta_max, 200)

# 2. Generate basis expansions for the grid
# Using the mu and bw parameters from your fitted models
dm_tr_grid = norm.pdf((eta_grid[:, None] - muTr[None, :]), scale=bwTr)
dm_co_grid = norm.pdf((eta_grid[:, None] - muCon[None, :]), scale=bwCon)

# 3. Calculate the linear component difference at the mean of X
# We evaluate the X-part at the sample mean to see the 'average' effect shape
# x_mean = np.mean(x, axis=0)
# linear_diff = x_mean @ bTr - x_mean @ bCon

# 4. Calculate alpha(eta)
# This is the difference in nonparametric parts + the baseline linear difference
alpha_eta = (dm_tr_grid @ hTr - dm_co_grid @ hCon) # + linear_diff

# 5. Plotting
plt.figure(figsize=(10, 6))
plt.plot(eta_grid, alpha_eta, color='blue', lw=2, label=r'$\hat{\alpha}(\eta)$')

# Add a horizontal line for the average ATE you calculated earlier
plt.axhline(y=ate, color='red', linestyle='--', label=f'Mean ATE ({ate:.4f})')
plt.axhline(y=att, color='green', linestyle='--', label=f'Mean ATT ({att:.4f})')

plt.title(f'Estimated Treatment Effect as a function of Score Residual ($\eta$) KN = {KN}')
plt.xlabel('Score Residual ($\eta$)')
plt.ylabel('Treatment Effect (Change in GPA)')
plt.grid(True, alpha=0.3)
plt.legend()
plt.show()

print(f"Average Treatment Effect (ATE): {ate:.4f}")
print(f"Sample Size for ATE: {len(individual_effects)} (out of {n})")
# print(df.columns)
# # Identify Treated (Admitted) and Control (Denied)
# idx_all = np.arange(n)
# iTr = idx_all[q > phi]
# iCon = idx_all[q <= phi]
#
# # First stage: regress q on [1, x] to estimate gamma and recover etaHat
# X_design = np.column_stack((np.ones(n), x))
# coef_q, *_ = np.linalg.lstsq(X_design, q, rcond=None)
# b0_hat = coef_q[0]
# b_vec_hat = coef_q[1:]
# etaHat = q - (b0_hat + x @ b_vec_hat)
#
# etaCon = etaHat[iCon]
# etaTr = etaHat[iTr]
#
# # Partial linear fits on controls and treated
# bCon, hCon, muCon, bwCon = TSplm_multi(y[iCon], x[iCon], etaCon, KN, 1.0)
# bTr, hTr, muTr, bwTr = TSplm_multi(y[iTr], x[iTr], etaTr, KN, 1.0)
#
# # Construct alpha(eta) estimates for treated sample
# dmt = norm.pdf((etaTr[:, None] - muTr[None, :]), scale=bwTr)
# dmc = norm.pdf((etaTr[:, None] - muCon[None, :]), scale=bwCon)
# insupp = (dmt.sum(axis=1) > 1.0) & (dmc.sum(axis=1) > 1.0)
# if insupp.any():
#     y_hat_treated = dmt[insupp] @ hTr
#     y_hat_control = dmc[insupp] @ hCon
#     ewA_val = np.mean(y_hat_treated - y_hat_control)
# else:
#     ewA_val = np.nan
#
# print(f"Treatment Effect (Alpha): {ewA_val:.4f}")
# gX_hat = q - etaHat
# xsl, xsr = gX_hat.min(), gX_hat.max()
# j_grid = np.arange(1, n + 1)
# t_j = xsl + (xsr - xsl) * j_grid / n
# Gnbar = (gX_hat[:, None] > t_j[None, :]).mean(axis=0)

