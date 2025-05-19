# Objective: Using the functions I made in non_performative_simulations.py, implement Algorithm 1

#libraries
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
# from statsmodels.distributions.empirical_distribution import ECDF
# from statsmodels.nonparametric.kde import KDEUnivariate

# Functions
def sim_q(n, z_bar, z_var, eta_var, gamma):
    "function that creates a dataset of Q to use in the model"
    z = np.random.normal(z_bar, z_var, n)
    i_1 = np.ones(n)
    eta = np.random.normal(0, eta_var, n)
    q = i_1 + gamma * z + eta
    return z, q, eta

def binary(q, phi):
    "The 1 value that checks if phi is crossed"
    oneVector = []
    for i in range(len(q)):
        if q[i] > phi:
            oneVector.append(1)
        else:
            oneVector.append(0)
    return np.asarray(oneVector)

def sim_y(n, z_bar, z_var, eta_var, gamma, phi, x_bar, x_var, theta, w, rho, demographics):
    "Meant to simulate Y as a dataset"
    epsilon = np.random.normal(0, 1, n)
    q = sim_q(n, z_bar, z_var, eta_var, gamma)
    nu = rho*q[2] + epsilon
    if demographics == True:
        x = q[0]
        y = w*binary(q[1], phi) + theta*x+ nu
    if demographics == False:
        x = np.random.normal(x_bar, x_var, n)
        y = w*binary(q[1], phi) - theta*x + nu
    return y, q[0], q[1], x

def non_perf_data(n, z_bar, z_var, eta_var, gamma, phi, x_bar, x_var, theta, w, rho, demographics, plot):
    "Runs a dataset and makes plots if wanted"
    results = sim_y(n, z_bar, z_var, eta_var, gamma, phi, x_bar, x_var, theta, w, rho, demographics)

    non_treatment_mask = (results[2] < phi)
    non_treatment_subset = results[2][non_treatment_mask]
    non_treatment_results = results[0][non_treatment_mask]

    treatment_mask = (results[2] >= phi)
    treatment_subset = results[2][treatment_mask]
    treatment_results = results[0][treatment_mask]

    non_treatment_coeff = np.polyfit(non_treatment_subset, non_treatment_results, deg=1)
    non_treatment_lsrl = np.poly1d(non_treatment_coeff)
    non_treatment_fit = np.linspace(np.min(non_treatment_subset), phi, len(non_treatment_subset))

    treatment_coeff = np.polyfit(treatment_subset, treatment_results, deg=1)
    treatment_lsrl = np.poly1d(treatment_coeff)
    treatment_fit = np.linspace(phi, np.max(treatment_subset), len(treatment_subset))
    if plot == True:
        plt.scatter(results[1], results[0], c='royalblue')
        plt.xlabel("Z value")
        plt.ylabel("Y value")
        plt.show()

        plt.scatter(results[1], results[2], c='royalblue')
        plt.xlabel("Z value")
        plt.ylabel("Q value")
        plt.show()

        plt.scatter(results[2], results[0], c='royalblue')
        plt.axvline(x=phi, color='black', linestyle='--', label=f'phi = {phi}')
        plt.plot(non_treatment_fit, non_treatment_lsrl(non_treatment_fit), color='red')
        plt.plot(treatment_fit, treatment_lsrl(treatment_fit), color='red')
        plt.xlabel("Q value")
        plt.ylabel("Y value")
        plt.show()

    return results, non_treatment_subset, non_treatment_results, treatment_subset, treatment_results

def algorithm_one(n, z_bar, z_var, eta_var, gamma, phi, x_bar, x_var, theta, w, rho, demographics, plot):
    # Step 1
    step_one = non_perf_data(n, z_bar, z_var, eta_var, gamma, phi, x_bar, x_var, theta, w, rho, demographics, plot)
    t_set = step_one[0][2]
    s_a = []
    s_d = []
    for i in range(len(t_set)):
        if t_set[i] <= phi:
            s_d.append(i)
        else:
            s_a.append(i)

    # Step 2
    z_t = step_one[0][1]
    q_t = step_one[0][2]
    gamma_hat = np.polyfit(z_t, q_t, deg=1)[0]
    eta_t = q_t - gamma_hat*z_t

    # Step 3 (assume Z is Gaussian, for now)
        # big_g_hat = ECDF(gamma_hat*z_t)
        # big_g_hat_bar = 1 - big_g_hat
        # little_g_hat = KDEUnivariate(gamma_hat*z_t)
        # little_g_hat.fit(bw='scott')
    z_t_mean = np.mean(z_t)
    z_t_std = np.std(z_t)

    def big_g_hat(x, gamma_hat, z_t):
        mu = np.mean(gamma_hat*z_t)
        sigma = np.std(gamma_hat*z_t)
        return norm.cdf(x, mu, sigma)

    def little_g_hat(x, gamma_hat, z_t):
        mu = np.mean(gamma_hat*z_t)
        sigma = np.std(gamma_hat*z_t)
        return norm.cdf(x, mu, sigma)

    def big_g_hat_bar(x, gamma_hat, z_t):
        mu = np.mean(gamma_hat*z_t)
        sigma = np.std(gamma_hat*z_t)
        return 1 - norm.cdf(x, mu, sigma)

    # Step 4
    eta_s_d = []
    eta_s_a = []
    for i in s_a:
        eta_s_a.append(eta_t[i])
    for i in s_d:
        eta_s_d.append(eta_t[i])
    s_t_set = []
    t_s_set = []
    s_t_and_t_s = []
    for i in range(n):
        if i in s_a:
            s_t = np.argmin(np.abs(eta_s_d - eta_t[i]))
            s_t_set.append(s_d[s_t])
            s_t_and_t_s.append(s_d[s_t])
        if i in s_d:
            t_s = np.argmin(np.abs(eta_s_a - eta_t[i]))
            t_s_set.append(s_a[t_s])
            s_t_and_t_s.append(s_a[t_s])
    zeta = 1
    s_a_tilde = []
    s_d_tilde = []
    for i in range(len(s_a)):
        if np.abs(eta_t[s_a[i]] - eta_t[s_t_set[i]]) < n**(-1*zeta):
            s_a_tilde.append(s_a[i])
    for i in range(len(s_d)):
        if np.abs(eta_t[s_d[i]] - eta_t[t_s_set[i]]) < n**(-1*zeta):
            s_d_tilde.append(s_d[i])

    # Step 5 (I assume the PLM is for all 10000 points, hence s_t_and_t_s)

    # Compute alpha(eta) normally, but because W is constant in this case, it is just W

    y_t_minus_y_s_t = []
    x_t_minus_x_s_t = []
    y_set = step_one[0][0]
    x_set = step_one[0][3]
    for i in range(n):
        y_t_minus_y_s_t.append(y_set[i] - y_set[s_t_and_t_s[i]])
        x_t_minus_x_s_t.append(x_set[i] - x_set[s_t_and_t_s[i]])


    return s_a, s_d, z_t, q_t, gamma_hat, eta_t,

n = 10000
z_bar = 0
z_var = 1
eta_var = 1
gamma = 1
phi = 1.5
w = 130
x_bar = 0
x_var = 1
theta = 1
rho = 8
demographics = True

algorithm_one(n, z_bar, z_var, eta_var, gamma, phi, x_bar, x_var, theta, w, rho, demographics, True)
