#libraries
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.stats import norm, stats
from scipy.stats import linregress
from scipy.optimize import brentq
from sklearn.linear_model import LinearRegression
import pandas as pd
import seaborn as sns

def sim_q(n, z_bar, z_var, eta_var, gamma):
    "function that creates a dataset of Q to use in the model"
    z = np.random.normal(z_bar, z_var, n)
    i_1 = np.zeros(n)
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

def sim_y(n, z_bar, z_var, eta_var, gamma, phi, x_bar, x_var, theta, w_func, rho, demographics):
    "Meant to simulate Y as a dataset"
    epsilon = np.random.normal(0, 1, n)
    q = sim_q(n, z_bar, z_var, eta_var, gamma)
    nu = rho*q[2] + epsilon
    w = []
    if w_func == True:
        for i in range(n):
            reward = np.random.normal(q[2][i], 1)
            w.append(reward)
    if w_func == False:
        for i in range(n):
            reward = np.random.normal(-1 * q[2][i], 1)
            w.append(reward)
    if demographics == True:
        x = q[0]
        y = w*binary(q[1], phi) + theta*x+ nu
    if demographics == False:
        x = np.random.normal(x_bar, x_var, n)
        y = w*binary(q[1], phi) + theta*x + nu
    return y, q[0], q[1], x, w, q[2]

def non_perf_data(n, z_bar, z_var, eta_var, gamma, phi, x_bar, x_var, theta, w_func, rho, demographics, plot):
    "Runs a dataset and makes plots if wanted"
    results = sim_y(n, z_bar, z_var, eta_var, gamma, phi, x_bar, x_var, theta, w_func, rho, demographics)

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

def algorithm_three_one(n, z_bar, z_var, eta_var, gamma, phi, x_bar, x_var, theta, w_func, rho, demographics, c, k):
    # step 1
    data = sim_y(n, z_bar, z_var, eta_var, gamma, phi, x_bar, x_var, theta, w_func, rho, demographics)
    y_data = data[0]
    if demographics == True:
        x_data = data[3]
        z_data = x_data
    else:
        x_data = data[3]
        z_data = data[1]
    q_data = data[2]
    w_data = data[4]
    s_a = []
    s_d = []
    for i in range(n):
        if q_data[i] < phi:
            s_d.append(i)
        else:
            s_a.append(i)

    # step 2
    gamma_hat = np.polyfit(z_data, q_data, 1)[0]
    intercept_hat = np.polyfit(z_data, q_data, 1)[1]
    eta_hat_data = q_data - gamma_hat * z_data - intercept_hat

    # step 3
    gamma_times_z_mean = np.mean(gamma*z_data)
    gamma_times_z_var = np.var(gamma*z_data)

    def big_g_hat(x, gamma_times_z_mean, gamma_times_z_var):
        return norm(x, gamma_times_z_mean, gamma_times_z_var)

    def big_g_hat_bar(x, gamma_times_z_mean, gamma_times_z_var):
        return 1 - big_g_hat(x, gamma_times_z_mean, gamma_times_z_var)

    def little_g_hat(x, gamma_times_z_mean, gamma_times_z_var):
        return norm.pdf(x, gamma_times_z_mean, gamma_times_z_var)

    # step 4
    s_a_tilde = []
    s_d_tilde = []
    zeta = 0.5
    for t in s_a:
        s_t = np.argmin(np.abs(eta_hat_data[s_d] - eta_hat_data[t]))
        if np.abs(eta_hat_data[s_t] - eta_hat_data[t]) < n**(-1*zeta):
            s_a_tilde.append(t)

    for s in s_d:
        t_s = np.argmin(np.abs(eta_hat_data[s_a] - eta_hat_data[s]))
        if np.abs(eta_hat_data[t_s] - eta_hat_data[s]) < n**(-1*zeta):
            s_d_tilde.append(s)


    return s_a, s_d, gamma_hat, gamma_times_z_mean, gamma_times_z_var, s_a_tilde, s_d_tilde

# parameters:
n = 1000
z_bar = 0
z_var = 1
eta_var = 1
gamma = 1
phi = 1.5
x_bar = 0
x_var = 1
theta = 1
w_func = True
rho = 8
demographics = True
c = 1
k_vec = [1]

data = algorithm_three_one(n, z_bar, z_var, eta_var, gamma, phi, x_bar, x_var, theta, w_func, rho, demographics, c, k_vec[0])
print(data[6])
print(data[5])