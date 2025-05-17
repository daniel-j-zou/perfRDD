#libraries
import numpy as np
import matplotlib.pyplot as plt

# Functions
def sim_q(n, z_bar, z_var, eta_var, gamma):
    "function that creates a dataset of Q to use in the model"
    z = np.random.normal(z_bar, z_var, n)
    i_1 = np.ones(n)
    eta = np.random.normal(0, eta_var, n)
    q = i_1 + gamma * z + eta
    return z, q

def binary(q, phi):
    "The 1 value that checks if phi is crossed"
    oneVector = []
    for i in range(len(q)):
        if q[i] > phi:
            oneVector.append(1)
        else:
            oneVector.append(0)
    return np.asarray(oneVector)

def sim_y(n, z_bar, z_var, eta_var, gamma, phi, x_bar, x_var, theta, demographics):
    "Meant to simulate Y as a dataset"
    x = np.random.normal(x_bar, x_var, n)
    nu = np.random.normal(0, 1, n)
    q = sim_q(n, z_bar, z_var, eta_var, gamma)
    if demographics == True:
        y = w*binary(q[1], phi) + theta*q[0]+ nu
    if demographics == False:
        y = w*binary(q[1], phi) - theta*x + nu
    return y, q[0], q[1]

def non_perf_data(n, z_bar, z_var, eta_var, gamma, phi, x_bar, x_var, theta, demographics):
    results = sim_y(n, z_bar, z_var, eta_var, gamma, phi, x_bar, x_var, theta, demographics)

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




    plt.scatter(results[1], results[0], c='royalblue')
    plt.xlabel("Z value")
    plt.ylabel("Y value")
    plt.show()

    plt.scatter(results[2], results[0], c='royalblue')
    plt.axvline(x=phi, color='black', linestyle='--', label=f'phi = {phi}')
    plt.plot(non_treatment_fit, non_treatment_lsrl(non_treatment_fit), color='red')
    plt.plot(treatment_fit, treatment_lsrl(treatment_fit), color='red')
    plt.xlabel("Q value")
    plt.ylabel("Y value")
    plt.show()

#Variables
n = 10000
z_bar = 0
z_var = 1
eta_var = 1
gamma = 1
phi = 1.5
w = 3
x_bar = 0
x_var = 1
theta = 1

# Example
non_perf_data(n, z_bar, z_var, eta_var, gamma, phi, x_bar, x_var, theta, False) # Z != Q
non_perf_data(n, z_bar, z_var, eta_var, gamma, phi, x_bar, x_var, theta, True) # Z = Q





# Different data simulations