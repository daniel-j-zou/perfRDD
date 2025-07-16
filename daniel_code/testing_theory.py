# In this file, we will be testing if our theoretical computations for U_evo,
# its derivative, and its maximizer are correct by comparing them to our
# numerical computations with Monte Carlo.

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# This function generates our data for some sample size n
#
# Inputs:
# n: sample size
# c: constant
# gamma: constant
# intercept: constant

# Outputs:
# W, Q: both of length n


def data_gen(n, gamma, intercept):
    eta = np.random.normal(0, 1, n)
    z = np.random.normal(0, 1, n)
    Q = intercept + gamma * z + eta
    W = np.random.normal(eta, 1, n)
    return W, Q

# For given constant values, compute the value of U_evo, which is E[(W-c)1(Q>phi)] for different values of phi

# Inputs:
# n: sample size
# c: constant
# gamma: constant
# intercept: constant

# Outputs:
# U_evo: a vector of length 100 for phi values from -5 to 5

def U_evo_gen(n, c, gamma, intercept):
    phi_vec = np.linspace(-5, 5, 1000)
    U_evo = np.zeros(1000)
    W, Q = data_gen(n, gamma, intercept)
    for i in range(1000):
        U_evo[i] = np.mean((W - c) * (Q > phi_vec[i]))
    return U_evo

# Function for the theoretical values of U_evo at a given phi

# Inputs:

# c: constant
# gamma: constant
# intercept: constant
# phi: value of phi
#
# Outputs:

# U_evo: value of U_evo at phi

def U_evo_theory(c, gamma, intercept, phi):
    term1 = (1 / np.sqrt(2 * np.pi * (1 + gamma ** 2))) * \
            np.exp(- (phi - intercept) ** 2 / (2 * (1 + gamma ** 2)))
    # Survival tail of the Gaussian
    term2 = c * (1 - norm.cdf((phi - intercept) / np.sqrt(1 + gamma ** 2)))
    return term1 - term2

def U_evo_theory_vec(c, gamma, intercept):
    phi_vec = np.linspace(-5, 5, 1000)
    U_evo = np.zeros(1000)
    for i in range(1000):
        U_evo[i] = U_evo_theory(c, gamma, intercept, phi_vec[i])
    return U_evo

# Plot the value of U_evo for different values of phi

# Inputs:
# n: sample size
# c: constant
# gamma: constant
# intercept: constant

# Outputs:
# Plot of U_evo and the theoretical values for different values of phi
# include the maximizer in the plot title as well as the constant values

def plot_U_evo(n, c, gamma, intercept):
    U_evo = U_evo_gen(n, c, gamma, intercept)
    maximizer = np.argmax(U_evo)

    U_evo_theory = U_evo_theory_vec(c, gamma, intercept)

    plt.plot(np.linspace(-5, 5, 1000), U_evo, c='royalblue')
    plt.plot(np.linspace(-5, 5, 1000), U_evo_theory, c='red')
    plt.xlabel("Phi")
    plt.ylabel("U_evo")
    plt.title("Maximizer: " + str(np.round(np.linspace(-5, 5, 1000)[maximizer], 2)) + "; c = " + str(c) + "; gamma = " + str(gamma) + "; intercept = " + str(intercept))
    plt.show()


plot_U_evo(1000000, 1, 1, 1)