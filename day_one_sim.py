# Q = I_1 + gamma * Z + eta
#  Noise and Z should be normal because we want Q to be normal as
# well as for homoscedasticity and normality of G_bar
# No trivial aspects of the simulation
#
# Create the dataset and runs the algorithm on the dataset
# Make this first step a function that outputs interesting results (optimal threshold in non-performative)

#libraries
import numpy as np
import matplotlib.pyplot as plt

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


results = sim_y(n, z_bar, z_var, eta_var, gamma, phi, x_bar, x_var, theta, True)
print(results[0])

plt.scatter(results[1], results[0])
plt.xlabel("Z value")
plt.ylabel("Y value")
plt.show()

plt.scatter(results[2], results[0])
plt.xlabel("Q value")
plt.ylabel("Y value")
plt.show()