#libraries
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.optimize import brentq
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

def sim_y(n, z_bar, z_var, eta_var, gamma, phi, x_bar, x_var, theta, w_func, rho, demographics):
    "Meant to simulate Y as a dataset"
    epsilon = np.random.normal(0, 1, n)
    q = sim_q(n, z_bar, z_var, eta_var, gamma)
    nu = rho*q[2] + epsilon
    if w_func == True:
        w = np.random.normal(np.mean(q[2]), 1)
    if w_func == False:
        w = np.random.normal(-1 * np.mean(q[2]), 1)
    if demographics == True:
        x = q[0]
        y = w*binary(q[1], phi) + theta*x+ nu
    if demographics == False:
        x = np.random.normal(x_bar, x_var, n)
        y = w*binary(q[1], phi) - theta*x + nu
    return y, q[0], q[1], x

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

def algorithm_two(n, z_bar, z_var, eta_var, gamma, phi, x_bar, x_var, theta, w_func, rho, demographics, plot, c):
    # Step 1
    step_one = non_perf_data(n, z_bar, z_var, eta_var, gamma, phi, x_bar, x_var, theta, w_func, rho, demographics, plot)
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
    # Plot eta_t against eta for diagnostics


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
        return norm.pdf(x, mu, sigma)

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
    zeta = 0.5
    s_a_tilde = []
    s_d_tilde = []
    for i in range(len(s_a)):
        if np.abs(eta_t[s_a[i]] - eta_t[s_t_set[i]]) < n**(-1*zeta):
            s_a_tilde.append(s_a[i])
    for i in range(len(s_d)):
        if np.abs(eta_t[s_d[i]] - eta_t[t_s_set[i]]) < n**(-1*zeta):
            s_d_tilde.append(s_d[i])

    # Step 5

    # Compute alpha(eta) normally, but because W is constant in this case, it is just W
    y_t_minus_y_s_t = []
    x_t_minus_x_s_t = []
    y_set = step_one[0][0]
    x_set = step_one[0][3]
    for i in range(n):
        if i in s_a_tilde:
            y_t_minus_y_s_t.append(y_set[i] - y_set[s_t_and_t_s[i]])
            x_t_minus_x_s_t.append(x_set[i] - x_set[s_t_and_t_s[i]])
        if i in s_d:
            continue
    theta_transpose = np.polyfit(x_t_minus_x_s_t, y_t_minus_y_s_t, deg=1)[0]

    # Step 6
    # y_tilde_a = []
    # x_tilde_a = []
    # y_tilde_d = []
    # x_tilde_d = []
    # for i in s_a_tilde:
    #     x_tilde_a.append(x_set[i])
    #     y_tilde_a.append(y_set[i])
    # for i in s_d_tilde:
    #     x_tilde_d.append(x_set[i])
    #     y_tilde_d.append(y_set[i])
    # r_t = np.asarray(y_tilde_a) - theta_transpose*np.asarray(x_tilde_a)
    # r_s = np.asarray(y_tilde_d) - theta_transpose*np.asarray(x_tilde_d)

    r = y_set - theta_transpose*x_set
    def u_evo(phi_prime):
        sum_one = 0
        j = 0
        for i in s_a:
            if i in s_a_tilde:
                k = s_t_set[j]
                sum_one = sum_one + (r[i] - r[k] - c)*big_g_hat_bar(phi_prime - eta_t[i], gamma_hat, z_t)
            j = j + 1
        sum_two = 0
        j = 0
        for i in s_d:
            if i in s_d_tilde:
                k = t_s_set[j]
                sum_two = sum_two + (r[k] - r[i] - c) * big_g_hat_bar(phi_prime - eta_t[i], gamma_hat, z_t)
            j = j + 1
        return ((sum_one + sum_two)/n)
    def u_mbs(phi_prime):
        numerator = n*u_evo(phi_prime)
        sum_three = 0
        for i in s_a_tilde:
            sum_three = sum_three + big_g_hat_bar(phi_prime - eta_t[i], gamma_hat, z_t)
        sum_four = 0
        for i in s_d_tilde:
            sum_four = sum_four + big_g_hat_bar(phi_prime - eta_t[i], gamma_hat, z_t)
        denominator = sum_three + sum_four
        return numerator/denominator

    # Step 7
    def little_u_evo(phi_prime):
        sum_one = 0
        j = 0
        for i in s_a:
            if i in s_a_tilde:
                k = s_t_set[j]
                sum_one = sum_one + (r[i] - r[k] - c) * little_g_hat(phi_prime - eta_t[i], gamma_hat, z_t)
            j = j + 1
        sum_two = 0
        j = 0
        for i in s_d:
            if i in s_d_tilde:
                k = t_s_set[j]
                sum_two = sum_two + (r[k] - r[i] - c) * little_g_hat(phi_prime - eta_t[i], gamma_hat, z_t)
            j = j + 1
        return (((sum_one + sum_two)/ n)*(-1))

    def optimal_function(phi_prime):
        numerator = n * little_u_evo(phi_prime)
        sum_three = 0
        for i in s_a_tilde:
            sum_one = sum_three + little_g_hat(phi_prime - eta_t[i], gamma_hat, z_t)
        sum_four = 0
        for i in s_d_tilde:
            sum_four = sum_four + little_g_hat(phi_prime - eta_t[i], gamma_hat, z_t)
        denominator = sum_three + sum_four
        print(denominator)
        return ((numerator/denominator) - u_mbs(phi_prime))

    try:
        brentq(u_evo, -10, 11)
    except:
        print("Optimization failed: no root")

    # Graphs of interest
    data_x = []
    data_y = []
    for i in range(-10, 11):
        data_x.append(i)
        data_y.append(u_evo(i))
    m = np.nanmean(data_y)
    plt.scatter(data_x, data_y)
    plt.title("u_evo W = 10; c = 5; avg = " + str(m))
    plt.show()

    data_x = []
    data_y = []
    for i in range(-25, 26):
        data_x.append(i)
        data_y.append(u_mbs(i))
    m = np.nanmean(data_y)
    plt.scatter(data_x, data_y)
    plt.title("u_mbs W = 10; c = 5; avg = " + str(m))
    plt.show()

    data_x = []
    data_y = []
    for i in range(-10, 11):
        data_x.append(i)
        data_y.append(little_u_evo(i))
    m = np.nanmean(data_y)
    plt.scatter(data_x, data_y)
    plt.title("little_u_evo W = 10; c = 5; avg = " + str(m))
    plt.show()

    data_x = []
    data_y = []
    for i in range(-10, 11):
        data_x.append(i)
        data_y.append(optimal_function(i))
    m = np.nanmean(data_y)
    plt.scatter(data_x, data_y)
    plt.title("phi_mbs W = 10; c = 5; avg = " + str(m))
    plt.show()

    return s_a, s_d, z_t, q_t, gamma_hat, eta_t

n = 1000
z_bar = 0
z_var = 1
eta_var = 1
gamma = 1
phi = 1.5
x_bar = 0
x_var = 1
theta = 1
w_func = False
rho = 8
demographics = True
c = 5

algorithm_two(n, z_bar, z_var, eta_var, gamma, phi, x_bar, x_var, theta, w_func, rho, demographics, False, c)