#libraries
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.optimize import brentq
from sklearn.linear_model import LinearRegression
import pandas as pd
import seaborn as sns
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
        y = w*binary(q[1], phi) - theta*x + nu
    return y, q[0], q[1], x, w

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
    eta_t = q_t - gamma_hat*z_t - np.polyfit(z_t, q_t, deg = 1)[1]
    intercept_1_hat = np.polyfit(z_t, q_t, deg = 1)[1]
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
    w = step_one[0][4]
    alpha_eta_set = []
    y_t_minus_y_s_t = []
    x_t_minus_x_s_t = []
    y_set = step_one[0][0]
    x_set = step_one[0][3]
    for i in range(n):
        if i in s_a_tilde:
            y_t_minus_y_s_t.append(y_set[i] - y_set[s_t_and_t_s[i]])
            x_t_minus_x_s_t.append(x_set[i] - x_set[s_t_and_t_s[i]])
            alpha_eta_set.append(eta_t[i])
        if i in s_d:
            continue
    predictors = np.array([alpha_eta_set, x_t_minus_x_s_t]).T
    model = LinearRegression()
    model.fit(predictors, y_t_minus_y_s_t)
    theta_transpose = model.coef_[1]

    # Step 6
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

    def optimal_mbs(phi_prime):
        numerator = n * little_u_evo(phi_prime)
        sum_three = 0
        for i in s_a_tilde:
            sum_one = sum_three + little_g_hat(phi_prime - eta_t[i], gamma_hat, z_t)
        sum_four = 0
        for i in s_d_tilde:
            sum_four = sum_four + little_g_hat(phi_prime - eta_t[i], gamma_hat, z_t)
        denominator = sum_three + sum_four
        return ((numerator/denominator) - u_mbs(phi_prime))

    def optimal_evo():
        try:
            phi_hat_evo = brentq(little_u_evo, -10, 10)
        except:
            phi_hat_evo = np.nan
        return phi_hat_evo

    mu = 1
    sigma = np.sqrt(1 + gamma**2)
    x_curve = np.linspace(-5, 5, 100)
    y_pdf = norm.pdf(x_curve, loc=mu, scale=sigma)
    y_cdf = norm.cdf((x_curve - mu) / sigma, loc=0, scale=1)
    y_curve = y_pdf - c * (1 - y_cdf)

    sigma_hat = np.sqrt(1 + gamma_hat**2)
    x_curve_hat = np.linspace(-5, 5, 100)
    y_pdf_hat = norm.pdf(x_curve, loc=intercept_1_hat, scale=sigma)
    y_cdf_hat = norm.cdf((x_curve - intercept_1_hat) / sigma, loc=0, scale=1)
    y_curve_hat = y_pdf_hat - c * (1 - y_cdf_hat)

    # Graphs of interest
    if plot == True:
        data_x = []
        data_y = []
        for i in range(-10, 11):
            data_x.append(i/2)
            data_y.append(u_evo(i/2))
        m = np.nanmean(data_y)
        plt.scatter(data_x, data_y)
        plt.title(f"u_evo c = {c}; avg = {m}; n = {n}")
        plt.plot(x_curve, y_curve, color='r')
        plt.plot(x_curve_hat, y_curve_hat, color='green')
        plt.show()

        # data_x = []
        # data_y = []
        # for i in range(-10, 11):
        #     data_x.append(i/2)
        #     data_y.append(u_mbs(i/2))
        # m = np.nanmean(data_y)
        # plt.scatter(data_x, data_y)
        # plt.title(f"u_mbs c = {c}; avg = {m}; n = {n}")
        # plt.show()
        #
        # data_x = []
        # data_y = []
        # for i in range(-10, 11):
        #     data_x.append(i/2)
        #     data_y.append(little_u_evo(i/2))
        # m = np.nanmean(data_y)
        # plt.scatter(data_x, data_y)
        # plt.title(f"little_u_evo c = {c}; avg = {m}; n = {n}")
        # plt.show()
        #
        # data_x = []
        # data_y = []
        # for i in range(-10, 11):
        #     data_x.append(i/2)
        #     data_y.append(optimal_mbs(i/2))
        # m = np.nanmean(data_y)
        # plt.scatter(data_x, data_y)
        # plt.title(f"phi_mbs c = {c}; avg = {m}; n = {n}")
        # plt.show()

    return s_a, s_d, z_t, q_t, gamma_hat, eta_t, theta_transpose, step_one[0][4], optimal_evo()

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
demographics = False
c = 1

# print(algorithm_two(n, z_bar, z_var, eta_var, gamma, phi, x_bar, x_var, theta, w_func, rho, demographics, True, c)[8])

# Monte Carlo for Expectation
monte_carlo_values = []
z = sim_q(n, z_bar, z_var, eta_var, gamma)[0]
domain = np.linspace(-10, 10, 1000)
for i in range(n):
    standard_cdf = norm.cdf(domain - 1 - gamma*z[i], 0, 1)
    value = 1 - standard_cdf
    monte_carlo_values.append(value)

true_expectation_curve = 1 - norm.cdf((domain - 1) / np.sqrt(1 + gamma**2), 0, 1)
monte_carlo_function = np.mean(monte_carlo_values, axis = 0)

plt.plot(domain, monte_carlo_function, label="Monte Carlo", color='red')
plt.plot(domain, true_expectation_curve, label="True Expectation", color='blue')
plt.legend(loc='upper right')
plt.show()

# Theory Checking
values = []
variances = []
biases = []
means = []
sims = []
n_vec = [100, 200, 500, 1000, 1250, 1500, 2000, 5000, 10000]
true_mean = 3
# for m in n_vec:
# for i in range(1):
#     simulations = str(100)
#     print(f"processing: {m}")
    # algorithm_two(m, z_bar, z_var, eta_var, gamma, phi, x_bar, x_var, theta, w_func, rho, demographics, True, c)
    # algorithm_two(1000, z_bar, z_var, eta_var, gamma, phi, x_bar, x_var, theta, w_func, rho, demographics, True, c)
#     for i in range(1, 100):
#         x = algorithm_two(m, z_bar, z_var, eta_var, gamma, phi, x_bar, x_var, theta, w_func, rho, demographics, False, c)[8]
#         values.append(x)
#         sims.append([m, x])
#
#     print(f"Optimal Phi Hat Evo for {simulations} simulations and n = {m}:", np.nanmean(values))
#     print(f"Variance of Phi Hat Evo for {simulations} simulations and n = {m}:", np.nanvar(values))
#     print(f"Bias of Phi Hat Evo for {simulations} simulations and n = {m}:", np.nanmean(values) - true_mean, "\n")
#
#     means.append([m, np.nanmean(values)])
#     variances.append([m, np.nanvar(values)])
#     biases.append([m, np.abs(np.nanmean(values) - true_mean)])
#
# df_means = pd.DataFrame(means, columns=["n", "mean"])
# df_variances = pd.DataFrame(variances, columns=["n", "var"])
# df_biases = pd.DataFrame(biases, columns=["n", "bias"])
# df_sims = pd.DataFrame(sims, columns=["n", "simulations"])
#
# sns.violinplot(x = 'n', y = 'simulations', data = df_sims)
# plt.title("Phi Hat Evos for different n")
# plt.xlabel("n")
# plt.ylabel("simulations")
# plt.show()
#
# sns.barplot(x='n', y='mean', data=df_means)
# plt.title("Mean over 100 Simulations")
# plt.xlabel("Number of Simulations")
# plt.ylabel("Mean")
# plt.show()
#
# sns.barplot(x='n', y='var', data=df_variances)
# plt.title("Var over 100 Simulations")
# plt.xlabel("Number of Simulations")
# plt.ylabel("Variance")
# plt.show()
#
# sns.barplot(x='n', y='bias', data=df_biases)
# plt.title("Absolute Value of Bias over 100 Simulations")
# plt.xlabel("Number of Simulations")
# plt.ylabel("Absolute Value of Bias")
# plt.show()

# Convergence Checking
# n_vec = [200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000]
# data_list_theta = []
# variances_theta = []
# data_list_eta = []
# variances_eta = []
#
# for j, n in enumerate(n_vec):
#     temp_data_theta = []
#     theta_data = []
#     temp_data_eta = []
#     eta_data = []
#
#     for _ in range(100):
#         alg_one = algorithm_two(n_vec[j], z_bar, z_var, eta_var, gamma, phi, x_bar, x_var, theta, w_func, rho, demographics, False, c)
#         theta_residual = (theta - alg_one[6])
#         eta_residual = np.mean(alg_one[7] - alg_one[5])
#         temp_data_theta.append([n, theta_residual])
#         temp_data_eta.append([n, eta_residual])
#         theta_data.append(theta_residual)
#         eta_data.append(eta_residual)
#         print(eta_residual)
#     print(n)
#     variances_theta.append([n, np.var(theta_data)])
#     data_list_theta.extend(temp_data_theta)
#     variances_eta.append([n, np.var(eta_data)])
#     data_list_eta.extend(temp_data_eta)
#
#
# df_theta = pd.DataFrame(data_list_theta, columns=['n_val', 'theta_residual'])
# df_var_theta = pd.DataFrame(variances_theta, columns=['n_val', 'theta_res_var'])
# df_eta = pd.DataFrame(data_list_eta, columns=['n_val', 'eta_residual'])
# df_var_eta = pd.DataFrame(variances_eta, columns=['n_val', 'eta_res_var'])
#
# sns.violinplot(x='n_val', y='theta_residual', data=df_theta)
# plt.title("Theta Residuals by n_val")
# plt.show()
#
# sns.barplot(x='n_val', y='theta_res_var', data=df_var_theta, errorbar=None)
# plt.title('Theta Residual Variances by n_val')
# plt.xlabel('n_val')
# plt.ylabel('Variance of Theta Residual')
# plt.show()
#
# sns.violinplot(x='n_val', y='eta_residual', data=df_eta)
# plt.title('Eta Mean Residual by n_val')
# plt.show()
#
# sns.barplot(x='n_val', y='eta_res_var', data=df_var_eta, errorbar=None)
# plt.title('Eta Mean Residual Variance by n_val')
# plt.xlabel('n_val')
# plt.ylabel('Variance of Eta Mean Residual Variance')
# plt.show()