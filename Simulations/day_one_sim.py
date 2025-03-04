import os
import numpy as np
import sklearn.linear_model as reg


os.chdir("C:/Users/ltext/perfRDD/Simulations")
gamma = 10
n = 10000


np.random.seed(41)
Z = np.random.normal(0, 1, n)
eta = np.random.normal(0, 1, n)

Q = gamma * Z + eta

array = np.column_stack((Z, eta))
model = reg.LinearRegression().fit(Z.reshape(-1,1), Q)

# print(array[0:5, 0:5])
# print(Q[0:5])
gamma_hat = model.coef_[0]
# print(gamma_hat)
# print(model.intercept_)

eta_t = Q - gamma_hat * Z
phi = np.mean(Q)
print(phi)
# print(eta_t[0:5])

S_a = []
S_d = []

for i in range(n):
    if Q[i] > phi:
        S_a.append(i)
    else:
        S_d.append(i)

print(len(S_a))
print(len(S_d))

