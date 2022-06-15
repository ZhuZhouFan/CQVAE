import numpy as np
import pandas as pd


def generate_bar_c(N, T):
    c = np.zeros((N, 5000 + T))
    rho = np.random.uniform(0.9, 1)
    # rho = np.random.uniform(0.7, 0.8)
    for t in range(1, 5000 + T):
        c[:, t] = c[:, t-1]*rho + np.random.randn(N)
    return c[:, 5000:]


def generate_c(N, T):
    bar_c = generate_bar_c(N, T)
    bar_c_pd = pd.DataFrame(bar_c)
    c = bar_c_pd.rank(axis = 1).values
    c = 2/(N+1) * c - 1
    return c


def generate_C(N, T, P_c):
    C = np.zeros([N, T, P_c])
    for i in range(P_c):
        C[:, :, i] = generate_c(N, T)
    return C


def linear_g(C, N, T):
    beta = np.zeros([N, T, 3])
    beta[:, :, 0] = 1.2*C[:, :, 1]
    beta[:, :, 1] = 1.0*C[:, :, 2]
    beta[:, :, 2] = 0.8*C[:, :, 3]
    return beta


def nonlinear_g(C, N, T):
    beta = np.zeros([N, T, 3])
    beta[:, :, 0] = C[:, :, 1] * C[:, :, 1]
    beta[:, :, 1] = 2*C[:, :, 1]*C[:, :, 2]
    beta[:, :, 2] = np.sign(C[:, :, 3])
    return beta


def DGP_gu(N, T, P_f, P_x, P_c, W, linear_index):
    x = np.random.multivariate_normal(mean = 0.03*np.ones(P_x), cov = 0.01*np.identity(P_x), size = T)
    eta = np.random.multivariate_normal(mean = np.zeros(P_f), cov = pow(0.01,2) * np.identity(P_f), size = T)
    f = np.matmul(x, W.transpose()) + eta
    epsilon = 0.1 * np.random.standard_t(df = 5, size = (N, T))
    C = generate_C(N, T, P_c)
    if linear_index:
        beta = linear_g(C, N, T)
    else:
        beta = nonlinear_g(C, N, T)
    r = np.zeros([N, T])
    for t in range(T):
        r[:, t] = np.matmul(beta[:, t, :], f[t, :].transpose())
    r += epsilon
    return r, C


def DGP_t3(N, T, P_f, P_x, P_c, W, linear_index):
    x = 0.03 + 0.1 * np.random.standard_t(df = 3, size = (T, P_x))/np.sqrt(3)
    eta = np.random.multivariate_normal(mean = np.zeros(P_f), cov = pow(0.01,2) * np.identity(P_f), size = T)
    f = np.matmul(x, W.transpose()) + eta
    epsilon = 0.1 * np.random.standard_t(df = 5, size = (N, T))
    C = generate_C(N, T, P_c)
    if linear_index:
        beta = linear_g(C, N, T)
    else:
        beta = nonlinear_g(C, N, T)
    r = np.zeros([N, T])
    for t in range(T):
        r[:, t] = np.matmul(beta[:, t, :], f[t, :].transpose())
    r += epsilon
    return r, C