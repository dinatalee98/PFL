import numpy as np
import cvxpy as cp

K = 40 # number of IoT devices
N = 30 # number of round

c = 3e8

c_k = 10
f_k = 5 
alpha_k = 10 ** -28
zeta1 = 9.61
zeta2 = 0.16
H_u = 100
f_c = 2
eta_LoS = 1
eta_NLoS = 20
alpha_0 = -60
beta = 2.2
K_bar = 3
sigma_sq = -110
b_k = 80

E_k = np.full(K, 1)
D_k = np.full(K, 1)
B = 10
kappa = 10
eta = 0.001
epsilon = 0.001
D = np.sum(D_k)
F_w0 = 1
F_star = 0
p_k = np.full(K, 1)
s = 10

location = np.ones((K, N))


def euclidean_distance(uav_pos, device_pos, altitude): # System model d_k(n)
    return np.sqrt(np.sum((uav_pos - device_pos) ** 2) - altitude ** 2)

def computation_time_energy(k, n, a): # Eq 3 & 4
    t_comp = (c_k * D_k[k]) / f_k
    E_comp = a[k][n] * (alpha_k / 2) * c_k * D_k[k] * (f_k ** 2)
    return t_comp, E_comp

def P_LoS(k, n):
    theta_k = np.arctan(H_u / location[k][n]) # d[k][n] distance between UAV and IoT
    return 1 / (1 + zeta1 * np.exp(-zeta2 * (180 / np.pi * theta_k - zeta1)))

def PL_LoS(k, n):
    return 20 * np.log10((4 * np.pi * f_c * location[k][n]) / c) + eta_LoS

def PL_NLoS(k, n):
    return 20 * np.log10((4 * np.pi * f_c * location[k][n]) / c) + eta_NLoS

def path_loss(k, n, P_LoS): # Eq 6 & 7 & 8
    return P_LoS * PL_LoS(k, n) + (1 - P_LoS) * PL_NLoS(k, n)


def channel_coefficient(k, n): # Eq 9, 10, 11
    PL_LoS_k_n = PL_LoS(k, n)
    PL_NLoS_k_n = PL_NLoS(k, n)

    large_scale_fading = alpha_0 * location[k][n] ** beta
    small_scale_fading = np.sqrt((K_bar / (K_bar+1)) * PL_LoS_k_n) + np.sqrt((1 / (K_bar+1)) * PL_NLoS_k_n)
    h_k = np.sqrt(large_scale_fading * small_scale_fading)
    return h_k


def data_rate_and_energy(k, n, a): # Eq 12, 13, 14
    h_k = channel_coefficient(k, n)
    
    R_k = b_k * np.log2(1 + (p_k[n] * np.linalg.norm(h_k) ** 2) / sigma_sq)
    t_comm = (a[k][n] * s) / R_k
    E_comm = (t_comm * (sigma_sq) * (2 ** (a[k][n] * s / (b_k * t_comm)) - 1)) / np.linalg.norm(h_k) ** 2
    return R_k, t_comm, E_comm

