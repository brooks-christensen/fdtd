import numpy as np


def vacuum(N):
    eps_r = np.ones(N)
    mu_r = np.ones(N)
    sigma_e = np.zeros(N)
    return eps_r, mu_r, sigma_e


def thin_film(N, i_start, width, n_film=1.5, sigma=0.0):
    eps_r = np.ones(N)
    mu_r = np.ones(N)
    sigma_e = np.zeros(N)
    i_end = min(N, i_start + width)
    eps_r[i_start:i_end] = n_film**2
    sigma_e[i_start:i_end] = sigma
    return eps_r, mu_r, sigma_e


def bulk_medium(N, i_start, n_bulk=2.0, sigma=0.0):
    eps_r = np.ones(N)
    mu_r = np.ones(N)
    sigma_e = np.zeros(N)
    eps_r[i_start:] = n_bulk**2
    sigma_e[i_start:] = sigma
    return eps_r, mu_r, sigma_e
