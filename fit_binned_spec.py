#Monica Rizzo, 2020
import numpy as np
import numpy.linalg as la


def calc_G(n, m, eccentricity, inclination, n_sinusoid=1):
    """
    n: number of data pts
    m: length of weight vector
    """
    e_max = max(eccentricity)
    iota_max = max(inclination)
    
    G = np.zeros((n, m))

    for i, j in zip(range(0, 4*n_sinusoid, 4), range(1, n_sinusoid+1)):
            #populate sin basis
            G[:, i] = np.cos(j * 2 * np.pi / e_max * eccentricity)
            G[:, i+1] = np.sin(j * 2 * np.pi / e_max * eccentricity)
            G[:, i+2] = np.cos(j * 2 * np.pi / iota_max * inclination)
            G[:, i+3] = np.sin(j * 2 * np.pi / iota_max * inclination)
    
    #populate linear terms
    G[:, -1] = 1.
    G[:, -2] = inclination
    G[:, -3] = 1.
    G[:, -4] = eccentricity
    
    return G
    
def pred_bin_value(eccentricity, inclination, e_max, iota_max, weights, n_sinusoid=1):
    
    fit_func = np.zeros(len(weights))
    
    for i, j in zip(range(0, 4*n_sinusoid, 4), range(1, n_sinusoid+1)):
        fit_func[i] = np.cos(j * 2 * np.pi / e_max * eccentricity)
        fit_func[i+1] = np.sin(j * 2 * np.pi / e_max * eccentricity)
        fit_func[i+2] = np.cos(j * 2 * np.pi / iota_max * inclination)
        fit_func[i+3] = np.sin(j * 2 * np.pi / iota_max * inclination)
    
        #populate linear terms
    fit_func[-1] = 1.
    fit_func[-2] = inclination
    fit_func[-3] = 1.
    fit_func[-4] = eccentricity
    
    return np.dot(fit_func, weights)


def fit_bin(bin_num, hist_data, eccentricity, inclination, eps, n_sinusoid=1, verbose=False):
    #sinusoids with a polynomial term
    #

    #start m
    #2 = e and iota
    #2 = sin and cos
    #+2 = linear and constant offset
    m = np.ones(2 * (2*n_sinusoid + 2))

    G = calc_G(len(hist_data)-1, len(m), eccentricity, inclination, n_sinusoid)

    R = np.eye(np.dot(G.T, G).shape[0])

    if verbose:
        print("G shape:", G.shape)

    #difference between specified bin and reference spec
    d = np.log10(hist_data[1:, bin_num]) - np.log10(hist_data[0, bin_num])

    #calculate new m
    G_inv = np.dot(la.inv(np.dot(G.T, G) + eps**2 * R), G.T)
    #G_inv = np.dot(la.inv(np.dot(G.T, G)), G.T)
    m = np.dot(G_inv, d)

    d_pred = np.dot(G, m)

    cost = np.sum(np.dot((d - d_pred), (d - d_pred)))

    return m, G, d_pred, d, cost
