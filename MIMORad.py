import matplotlib.pyplot as plt
import numpy as np
import random
from scipy.linalg import eig
from scipy.optimize import minimize
from scipy.stats import unitary_group
from scipy.stats import bootstrap
from collections import defaultdict
import time

"""
general unctions
"""
dB2lin = lambda x: pow(10.0, x / 10.0)
lin2dB = lambda x: 10.0 * np.log10(x)
rad2deg = lambda x: (x / np.pi) * 180.0
deg2rad = lambda x: (x / 180.0) * np.pi

"""
specific unctions
"""
def gen_wei_chan(UR, UT, omega):
    """
    generate a MIMO clutter channel following Weicheselberge channel model 
    
    Input:
    ------
        UR: numpy array of shape (Mr, Mr) and type np.complex128, Unitary matrix as RX side eigenbases
        UT: numpy array of shape (MT, MT) and type np.complex128, Unitary matrix as TX side eigenbases
        omega: numpy array of shape (Mr, Mt) and type positive np.float64, power coupling matrix between TX and RX side eigenbases

    Output:
    ------
        Hwei: numpy array of shape (Mr, Mt) and type np.complex128, generated Weichselberger model
    """
    # sanity check on matrix shape
    (Mr, Mt) = omega.shape
    assert UR.shape == (Mr, Mr), "Mismatch in UR.shape: {} with omega.shape: {}".format(UR.shape, omega.shape)
    assert UT.shape == (Mt, Mt), "Mismatch in UT.shape: {} with omega.shape: {}".format(UT.shape, omega.shape)
    
    G = np.random.normal(0, 1, (Mr, Mt))
    Hwei = UR @ (np.sqrt(omega) * G) @ UT.conj().T
    return Hwei

def gen_angular_chan(Mt_row, Mt_col, Mr_row, Mr_col, sub_chan_list, ant_space_wl = 0.5):
    """
    generate angular MIMO channel

    Input:
    ------
        Mt_row: int, number of rows in transmit antenna array
        Mt_col: int, number of columns in transmit antenna array
        Mr_row: int, number of rows in receive antenna array
        Mr_col: int, number of columns in receive antenna array
        sub_chan_list: list of 3-element tuple, (theta_deg, phi_deg, pwr_dB)
        ant_space_wl: float, antenna spacing normalized by wavelength

    Output:
    ------
        Hangular: numpy array of shape (Mr, Mt) and type np.complex128, generated angular model
    """
    Mt = Mt_row * Mt_col
    Mr = Mr_row * Mr_col
    Hangular = np.zeros((Mr, Mt), dtype = np.complex128)

    for sub_chan in sub_chan_list:
        (theta_deg, phi_deg, pwr_dB) = sub_chan
        theta_rad, phi_rad = deg2rad(theta_deg), deg2rad(phi_deg)

        # calculate transmit steering vector
        tx_steering_row = np.exp(1j * 2 * np.pi * np.cos(theta_rad) * ant_space_wl * np.arange(Mt_row))
        tx_steering_col = np.exp(1j * 2 * np.pi * np.sin(theta_rad) * np.cos(phi_rad) * ant_space_wl * np.arange(Mt_col))
        tx_steering_vec = (tx_steering_row.reshape(-1, 1) @ tx_steering_col.reshape(1, -1)).reshape(-1, 1)

        # calculate receive steering vector
        rx_steering_row = np.exp(1j * 2 * np.pi * np.cos(theta_rad) * ant_space_wl * np.arange(Mr_row))
        rx_steering_col = np.exp(1j * 2 * np.pi * np.sin(theta_rad) * np.cos(phi_rad) * ant_space_wl * np.arange(Mr_col))
        rx_steering_vec = (rx_steering_row.reshape(-1, 1) @ rx_steering_col.reshape(1, -1)).reshape(-1, 1)

        Hangular_sub = rx_steering_vec.reshape(-1, 1) @ rx_steering_vec.reshape(1, -1) 
        Hangular_sub *= np.sqrt((dB2lin(pwr_dB)) / (np.sum(abs(Hangular_sub.reshape(-1,))**2)))
        Hangular = Hangular + Hangular_sub

    Hangular *= (np.random.normal(0, 1) + 1j * np.random.normal(0, 1))

    return Hangular

def gen_Rician_chan(Hlos, K_factor):
    """
    generate Rician distributed MIMO channel

    Input:
    ------
        Hlos: numpy array of shape (Mr, Mt) and type np.complex128, deterministic LOS component of the generated Rician channel
        Kfactor: float, K-factor of the Rician channel

    Output:
    ------
        Hrician: numpy array of shape (Mr, Mt) and type np.complex128, generated Rician channel
    """
    (Mr, Mt) = Hlos.shape 
    pwr_Hnlos = np.sum(abs(Hlos.reshape(-1,))**2) / K_factor
    sigma = np.sqrt(pwr_Hnlos / (Mr * Mt * 2))
    Hnlos = np.random.normal(0, sigma, (Mr, Mt)) + 1j * np.random.normal(0, sigma, (Mr, Mt))
    Hrician = Hlos + Hnlos

    # normalize the power of generated channel
    Hrician /= np.sqrt(np.sum(abs(Hrician.reshape(-1,))**2))
    
    return Hrician

def calc_wei_params(Hsamples):
    """
    calculate parameters of Weichselberger model given samples of MIMO channel

    Input:
    ------
        Hsamples: numpy array of shape (Mr, Mt, Num_samples) and type np.complex128, measured MIMO channel samples

    Output:
    ------
        UR_est: numpy array of shape (Mr, Mr) and type np.complex128, estimated Unitary matrix as RX side eigenbases
        UT_est: numpy array of shape (Mt, Mt) and type np.complex128, estimated Unitary matrix as TX side eigenbases
        omega_est: numpy array of shape (Mr, Mt) and type positive np.float64, estimated power coupling matrix between TX and RX side eigenbases
    """
    (Mr, Mt, num_samples) = Hsamples.shape

    # calculate UR, UT
    Rr = np.zeros((Mr, Mr), dtype = np.complex128)
    Rt = np.zeros((Mt, Mt), dtype = np.complex128)

    for idx_samp in range(num_samples):
        H = Hsamples[:, :, idx_samp]
        Rr = Rr + (H @ H.conj().T)
        Rt = Rt + (H.conj().T @ H)

    Rr /= num_samples
    Rt /= num_samples

    _, UR_est = eig(Rr)
    _, UT_est = eig(Rt)

    # calculate omega_est
    omega_est = np.zeros((Mr, Mt))
    for i in range(Mr):
        for j in range(Mt):
            omega_ij = 0.0
            for idx_samp in range(num_samples):
                H = Hsamples[:, :, idx_samp]
                omega_ij += abs(UR_est[:, i].reshape(-1, 1).conj().T @ H @ UT_est[:, j].reshape(-1, 1))[0, 0]**2
            omega_est[i, j] = omega_ij / num_samples

    return UR_est, UT_est, omega_est

def calc_sigma(UR, omega, UT, X, pwr_n):
    """
    calculate sigma for a Weichselberger MIMO model
    
    Input:
    ------
        UR: numpy array of shape (Mr, Mr) and type np.complex128, Unitary matrix as RX side eigenbases
        omega: numpy array of shape (Mr, Mt) and type positive np.float64, power coupling matrix between TX and RX side eigenbases
        UT: numpy array of shape (Mt, Mt) and type np.complex128, Unitary matrix as TX side eigenbases
        X: numpy array of shape (Mt, 1) and type np.complex128, power normalized X
        pwr_n: float, input noise power

    Output:
    ------
        sigma_c: numpy array of shape (Mr, Mr) and type float, positive definite covariance matrix for clutter signal
        sigma_y: numpy array of shape (Mr, Mr) and type float, positive definite covariance matrix for received signal 
        sigma_y_inv: numpy array of shape (Mr, Mr) and type float, inverse of covariance matrix for received signal
    """
    X = X.reshape(-1, 1)
    (Mr, Mt) = omega.shape
    assert UR.shape == (Mr, Mr), "Mismatch in UR.shape: {} with omega.shape: {}".format(UR.shape, omega.shape)
    assert UT.shape == (Mt, Mt), "Mismatch in UT.shape: {} with omega.shape: {}".format(UT.shape, omega.shape)
    assert X.shape == (Mt, 1), "Mismatch in X.shape: {} with omega.shape: {}".format(X.shape, omega.shape)
    
    # convert X to X_T 
    X_T = UT.conj().T @ X
    X_T_pwr = abs(X_T)**2
    
    # calculate inner part
    R = np.zeros((Mr, Mr))
    for i in range(Mr):
        R[i, i] = np.sum(omega[i, :].reshape(-1,) * X_T_pwr.reshape(-1,))

    # calculate sigma_c
    sigma_c = UR @ R @ UR.conj().T
    
    # calculate sigma_y
    sigma_y = sigma_c + pwr_n * np.identity(Mr)
    
    # calculate sigma_y_inv
    sigma_y_inv = np.linalg.inv(sigma_y)
    
    return sigma_c, sigma_y, sigma_y_inv

def gen_random_X(Mt, pwr_norm = True):
    """
    generate random transmitted signal X
    
    Input:
    ------
        Mt: int, number of transmit antennas
        pwr_norm: bool, whether to normalize the power of generated X
        
    Output:
    ------
        X: numpy array of shape (Mt, 1) and type np.complex128, generated power normalized X
    """
    X = np.random.normal(0, 1, (Mt, 1)) + 1j * np.random.normal(0, 1, (Mt, 1))
    if pwr_norm:
        X /= np.sqrt(np.sum(abs(X.reshape(-1,))**2))
    return X

def gen_Y(Hc, X, hp, pwr_n, Hp_error_percent = 0.0, print_pwr = False):
    """
    generate received signal Y
    
    Input:
    ------
        Hc: numpy array of shape (Mr, Mt) and type np.complex128, ground-truth clutter channel
        X: numpy array of shape (Mt, 1) and type np.complex128, input signal X
        hp: float of type np.complex128, ground-truth precipitation channel to be estimated
        pwr_n: float, input noise power
        Hp_error_percent: float, power percentage in Hp 
        print_pwr: bool, whether to print power of precipitation, clutter, and noise
        
    Output:
    ------
        Y: numpy array of shape (Mr, 1) and type np.complex128, output signal Y
        pwr_dict: dict, dict of power file
    """
    # sanity check
    (Mr, Mt) = Hc.shape
    assert X.shape == (Mt, 1), "Mismatch in X.shape: {} with Hc.shape: {}".format(X.shape, Hc.shape)

    # generate Hp (with potential error)
    Hp = hp * np.ones((Mr, Mt), dtype = np.complex128)
    Hp_error = np.random.normal(0, 1, Hp.shape) + 1j * np.random.normal(0, 1, Hp.shape)
    Hp_error *= np.sqrt(Hp_error_percent) * np.sqrt(np.sum(abs(Hp.reshape(-1,))**2)) / np.sqrt(np.sum(abs(Hp_error.reshape(-1,))**2))
    Hp = Hp + Hp_error

    # generate signal 
    signal = (Hp + Hc) @ X
    
    # generate noise 
    noise = np.random.normal(0, 1, (Mr, 1)) + 1j * np.random.normal(0, 1, (Mr, 1))
    noise /= np.sqrt(np.sum(abs(noise.reshape(-1,))**2))
    noise *= np.sqrt(pwr_n)

    # generate received signal by combining both signal part and noise part
    Y = signal + noise

    # power analysis
    pwr_dict = {}
    pwr_p = np.sqrt(np.sum(abs(Hp @ X)**2))    
    pwr_c = np.sqrt(np.sum(abs(Hc @ X)**2))
    pwr_dict['pwr_p_lin'] = pwr_p
    pwr_dict['pwr_c_lin'] = pwr_c
    pwr_dict['pwr_n_lin'] = pwr_n
    pwr_dict['pwr_p_dB'] = lin2dB(pwr_p)
    pwr_dict['pwr_c_dB'] = lin2dB(pwr_c)
    pwr_dict['pwr_n_dB'] = lin2dB(pwr_n)
    if print_pwr:
        print("\nprecipitation power: {:.2f} dB\nclutter power: {:.2f} dB\nnoise power: {:.2f} dB\n".format(pwr_p_dB, pwr_c_dB, pwr_n_dB))

    return Y, pwr_dict

def calc_LS_Prx(sigma_y_inv, X):
    """
    RX beamform: calculate (analytically) the LS receiver side beamformer Prx
    
    Input:
    ------
        sigma_y_inv: numpy array of shape (Mr, Mr) and type float, inverse of covariance matrix for received signal
        X: numpy array of shape (Mt, 1) and type np.complex128, input signal X

    Output:
    ------
        Prx: numpy array of shape (Mr, 1) and type np.complex128, LS receiver side beamformer
    """
    # sanity check
    Mr = sigma_y_inv.shape[0]
    assert sigma_y_inv.shape == (Mr, Mr), "Wrong sigma_y_inv.shape: {}".format(sigma_y_inv.shape)
    
    # calculate the LS beamformer
    vec_ones = np.ones((Mr, 1))
    Prx = (vec_ones.conj().T @ sigma_y_inv) / (vec_ones.conj().T @ sigma_y_inv @ vec_ones)[0, 0]
    Prx /= np.sum(X.reshape(-1,))
    Prx = Prx.conj().T
    
    # normalize power
    Prx /= np.sqrt(np.sum(abs(Prx.reshape(-1,))**2))
    
    return Prx
    
def calc_MLE_hp(Y, Hc_mu, X, Prx = None):
    """
    calculate the MLE estimator of hp using receiver side beamforming Prx
    
    Input:
    ------
        Y: numpy array of shape (Mr, 1) and type np.complex128, output signal Y
        Hc_mu: numpy array of shape (Mr, Mt) and type np.complex128, estimated mean clutter channel
        X: numpy array of shape (Mt, 1) and type np.complex128, input signal X
        Prx: None or numpy array of shape (Mr, 1) and type np.complex128, receive side beamformer

    Output:
    ------
        hp_MLE: float of type np.complex128, estimated scalar precipitation channel
    """
    (Mr, Mt) = Hc_mu.shape
    assert X.shape == (Mt, 1), "Mismatch in X.shape: {} with Hc_mu.shape: {}".format(X.shape, Hc_mu.shape)
    assert Y.shape == (Mr, 1), "Mismatch in Y.shape: {} with Hc_mu.shape: {}".format(Y.shape, Hc_mu.shape)
    assert isinstance(Prx, type(None)) or Prx.shape == (Mr, 1), "Mismatch in Prx.shape: {} with Hc_mu.shape: {}".format(Prx.shape, Hc_mu.shape)

    if isinstance(Prx, type(None)):
        Prx = np.ones((Mr, 1))
        Prx /= np.sqrt(np.sum(abs(Prx.reshape(-1,))**2))
               
    hp_MLE = (Prx.conj().T @ (Y - Hc_mu @ X))[0, 0]
    hp_MLE /= (np.sum(X) * (Prx.conj().T @ np.ones((Mr, 1)))[0, 0])

    return hp_MLE

def calc_var_LS(X, UR, omega, UT, pwr_n):
    """
    calculate theoretical variance of hp with least square RX beamforming
    
    Input:
    ------
        UR: numpy array of shape (Mr, Mr) and type np.complex128, Unitary matrix as RX side eigenbases
        omega: numpy array of shape (Mr, Mt) and type positive np.float64, power coupling matrix between TX and RX side eigenbases
        UT: numpy array of shape (Mt, Mt) and type np.complex128, Unitary matrix as TX side eigenbases
        X: numpy array of shape (Mt, 1) and type np.complex128, power normalized X
        pwr_n: float, input noise power

    Output:
    ------
        var: float, calculated theoretical variance
    """   
    # sanity check
    (Mr, Mt) = omega.shape
    assert UR.shape == (Mr, Mr), "Wrong UR.shape: {}".format(UR.shape)
    assert UT.shape == (Mt, Mt), "Wrong UT.shape: {}".format(UT.shape)

    # normalize power of X
    X /= np.sqrt(np.sum(abs(X.reshape(-1,))**2))

    sigma_c, sigma_y, sigma_y_inv = calc_sigma(UR = UR, omega = omega, UT = UT, X = X, pwr_n = pwr_n)
    
    denom1 = abs(np.sum(X.reshape(-1,)))**2
    denom2 = abs(np.ones((1, Mr)) @ sigma_y_inv @ np.ones((Mr, 1)))[0, 0]
    var = 1.0 / (denom1 * denom2)
    
    return var

def calc_var_Prx(X, UR, omega, UT, pwr_n, Prx):
    """
    calculate theoretical variance of hp with any RX beamforming
    
    Input:
    ------
        UR: numpy array of shape (Mr, Mr) and type np.complex128, Unitary matrix as RX side eigenbases
        omega: numpy array of shape (Mr, Mt) and type positive np.float64, power coupling matrix between TX and RX side eigenbases
        UT: numpy array of shape (Mt, Mt) and type np.complex128, Unitary matrix as TX side eigenbases
        X: numpy array of shape (Mt, 1) and type np.complex128, power normalized X
        pwr_n: float, input noise power
        Prx: None or numpy array of shape (Mr, 1) and type np.complex128, receive side beamformer

    Output:
    ------
        var: float, calculated theoretical variance
    """
    # sanity check
    (Mr, Mt) = omega.shape
    assert UR.shape == (Mr, Mr), "Wrong UR.shape: {}".format(UR.shape)
    assert UT.shape == (Mt, Mt), "Wrong UT.shape: {}".format(UT.shape)
    assert Prx.shape == (Mr, 1), "Wrong Prx.shape: {}".format(Prx.shape)

    # normalize power of X
    X /= np.sqrt(np.sum(abs(X.reshape(-1,))**2))

    sigma_c, sigma_y, sigma_y_inv = calc_sigma(UR = UR, omega = omega, UT = UT, X = X, pwr_n = pwr_n)

    nom = abs(Prx.conj().T @ sigma_y @ Prx)[0, 0]
    denom = abs((Prx.conj().T @ np.ones((Mr, 1)))[0, 0] * np.sum(X))**2
    var = nom / denom

    return var

def calc_emp_MSE_Prx(hp, UR, UT, omega, pwr_n, X, Prx, Hp_error_percent = 0.0, N_chan_real = 100000):
    """
    calculate the empirical normalized MSE
    
    Input:
    ------
        hp: float of type np.complex128, ground-truth precipitation channel to be estimated
        UR: numpy array of shape (Mr, Mr) and type np.complex128, ground-truth Unitary matrix as RX side eigenbases
        UT: numpy array of shape (Mt, Mt) and type np.complex128, ground-truth Unitary matrix as TX side eigenbases
        omega: numpy array of shape (Mr, Mt) and type positive np.float64, ground-truth power coupling matrix between TX and RX side eigenbases
        pwr_n: float, ground-truth noise power
        X: numpy array of shape (Mt, 1) and type np.complex128, input signal X
        Prx: None or numpy array of shape (Mr, 1) and type np.complex128, receive side beamformer
        Hp_error_percent: float, power percentage in Hp 
        N_chan_real: int, number of random realizations for MSE calculation

    Output:
    ------
        MSE_RXBF: float, mean squared error with given Prx in RX beamforming
    """
    # sanity check
    (Mr, Mt) = omega.shape
    assert UR.shape == (Mr, Mr), "Wrong UR.shape: {}".format(UR.shape)
    assert UT.shape == (Mt, Mt), "Wrong UT.shape: {}".format(UT.shape)
    assert Prx.shape == (Mr, 1), "Wrong Prx.shape: {}".format(Prx.shape)

    error_square_RXBF = []
    for idx in range(N_chan_real):
        # simulate the Hwei channel and received signal using ground truth Weichselberger model parameters
        Hwei = gen_wei_chan(UR = UR, UT = UT, omega = omega)
        Hc_mu = np.zeros(Hwei.shape)
        Y_rand, _ = gen_Y(Hc = Hwei, X = X, hp = hp, pwr_n = pwr_n, Hp_error_percent = Hp_error_percent, print_pwr = False)
        hp_RXBF = calc_MLE_hp(Y = Y_rand, Hc_mu = Hc_mu, X = X, Prx = Prx)

        # calculate and append estimation error power
        err_sq_RXBF = abs(hp_RXBF - hp)**2
        error_square_RXBF.append(err_sq_RXBF)

    MSE_RXBF = np.mean(error_square_RXBF)

    return MSE_RXBF

##############################################################################################################
#
# numerical solution
#
##############################################################################################################
def unit_pwr_cons(X):
    """
    unit power constraint for X
    
    Input:
    ------
        X: numpy array of shape (Mt, 1) and type np.complex128, physically transmitted signal X
        
    Output:
    ------
        constraint_met: bool, 0 if constraint is met
    """
    constraint_met = (np.sum(abs(X.reshape(-1,))**2) - 1.0)
    return constraint_met
    
##############################################################################################################
#
# type 1 sub-optimal solution (eigen space of transmitted signal)
#
##############################################################################################################
def calc_eigen_opt_X(UR, omega, UT, pwr_n):
    """
    calculate the optimal eigen transmitted signal X
    
    Input:
    ------
        UR: numpy array of shape (Mr, Mr) and type np.complex128, Unitary matrix as RX side eigenbases
        omega: numpy array of shape (Mr, Mt) and type positive np.float64, power coupling matrix between TX and RX side eigenbases
        UT: numpy array of shape (Mt, Mt) and type np.complex128, Unitary matrix as TX side eigenbases
        pwr_n: float, input noise power

    Output:
    ------
        X_eigen_opt: numpy array of shape (Mt, 1) and type np.complex128, power normalized optimal eigen transmitted signal X
    """
    # sanity check
    (Mr, Mt) = omega.shape
    assert UR.shape == (Mr, Mr), "Wrong UR.shape: {}".format(UR.shape)
    assert UT.shape == (Mt, Mt), "Wrong UT.shape: {}".format(UT.shape)
    var_list = []
    
    for col_idx in range(Mt):
        X_eigen = UT[:, col_idx].reshape(-1, 1)
        X_eigen /= np.sqrt(np.sum(abs(X_eigen.reshape(-1,))**2))
        var = calc_var_LS(X = X_eigen, UR = UR, omega = omega, UT = UT, pwr_n = pwr_n)
        var_list.append(var)
        
    min_col_idx = np.argmin(np.array(var_list))
    X_eigen_opt = UT[:, min_col_idx].reshape(-1, 1)
    X_eigen_opt /= np.sqrt(np.sum(abs(X_eigen_opt.reshape(-1,))**2))
    
    return X_eigen_opt
        
##############################################################################################################
#
# type 2 sub-optimal solution (fourier TX beamformers)
#
##############################################################################################################
def calc_fourier_X(theta_deg, phi_deg, Mt_row, Mt_col, ant_space_wl = 0.5):
    """
    calculate a fourier TX beamformer
    
    Input:
    ------
        theta_deg: float, elevation angle measured in degrees
        phi_deg: float, azimuth angle measured in degrees
        Mt_row: int, number of TX antenna rows
        Mt_col: int, number of TX antenna columns
        ant_space_wl: float, antenna spacing normalized by wavelength
        
    Output:
    ------
        X_fourier: numpy array of shape (Mt_row * Mt_col, 1) and type np.complex128, power normalized optimal X
    """
    theta_rad, phi_rad = deg2rad(theta_deg), deg2rad(phi_deg)
    steering_row = np.exp(1j * 2 * np.pi * np.cos(theta_rad) * ant_space_wl * np.arange(Mt_row))
    steering_col = np.exp(1j * 2 * np.pi * np.sin(theta_rad) * np.cos(phi_rad) * ant_space_wl * np.arange(Mt_col))
    steering_vec = steering_row.reshape(-1, 1) @ steering_col.reshape(1, -1)
    X_fourier = steering_vec.reshape(-1, 1)
    X_fourier /= np.sqrt(np.sum(abs(X_fourier.reshape(-1,))**2))
    
    return X_fourier

def calc_fourier_opt_X(UR, omega, UT, pwr_n, Mt_row, Mt_col, ant_space_wl = 0.5, theta_search_num = 9, phi_search_num = 9):
    """
    calculate the optimal transmitted fourier signal X for the LS receiver side beamformer
    
    Input:
    ------
        UR: numpy array of shape (Mr, Mr) and type np.complex128, Unitary matrix as RX side eigenbases
        omega: numpy array of shape (Mr, Mt) and type positive np.float64, power coupling matrix between TX and RX side eigenbases
        UT: numpy array of shape (Mt, Mt) and type np.complex128, Unitary matrix as TX side eigenbases
        pwr_n: float, input noise power
        Mt_row: int, number of TX antenna rows
        Mt_col: int, number of TX antenna columns
        ant_space_wl: float, antenna spacing normalized by wavelength
        theta_search_num: int, number of searched theta angles from [0, 180] deg
        phi_search_num: int, number of searched phi angles from [0, 360] deg

    Output:
    ------
        X_fourier_opt: numpy array of shape (Mt, 1) and type np.complex128, power normalized optimal transmitted fourier signal X
    """
    # sanity check
    (Mr, Mt) = omega.shape
    assert UR.shape == (Mr, Mr), "Wrong UR.shape: {}".format(UR.shape)
    assert UT.shape == (Mt, Mt), "Wrong UT.shape: {}".format(UT.shape)
    assert (Mt_row * Mt_col) == Mt, "Wrong Mt setup with Mt_col: {}, Mt_col: {}, Mt: {}".format(Mt_row, Mt_col, Mt)
    var_list = []
    
    for theta_idx in range(theta_search_num):
        for phi_idx in range(phi_search_num):
            theta_deg = theta_idx * (180.0 / theta_search_num)
            phi_deg = phi_idx * (360.0 / phi_search_num)
            X_fourier = calc_fourier_X(theta_deg = theta_deg, 
                                       phi_deg = phi_deg, 
                                       Mt_row = Mt_row, 
                                       Mt_col = Mt_col, 
                                       ant_space_wl = ant_space_wl)
            var = calc_var_LS(X = X_fourier, UR = UR, omega = omega, UT = UT, pwr_n = pwr_n)
            var_list.append((X_fourier, var))
        
    var_list.sort(key = lambda x: x[1])
    X_fourier_opt = var_list[0][0]
    
    return X_fourier_opt

def calc_fourier_opt_X_Prx(Prx, UR, omega, UT, pwr_n, Mt_row, Mt_col, ant_space_wl = 0.5, theta_search_num = 18, phi_search_num = 18):
    """
    calculate the optimal transmitted fourier signal X for fixed Prx
    
    Input:
    ------
        Prx: numpy array of shape (Mr, 1) and type np.complex128, receive side beamformer
        UR: numpy array of shape (Mr, Mr) and type np.complex128, Unitary matrix as RX side eigenbases
        omega: numpy array of shape (Mr, Mt) and type positive np.float64, power coupling matrix between TX and RX side eigenbases
        UT: numpy array of shape (Mt, Mt) and type np.complex128, Unitary matrix as TX side eigenbases
        pwr_n: float, input noise power
        Mt_row: int, number of TX antenna rows
        Mt_col: int, number of TX antenna columns
        ant_space_wl: float, antenna spacing normalized by wavelength
        theta_search_num: int, number of searched theta angles from [0, 180] deg
        phi_search_num: int, number of searched phi angles from [0, 360] deg

    Output:
    ------
        X_fourier_opt: numpy array of shape (Mt, 1) and type np.complex128, power normalized optimal transmitted fourier signal X
    """
    # sanity check
    (Mr, Mt) = omega.shape
    assert UR.shape == (Mr, Mr), "Wrong UR.shape: {}".format(UR.shape)
    assert UT.shape == (Mt, Mt), "Wrong UT.shape: {}".format(UT.shape)
    assert (Mt_row * Mt_col) == Mt, "Wrong Mt setup with Mt_col: {}, Mt_col: {}, Mt: {}".format(Mt_row, Mt_col, Mt)
    var_list = []
    
    for theta_idx in range(theta_search_num):
        for phi_idx in range(phi_search_num):
            theta_deg = theta_idx * (180.0 / theta_search_num)
            phi_deg = phi_idx * (360.0 / phi_search_num)
            X_fourier = calc_fourier_X(theta_deg = theta_deg, 
                                       phi_deg = phi_deg, 
                                       Mt_row = Mt_row, 
                                       Mt_col = Mt_col, 
                                       ant_space_wl = ant_space_wl)
            var = calc_var_Prx(X = X_fourier, UR = UR, omega = omega, UT = UT, pwr_n = pwr_n, Prx = Prx)
            var_list.append((X_fourier, var))
        
    var_list.sort(key = lambda x: x[1])
    X_fourier_opt = var_list[0][0]
    
    return X_fourier_opt

def plot_omega(omega):
    """
    Plot the distribution of omega

    Input:
    ------
        omega: numpy array of shape (Mr, Mt) and type positive np.float64, ground truth power coupling matrix between TX and RX side eigenbases

    Output:
    ------
        None
    """
    omega_1D = sorted(omega.reshape(-1,), reverse = True)
    plt.plot(omega_1D, '.-')
    plt.ylabel('omega')
    plt.show()

def Simulator(N_real, Mt_row, Mt_col, Mr_row, Mr_col, omega_min_dB, omega_max_dB, pwr_n_dB, hp_pwr_dB, theta_search_num = 9, phi_search_num = 9, UR_est_error_percent = 0.0, UT_est_error_percent = 0.0, Hp_error_percent = 0.0, omega_est_error_percent = 0.0, pwr_n_est_error_percent = 0.0, theo_MSE = True, N_chan_real = 1000, UR = None, UT = None, omega = None, reuse_fourier_TX = False, reuse_opt_TX = False):
    """
    Simulator to calculate the theoretical / empirical MSE from a set of beamformers

    Input:
    ------
        N_real: int, number of realizations for averaging
        Mt_row: int, number of rows in transmit antenna array
        Mt_col: int, number of columns in transmit antenna array
        Mr_row: int, number of rows in receive antenna array
        Mr_col: int, number of columns in receive antenna array
        omega_min_dB: float, minimal value of power coupling matrix element measured in dB scale
        omega_max_dB: float, maximum value of power coupling matrix element measured in dB scale
        pwr_n_dB: float, noise power in dB scale
        hp_pwr_dB: float, precipitation power in dB scale
        theta_search_num: int, number of searched theta angles from [0, 180] deg
        phi_search_num: int, number of searched phi angles from [0, 360] deg
        UR_est_error_percent: float, power percentage of error in UR estimation
        UT_est_error_percent: float, power percentage of error in UT estimation
        Hp_error_percent: float, power percentage in Hp 
        omega_est_error_percent: float, power percentage of error in omega estimation
        pwr_n_est_error_percent: float, power percentage of error in pwr_n estimation
        theo_MSE: bool, indicating whether to use theoretical or empirical MSE
        N_chan_real: int, number of channel realizations in empirical MSE calculation
        UR: None or numpy array of shape (Mr, Mr) and type np.complex128, ground truth Unitary matrix as RX side eigenbases
        UT: None or numpy array of shape (Mt, Mt) and type np.complex128, ground truth Unitary matrix as TX side eigenbases
        omega: None or numpy array of shape (Mr, Mt) and type positive np.float64, ground truth power coupling matrix between TX and RX side eigenbases
        reuse_fourier_TX: bool, whether to reuse the fourier TX beamformer from 1st run
        reuse_opt_TX: bool, whether to reuse the optimal TX beamformer from 1st run
    
    Output:
    ------
        time_dict: dict, time dict
        pwr_dB_dict: dict, power dict
        mse_dict: dict, mse dict
        std_dict: dict, standard deviation dict
    """
    Mt = Mt_row * Mt_col
    Mr = Mr_row * Mr_col
    pwr_n = dB2lin(pwr_n_dB)

    pwr_record = defaultdict(lambda: np.zeros(N_real))
    mse_record = defaultdict(lambda: np.zeros(N_real))
    time_record = defaultdict(lambda: np.zeros(N_real))

    for idx_real in range(N_real):
        hp = np.random.normal(0, 1) + 1j * np.random.normal(0, 1)
        hp *= (np.sqrt(dB2lin(hp_pwr_dB) / abs(hp)**2))

        # Weichselberger model parameters
        if isinstance(UR, type(None)) and isinstance(UT, type(None)) and isinstance(omega, type(None)):
            UR = unitary_group.rvs(Mr)
            UT = unitary_group.rvs(Mt)
            omega_min_lin, omega_max_lin = dB2lin(omega_min_dB), dB2lin(omega_max_dB)
            omega = np.random.uniform(omega_min_lin, omega_max_lin, Mt*Mr).reshape(Mr, Mt)
        else:
            assert UR.shape == (Mr, Mr), "Wrong UR.shape: {}".format(UR.shape)
            assert UT.shape == (Mt, Mt), "Wrong UT.shape: {}".format(UT.shape)
            assert omega.shape == (Mr, Mt), "Wrong omega.shape: {}".format(omega.shape)

        # assumes perfect knowledge of UR, UT, and pwr_n
        # reason: it's hard to assume imperfect knowledge of UR such that the error'ed version UR_est is still Unitary
        UR_est = UR
        UT_est = UT 
        pwr_n_est = pwr_n

        # assumes imperfect knowledge of omega parameters
        omega_error = np.random.normal(0, 1, omega.shape)
        omega_error *= np.sqrt(omega_est_error_percent) * np.sqrt(np.sum(abs(omega.reshape(-1,))**2)) / np.sqrt(np.sum(abs(omega_error.reshape(-1,))**2))
        omega_est = omega + omega_error

        # make sure that each element in omega is still non-negative
        omega_est = np.maximum(omega_est, 0.0)

        # random TX beamforming
        X_rand = gen_random_X(Mt = Mt, pwr_norm = True)

        # fourier TX beamforming solution for LS RX beamformer
        time_fourier_st = time.time()

        if reuse_fourier_TX and (idx_real > 0):
            # reuse previously calculated X_fourier_LS
            pass
        else:
            # calculate X_fourier_LS
            X_fourier_LS = calc_fourier_opt_X(UR = UR_est, 
                                              omega = omega_est, 
                                              UT = UT_est, 
                                              pwr_n = pwr_n_est, 
                                              Mt_row = Mt_row, 
                                              Mt_col = Mt_col,
                                              theta_search_num = theta_search_num, 
                                              phi_search_num = phi_search_num)

        time_fourier_ed = time.time()

        # numerically optimal TX beamforming solution for LS RX beamformer
        cons = {'type': 'eq', 'fun': unit_pwr_cons}
        time_opt_st = time.time()
        if reuse_opt_TX and (idx_real > 0):
            # reuse previously calculated X_fourier_LS
            pass
        else:
            TXBF_opt_LS = minimize(fun = calc_var_LS, x0 = X_rand, args = (UR_est, omega_est, UT_est, pwr_n_est,), method = 'SLSQP', constraints = cons)
        time_opt_ed = time.time()
        X_opt_LS = TXBF_opt_LS.x.reshape(-1,1)

        # calculate corresponding LS Prx for each TX beamformer
        sigma_c_rand, sigma_y_rand, sigma_y_inv_rand = calc_sigma(UR = UR_est, 
                                                                  omega = omega_est, 
                                                                  UT = UT_est, 
                                                                  X = X_rand, 
                                                                  pwr_n = pwr_n_est)
        Prx_LS_rand = calc_LS_Prx(sigma_y_inv = sigma_y_inv_rand, X = X_rand)

        sigma_c_fourier, sigma_y_fourier, sigma_y_inv_fourier = calc_sigma(UR = UR_est, 
                                                                           omega = omega_est, 
                                                                           UT = UT_est, 
                                                                           X = X_fourier_LS, 
                                                                           pwr_n = pwr_n_est)
        Prx_LS_fourier = calc_LS_Prx(sigma_y_inv = sigma_y_inv_fourier, X = X_fourier_LS)

        sigma_c_opt, sigma_y_opt, sigma_y_inv_opt = calc_sigma(UR = UR_est, 
                                                               omega = omega_est, 
                                                               UT = UT_est, 
                                                               X = X_opt_LS, 
                                                               pwr_n = pwr_n_est)
        Prx_LS_opt = calc_LS_Prx(sigma_y_inv = sigma_y_inv_opt, X = X_opt_LS)

        # calculate Uniform Magnitude (UM) RX beamforming solution, which does not depend on the 
        Prx_UM = np.sqrt(1.0 / Mr) * np.ones((Mr, 1), dtype = np.complex128)

        # # fourier TX beamforming solution for UM RX beamformer
        # X_fourier_UM = calc_fourier_opt_X_Prx(Prx = Prx_UM,
        #                                       UR = UR_est, 
        #                                       omega = omega_est, 
        #                                       UT = UT_est, 
        #                                       pwr_n = pwr_n_est, 
        #                                       Mt_row = Mt_row, 
        #                                       Mt_col = Mt_col,
        #                                       theta_search_num = theta_search_num, 
        #                                       phi_search_num = phi_search_num)

        # # numerically optimal TX beamforming solution for UM RX beamformer
        # cons = {'type': 'eq', 'fun': unit_pwr_cons}
        # TXBF_opt_UM = minimize(fun = calc_var_Prx, x0 = X_rand, args = (UR_est, omega_est, UT_est, pwr_n_est, Prx_UM), method = 'SLSQP', constraints = cons)
        # X_opt_UM = TXBF_opt_UM.x.reshape(-1,1)

        # calculate time consumption for each beamformer
        time_fourier = time_fourier_ed - time_fourier_st
        time_opt = time_opt_ed - time_opt_st

        # evaluate the power of received signal
        Hwei = gen_wei_chan(UR = UR, UT = UT, omega = omega)
        _, pwr_dict_rand = gen_Y(Hc = Hwei, X = X_rand, hp = hp, pwr_n = pwr_n, print_pwr = False)
        _, pwr_dict_fourier_LS = gen_Y(Hc = Hwei, X = X_fourier_LS, hp = hp, pwr_n = pwr_n, print_pwr = False)
        _, pwr_dict_opt_LS = gen_Y(Hc = Hwei, X = X_opt_LS, hp = hp, pwr_n = pwr_n, print_pwr = False)
        # _, pwr_dict_fourier_UM = gen_Y(Hc = Hwei, X = X_fourier_UM, hp = hp, pwr_n = pwr_n, print_pwr = False)
        # _, pwr_dict_opt_UM = gen_Y(Hc = Hwei, X = X_opt_UM, hp = hp, pwr_n = pwr_n, print_pwr = False)

        if theo_MSE:
            # calculate theoretical variance for UM RX beamformer
            mse_UMRXBF_rand = calc_var_Prx(X = X_rand, UR = UR_est, omega = omega_est, UT = UT_est, pwr_n = pwr_n_est, Prx = Prx_UM)
            # mse_UMRXBF_fourier = calc_var_Prx(X = X_fourier_UM, UR = UR_est, omega = omega_est, UT = UT_est, pwr_n = pwr_n_est, Prx = Prx_UM)
            # mse_UMRXBF_opt = calc_var_Prx(X = X_opt_UM, UR = UR_est, omega = omega_est, UT = UT_est, pwr_n = pwr_n_est, Prx = Prx_UM)

            # calculate theoretical variance for LS RX beamformer
            mse_LSRXBF_rand = calc_var_Prx(X = X_rand, UR = UR_est, omega = omega_est, UT = UT_est, pwr_n = pwr_n_est, Prx = Prx_LS_rand)
            mse_LSRXBF_fourier = calc_var_Prx(X = X_fourier_LS, UR = UR_est, omega = omega_est, UT = UT_est, pwr_n = pwr_n_est, Prx = Prx_LS_fourier)
            mse_LSRXBF_opt = calc_var_Prx(X = X_opt_LS, UR = UR_est, omega = omega_est, UT = UT_est, pwr_n = pwr_n_est, Prx = Prx_LS_opt)
        else:
            # calculate empirical variance
            mse_UMRXBF_rand = calc_emp_MSE_Prx(hp = hp, UR = UR, UT = UT, omega = omega, pwr_n = pwr_n, X = X_rand, Prx = Prx_UM, Hp_error_percent = Hp_error_percent, N_chan_real = N_chan_real)
            # mse_UMRXBF_fourier = calc_emp_MSE_Prx(hp = hp, UR = UR, UT = UT, omega = omega, pwr_n = pwr_n, X = X_fourier_UM, Prx = Prx_UM, Hp_error_percent = Hp_error_percent, N_chan_real = N_chan_real)
            # mse_UMRXBF_opt = calc_emp_MSE_Prx(hp = hp, UR = UR, UT = UT, omega = omega, pwr_n = pwr_n, X = X_opt_UM, Prx = Prx_UM, Hp_error_percent = Hp_error_percent, N_chan_real = N_chan_real)

            mse_LSRXBF_rand = calc_emp_MSE_Prx(hp = hp, UR = UR, UT = UT, omega = omega, pwr_n = pwr_n, X = X_rand, Prx = Prx_LS_rand, Hp_error_percent = Hp_error_percent, N_chan_real = N_chan_real)
            mse_LSRXBF_fourier = calc_emp_MSE_Prx(hp = hp, UR = UR, UT = UT, omega = omega, pwr_n = pwr_n, X = X_fourier_LS, Prx = Prx_LS_fourier, Hp_error_percent = Hp_error_percent, N_chan_real = N_chan_real)
            mse_LSRXBF_opt = calc_emp_MSE_Prx(hp = hp, UR = UR, UT = UT, omega = omega, pwr_n = pwr_n, X = X_opt_LS, Prx = Prx_LS_opt, Hp_error_percent = Hp_error_percent, N_chan_real = N_chan_real)

        mse_UMRXBF_rand_norm = mse_UMRXBF_rand / abs(hp)**2
        # mse_UMRXBF_fourier_norm = mse_UMRXBF_fourier / abs(hp)**2
        # mse_UMRXBF_opt_norm = mse_UMRXBF_opt / abs(hp)**2

        mse_LSRXBF_rand_norm = mse_LSRXBF_rand / abs(hp)**2
        mse_LSRXBF_fourier_norm = mse_LSRXBF_fourier / abs(hp)**2
        mse_LSRXBF_opt_norm = mse_LSRXBF_opt / abs(hp)**2

        # record time for each realization
        time_record['fourier_time'][idx_real] = time_fourier
        time_record['opt_time'][idx_real] = time_opt

        # record power for each realization
        pwr_record['rand_pwr_p_lin'][idx_real] = pwr_dict_rand['pwr_p_lin']
        pwr_record['rand_pwr_c_lin'][idx_real] = pwr_dict_rand['pwr_c_lin']
        pwr_record['rand_pwr_n_lin'][idx_real] = pwr_dict_rand['pwr_n_lin']

        # pwr_record['fourier_UM_pwr_p_lin'][idx_real] = pwr_dict_fourier_UM['pwr_p_lin']
        # pwr_record['fourier_UM_pwr_c_lin'][idx_real] = pwr_dict_fourier_UM['pwr_c_lin']
        # pwr_record['fourier_UM_pwr_n_lin'][idx_real] = pwr_dict_fourier_UM['pwr_n_lin']

        # pwr_record['opt_UM_pwr_p_lin'][idx_real] = pwr_dict_opt_UM['pwr_p_lin']
        # pwr_record['opt_UM_pwr_c_lin'][idx_real] = pwr_dict_opt_UM['pwr_c_lin']
        # pwr_record['opt_UM_pwr_n_lin'][idx_real] = pwr_dict_opt_UM['pwr_n_lin']

        pwr_record['fourier_LS_pwr_p_lin'][idx_real] = pwr_dict_fourier_LS['pwr_p_lin']
        pwr_record['fourier_LS_pwr_c_lin'][idx_real] = pwr_dict_fourier_LS['pwr_c_lin']
        pwr_record['fourier_LS_pwr_n_lin'][idx_real] = pwr_dict_fourier_LS['pwr_n_lin']

        pwr_record['opt_LS_pwr_p_lin'][idx_real] = pwr_dict_opt_LS['pwr_p_lin']
        pwr_record['opt_LS_pwr_c_lin'][idx_real] = pwr_dict_opt_LS['pwr_c_lin']
        pwr_record['opt_LS_pwr_n_lin'][idx_real] = pwr_dict_opt_LS['pwr_n_lin']

        # record channel power
        pwr_record['Hc_channel_pwr_lin'][idx_real] = np.sum(abs(Hwei.reshape(-1,))**2)
        pwr_record['Hp_channel_pwr_lin'][idx_real] = np.sum(abs(hp * np.ones(Mr*Mt))**2)

        # record MSE for each realization
        mse_record['UMRXBF_rand'][idx_real] = mse_UMRXBF_rand
        # mse_record['UMRXBF_fourier'][idx_real] = mse_UMRXBF_fourier
        # mse_record['UMRXBF_opt'][idx_real] = mse_UMRXBF_opt

        mse_record['LSRXBF_rand'][idx_real] = mse_LSRXBF_rand
        mse_record['LSRXBF_fourier'][idx_real] = mse_LSRXBF_fourier
        mse_record['LSRXBF_opt'][idx_real] = mse_LSRXBF_opt

        # record normalized MSE for each realization
        mse_record['UMRXBF_rand_norm'][idx_real] = mse_UMRXBF_rand_norm
        # mse_record['UMRXBF_fourier_norm'][idx_real] = mse_UMRXBF_fourier_norm
        # mse_record['UMRXBF_opt_norm'][idx_real] = mse_UMRXBF_opt_norm

        mse_record['LSRXBF_rand_norm'][idx_real] = mse_LSRXBF_rand_norm
        mse_record['LSRXBF_fourier_norm'][idx_real] = mse_LSRXBF_fourier_norm
        mse_record['LSRXBF_opt_norm'][idx_real] = mse_LSRXBF_opt_norm

        if idx_real % 100 == -1:
            print("realization {} simulation finished".format(idx_real + 1))

    # compute average
    time_dict = {}
    pwr_dB_dict = {}
    mse_dict = {}
    lo_95_dict = {}
    up_95_dict = {}

    for key in time_record:
        time_dict[key] = np.mean(time_record[key])

    for key in pwr_record:
        pwr_dB_dict[key[:-3] + 'dB'] = lin2dB(np.mean(pwr_record[key]))

    for key in mse_record:
        mse_dict[key] = np.mean(mse_record[key])

        ci = bootstrap((mse_record[key],), np.mean, confidence_level = 0.95, method = 'percentile')
        lower_bound = ci.confidence_interval.low
        upper_bound = ci.confidence_interval.high

        lo_95_dict['lo_95_' + key] = mse_dict[key] - lower_bound
        up_95_dict['up_95_' + key] = upper_bound - mse_dict[key]

    return time_dict, pwr_dB_dict, mse_dict, lo_95_dict, up_95_dict

#########################################################################################
#
# ARCHIVE
#
#########################################################################################
