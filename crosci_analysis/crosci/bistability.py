import numpy as np
from scipy.stats import expon
from scipy.optimize import minimize, differential_evolution
from scipy.signal import argrelextrema
import matplotlib.pyplot as plt
from math import log, e


def get_bistability_index(data, binNum, plot_histogram=False):
    def biexppdf(x, gamma1, gamma2, delta):
    # def biexppdf(x, alpha, gamma1, gamma2, delta):
        eps = 0
        # return np.maximum(eps, delta * gamma1 * np.exp(-gamma1 * (10**alpha) * x) + (1 - delta) * gamma2 * np.exp(-gamma2 * (10**alpha) * x))
        return np.maximum(eps, delta * gamma1 * np.exp(-gamma1 * x) + (1-delta) * gamma2 * np.exp(-gamma2 * x))

    def objective(params):
        gamma1 = np.exp(params[0])
        gamma2 = np.exp(params[1])
        delta = params[2]
        if gamma1 > gamma2:
            return np.inf  # set a high cost

        transformed_params = (gamma1,gamma2,delta)
    
        pdf = biexppdf(data, *transformed_params)

        return -np.sum(np.log(pdf))

        # Define the transformed bi-exponential function
    def biexppdf_log(y, gamma1, gamma2, delta):
    # def biexppdf_log(y, alpha, gamma1, gamma2, delta):
        x = np.exp(y)
        return biexppdf(x, gamma1, gamma2, delta) * x
    
    # x0 = ( 1-np.max(np.log10(data)) , 2, 2, 0.5)

    # print(1-max(np.log10(data)))
    # # Use differential evolution for global optimization

    # parameters of single exponential fit
    param1, scale = expon.fit(data, floc=0)
    gamma0 = 1/scale
    max_gamma = 10*gamma0

    # print(np.log(gamma0))
    x0 = [np.log(gamma0),np.log(gamma0),0.5]
    result = minimize(objective,x0=x0,bounds= [(-20,10), (-20,10), (0, 1)],method='Nelder-Mead'),

    result = result[0]
    bi_exp = result.x
    
    gamma1, gamma2, delta1 = bi_exp
    gamma1 = np.exp(gamma1)
    gamma2 = np.exp(gamma2)

    transformed_params = (gamma1, gamma2, delta1)

    bistab_objective = objective(bi_exp)
    
    counts, bin_edges = np.histogram(np.log(data), bins=binNum, density=True)
    centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # Get the values of the transformed bi-exponential function
    biM_log = biexppdf_log(centers, gamma1, gamma2, delta1)
    biM_log = biM_log / np.sum(biM_log)

    biM_log1x = biexppdf_log(centers, gamma1, 0, delta1)
    biM_log2x = biexppdf_log(centers, 0, gamma2, delta1)
    biM_log1 = biM_log1x / np.sum(biM_log1x+biM_log2x)
    biM_log2 = biM_log2x / np.sum(biM_log1x+biM_log2x)

    delta = 1
    gamma1 = 0
    gamma2 = gamma0
    result = delta * gamma1 * np.exp(-gamma1 * centers) + (1-delta) * gamma2 * np.exp(-gamma2 * centers)

    monoM_log = np.exp(centers - 1 / scale * np.exp(centers))
    monoM_log /= np.sum(monoM_log)
    
    monoM_peaks_idx = argrelextrema(monoM_log, np.greater)
    biM_peaks_idx = argrelextrema(biM_log, np.greater)
    # print(biM_peaks_idx)

    x_values = np.linspace(np.log(min(data)), np.log(max(data)), binNum)

    monoM_peak_log_power = np.squeeze(x_values[monoM_peaks_idx])
    monoM_peak_height_norm = np.squeeze(monoM_log[monoM_peaks_idx])

    idx_biM_log1 = np.argmax(biM_log1)
    idx_biM_log2 = np.argmax(biM_log2)
    biM_peaks_idx = np.sort(np.array([idx_biM_log1,idx_biM_log2]))
    # print(biM_peaks_idx)

    biM_peak_log_power = np.squeeze(x_values[biM_peaks_idx])
    biM_peak_height_norm = np.squeeze(biM_log[biM_peaks_idx])

    gamma1, gamma2, delta = transformed_params
    
    normalized_peak1 = biM_log1/np.max(biM_log1)
    normalized_peak2 = biM_log2/np.max(biM_log2)
    

    log_power = np.log(data)

    probabilities = counts / float(counts.sum())

    if plot_histogram:
        
        plt.figure()
        plt.bar(centers, probabilities, width=(centers[1] - centers[0]), alpha=0.6, label="Data")

        # print(centers)
        # print(np.shape(x_values))
        # print(monoM_log)
        # print(biM_log)
        # print(biM_log1)
        # print(biM_log2)
        plt.plot(centers, monoM_log, label='Fitted Mono-exponential Distribution', color='blue')
        plt.plot(x_values, biM_log, label='Fitted Bi-exponential Distribution', color='red')
        plt.plot(x_values, biM_log1, label='Peak1', color='gray')
        plt.plot(x_values, biM_log2, label='Peak2', color='black')

        # plt.legend()
        plt.show()
    
    # BIC values
    loglike_exp = -expon.nnlf((0, scale), data)
    loglike_exp2 = np.sum(np.log(1/scale*np.exp(-   (data/scale)  )))
    loglike_biexp = np.sum(np.log(biexppdf(data, *transformed_params)))
    # print(loglike_exp,loglike_exp2)
    # import pdb
    # pdb.set_trace()
    
    n = len(data)
    k_biexp = 3
    k_exp = 1
    fixed_n = 100000
    BIC_biexp = -2 * (loglike_biexp/n*fixed_n) + k_biexp * np.log(fixed_n)
    BIC_exp = -2 * (loglike_exp/n*fixed_n) + k_exp * np.log(fixed_n)
    # BIC_biexp = -2 * loglike_biexp + k_biexp * np.log(n)
    # BIC_exp = -2 * loglike_exp + k_exp * np.log(n)
    deltaBIC = BIC_exp - BIC_biexp

    #compute the bistability index
    if deltaBIC > 0:
        bistability_index = np.log10(deltaBIC)
        # print("bistable")
    else:
        bistability_index = 0
    
    gamma1, gamma2, delta1 = transformed_params
    return bistability_index,delta1