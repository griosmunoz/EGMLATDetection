"""Help on built-in module signalProcessingFunctions:

NAME
    signalProcessingFunctions

DESCRIPTION
    This module provides functions to process AF signals

CLASSES
    LATSettingsClass
    LATDetectionClass

FUNCTIONS
    calculateUnipolarEGMSlope(egm_signal, m)
        Calculates the m order slope approximation of the EGM signal(s)
        Returns the slope approximation (filtered signal)

    calculateUnipolarEGMSlopeParallel(egm_signal, m)
        Calculates the m order slope approximation of the EGM signal(s)
        For loop is executed in parallel
        Returns the slope approximation (filtered signal)
        
	detectLATs(egm_signal, LATSettings):
		Calculates LATs from unipolar EGM signals
		Returns LATDetection class containing the LAT indices, EGM values at LATs

"""

# signalProcessingFunctions
import numpy as np
import scipy.signal as ss
# from scipy.signal import filtfilt, butter, lfilter
from scipy.signal import medfilt
from scipy.signal import savgol_filter
import configparser # read kivy .ini file
from scipy import stats

from joblib import Parallel, delayed
import multiprocessing

#-------------------------------------------------------------------------------------------------
# CLASSES
#-------------------------------------------------------------------------------------------------
# CLASS LATSettingsClass
class LATSettingsClass:
    """
    A class used to store Local Activation Time (LAT) detection settings

    Attributes
    ----------
    M : int
        Slope filter order
    fs: int
        Sampling frequency
    tau_input : float
        Exponential decay constant
    blank_period_time : float
        Minimum blank period between consecutive local activations in seconds
    T_prev : float
        Time offset to apply LAT detection in seconds
    sigma_abs_th : float
        Minimum activity detection threshold
    LATs_search : int
        Maximum search window for robust LAT detection

    Methods
    -------
    ...(...)
        Calculates...
    """

    def __init__(self, M=10, fs=1000, tau_input=0.00035, blank_period_time=0.090, T_prev=0, sigma_abs_th=0.02, LATs_search=20):

        self.M = M
        self.fs = fs
        self.tau_input = tau_input
        self.blank_period_time = blank_period_time
        self.T_prev = T_prev
        self.sigma_abs_th = sigma_abs_th
        self.LATs_search = LATs_search

# CLASS LATDetectionClass
class LATDetectionClass:
    """
    A class used to store Local Activation Time (LAT) detection results

    Attributes
    ----------
    LATSettings : LATSettingsClass
        LATSettingsClass object containing the LAT settings employed
    activation_peaks_values : (numpy float array)
        EGM value at the LATs
    activation_peaks_indices : (numpy float array)
        Time instant for the LATs
    threshold : (numpy float array)
        Theshold
    abs_threshold : (numpy float array)
        Threshold
    isocro : (numpy float array)
        Elapsed time since last LAT detected

    Methods
    -------
    ...(...)
        Calculates...
    """
    def __init__(self, LATSettings, activation_peaks_values, activation_peaks_indices, threshold, abs_threshold, isocro):

        self.LATSettings = LATSettings
        self.activation_peaks_values = activation_peaks_values
        self.activation_peaks_indices = activation_peaks_indices
        self.threshold = threshold
        self.abs_threshold = abs_threshold
        self.isocro = isocro

#-------------------------------------------------------------------------------------------------
# FUNCTIONS
#-------------------------------------------------------------------------------------------------


# FUNCTION calculateUnipolarEGMSlopeParallel
def calculateUnipolarEGMSlopeParallel(egm_signals, m):
    """
    Function: calculateUnipolarEGMSlopeParallel(egm_signals, m)

    Parameters:
        egm_signals (numpy float array): Bipolar EGM signals. Size [Nu, L] or [L,]
        m (int) : Slope approximation filter order
    Returns:
        beta (numply float array): Slope filter result. Size equals to the egm_signals size
    """
    num_cores = multiprocessing.cpu_count()
    print('Num cores: ' + str(num_cores))

    aux_dim = egm_signals.ndim
    # Check input signal dimensions
    if aux_dim == 1:
        Nu = 1
        L = len(egm_signals)
    else:
        [Nu, L] = egm_signals.shape

    beta_list = Parallel(n_jobs=num_cores, verbose=0)(delayed(calculateUnipolarEGMSlope)(egm_signals[i, :], m) for i in range(Nu))

    beta = np.asarray(beta_list, dtype=type(egm_signals))

    return beta

# FUNCTION calculateUnipolarEGMSlope
def calculateUnipolarEGMSlope(egm_signals, m):
    """
    Function: calculateUnipolarEGMSlope(egm_signals, m)

    Parameters:
        egm_signals (numpy float array): Bipolar EGM signals. Size [Nu, L] or [L,]
        m (int) : Slope approximation filter order
    Returns:
        beta (numply float array): Slope filter result. Size equals to the egm_signals size
    """
    aux_dim = egm_signals.ndim
    # Check input signal dimensions
    if aux_dim == 1:
        Nu = 1
        L = len(egm_signals)
    else:
        [Nu, L] = egm_signals.shape

    # Filter coefficients
    h = np.linspace(-m, m, 2 * m + 1)
    # Constants
    aux_m_range = np.linspace(-m, m, 2*m+1)
    aux_m_range = np.power(aux_m_range, 2)
    aux_B = np.sum(aux_m_range)

    beta = np.zeros(egm_signals.shape)

    for n in range(L):

        aux_min_index = n - m
        aux_max_index = n + m

        # Check indices limits
        if aux_min_index < 0:
            aux_min_index = 0
        if aux_max_index > L-1:
            aux_max_index = L-1
        # Temporal window indices
        aux_indices = np.linspace(aux_min_index, aux_max_index, abs(aux_max_index - aux_min_index) + 1).astype(int)
        aux_h = h[aux_indices + m - n] * (1/aux_B)

        for s in range(Nu):
            # Process all windows
            if Nu == 1:
                aux_signal = egm_signals[aux_indices]
            else:
                aux_signal = egm_signals[s, aux_indices]

            aux_beta = np.multiply(aux_signal, aux_h)
            aux_beta = np.sum(aux_beta)

            if Nu == 1:
                beta[n] = aux_beta
            else:
                beta[s, n] = aux_beta
    return beta

# FUNCTION detectLATs()
def detectLATs(egm_signal, LATSettings):
    """
    Function: calculateUnipolarEGMSlopeParallel(egm_signals, LATSettings)

    Parameters:
        egm_signal (numpy float array): EGM signals. Size [L,]
        LATSettings (LATSettingsClass) : Settings to perform the LAT detection
    Raises:
            ValueError: Signal is not 1D
    Returns:
        LATDetection (LATDetectionClass) : containing the LAT indices, EGM values at LATs
    """
    # LAT Settings
    fs = LATSettings.fs
    tau_input = LATSettings.tau_input
    blank_period_time = LATSettings.blank_period_time
    T_prev = LATSettings.T_prev
    sigma_abs_th = LATSettings.sigma_abs_th
    LATs_search = LATSettings.LATs_search

    # Signal dimensions
    aux_dim = egm_signal.ndim
    # Check input signal dimensions
    if aux_dim == 1:
        L = len(egm_signal)
    else:
        # Raise the error
        aux_error_str = 'ERROR: only 1D signals supported'
        raise ValueError(aux_error_str)

    # Skip first T_prev seconds
    t0 = int(T_prev*fs)
    if t0 < 1:
        t0 = 1
    # End time
    tf = L-2

    # Threshold
    new_Mi = sigma_abs_th
    threshold = np.zeros(egm_signal.shape)
    abs_threshold = np.zeros(egm_signal.shape)
    isocro = np.zeros(egm_signal.shape)
    isocro[0] = -100

    # Exponential decay variable
    tau = tau_input * fs
    # Blank period in samples
    blank_period = blank_period_time * fs
    # Amplitude of the previously detected peak. Initialized as the maximum value in the signal
    Mi = np.max(egm_signal)
    # Local search window span
    local_ti = -blank_period/2
    sigma_t = 0

    # LATs
    activation_peaks_values = np.zeros(egm_signal.shape)
    activation_peaks_indices = np.empty((0,0))

    end_blank_period = 0
    rest_return = 1
    add_lat = 0

    t_indices = np.linspace(t0, tf, tf-t0+1).astype(int)
    new_t = -1

    for i, t in enumerate(t_indices):

        # Blank period
        isocro[t] = isocro[t-1] - 1

        if t > local_ti + blank_period or (t == tf and add_lat == 1) or t < blank_period:

            if end_blank_period:

                add_lat = 0

                activation_peaks_values[new_t] = egm_signal[new_t]
                activation_peaks_indices = np.append(activation_peaks_indices, new_t)

                Mi = new_Mi

                prev_t = np.linspace(new_t, t, t-new_t+1).astype(int)
                threshold[prev_t] = Mi

                # LINEAR ISOCRO UPDATE
                aux_isocro = np.linspace(0, -len(prev_t) + 1, len(prev_t))
                isocro[prev_t] = aux_isocro

                end_blank_period = 0

            # MODIFIED
            sigma_t = sigma_abs_th

            aux_exponent = -(Mi - sigma_t) * (t - (local_ti + blank_period)) / tau

            # aux_print = 'Mi: ' + str(Mi) + ', sigma_t: ' + str(sigma_t) + ', t: ' + str(t) + ', local_ti: ' + str(local_ti) + ', blank_period: ' + str(blank_period) + ', tau: ' + str(tau)
            # print(aux_print)
            # print('Exponent: ' + str(aux_exponent))

            aux_th = (Mi - sigma_t) * np.exp(aux_exponent) + sigma_t

            threshold[t] = aux_th

            if(threshold[t] < sigma_abs_th):
                threshold[t] = sigma_abs_th

            if(egm_signal[t] >= threshold[t] and rest_return):

                # PEAK
                if(egm_signal[t] - egm_signal[t+1] > 0):
                    add_lat = 1

                    local_ti = t
                    end_blank_period = 1

                    new_t = t
                    new_Mi = egm_signal[t]

                    rest_return = 0

        else:

            # SEARCH FOR LOCAL MAX
            threshold[t] = threshold[t-1]

            if t < local_ti + (blank_period / 2):
                local_search = 1
                # New peak
                if ((egm_signal[t] > threshold[t]) and (egm_signal[t] - egm_signal[t-1] > 0) and egm_signal[t] > new_Mi):
                    new_Mi = egm_signal[t]
                    new_t = t

                    # ?????
                    local_ti = t
            else:
                local_search = 0

        # rest_return condition
        if (egm_signal[t] == 0 and rest_return == 0) or (rest_return == 0 and t > local_ti + 1.2 * blank_period):
            rest_return = 1

        abs_threshold[t] = sigma_t

    isocro[L-1] = isocro[L-2] - 1

    LATDetection = LATDetectionClass(LATSettings, activation_peaks_values, activation_peaks_indices, threshold, abs_threshold, isocro)

    return LATDetection
