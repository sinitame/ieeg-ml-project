import numpy as np
from collections import defaultdict
import scipy.signal as sig

def calculate_min(signal):
    return np.min(signal)

def calculate_max(signal):
    return np.max(signal)

def calculate_energy(signal):
    return np.sum(np.power(signal,2))

def calculate_line_length(signal):
    offset_signal = signal[1:].copy()
    return(np.sum(np.abs(offset_signal-signal[:-1])))

def calculate_moving_avg(signal):
    n = len(signal)
    return np.sum(signal)/n

def calculate_skewness(signal):
    std = np.std(signal)
    mean = np.mean(signal)
    n = len(signal)
    coeff = (n*(n-1))**0.5/(n*(n-2))
    skewness = np.sum(np.power(signal-mean,3))/std**3
    
    return coeff * skewness

def calculate_kurtosis(signal):
    std = np.std(signal)
    mean = np.mean(signal)
    n = len(signal)
    coeff1 = (n*(n+1))/((n-1)*(n-2)*(n-3))
    coeff2 = (n-1)**2/((n-2)*(n-3))
    kurtosis = np.sum(np.power(signal-mean,4))/std**2
    return coeff1*kurtosis - coeff2*3

def calculate_shannon_entropy(signal, base=None):
    unique, counts = np.unique(signal, return_counts=True)
    n = len(signal)
    frequencies = counts/n
    return -np.sum(frequencies*np.log(frequencies))

def calculate_lbp(signal):
    n = len(signal)
    middle_point = signal[n//2]
    
    signal = signal - middle_point
    
    # Thresholding
    signal[signal>=0] = 1
    signal[signal<0] = 0
    
    lbp = 0
    for i, e in enumerate(signal):
        if i != n//2:
            lbp += e*2**i
        
    return lbp

def calculate_phase_synchrony(y1,y2):
    sig1_hill=sig.hilbert(y1)
    sig2_hill=sig.hilbert(y2)
    phase_y1=np.unwrap(np.angle(sig1_hill))
    phase_y2=np.unwrap(np.angle(sig2_hill))
    inst_phase_diff=phase_y1-phase_y2
    n = len(inst_phase_diff)
    
    sin_sum = np.sum(np.sin(inst_phase_diff))
    cos_sum = np.sum(np.cos(inst_phase_diff))
    
    return (1/n)*np.sqrt(np.power(sin_sum,2) + np.power(cos_sum,2))

def calculate_lbp(signal1, signal2):
    n_size = 9
    neighbourhoods1 = sliding_window(signal1,n_size,4)
    neighbourhoods2 = sliding_window(signal2,n_size,4)
    histogram1 = defaultdict(int)
    histogram2 = defaultdict(int)
    
    #Compute histogram of first signal
    for neighbours1 in neighbourhoods1:
        middle_point = neighbours1[n_size//2]
        neighbours1 = neighbours1 - middle_point

        # Thresholding
        neighbours1[neighbours1>=0] = 1
        neighbours1[neighbours1<0] = 0

        lbp = 0
        for i, e in enumerate(neighbours1):
            if i != n_size//2:
                lbp += e*2**i
                
        if np.sum(np.abs(neighbours1[1:]-neighbours1[:-1])) <= 2:
            histogram1[lbp] += 1
        else:
            histogram1[-1] += 1
    
    #Compute histogram of second signal
    for neighbours2 in neighbourhoods2:
        middle_point = neighbours2[n_size//2]
        neighbours2 = neighbours2 - middle_point

        # Thresholding
        neighbours2[neighbours2>=0] = 1
        neighbours2[neighbours2<0] = 0

        lbp = 0
        for i, e in enumerate(neighbours2):
            if i < n_size//2:
                lbp += e*2**i
            if i > n_size//2:
                lbp += e*2**(i-1)

        if np.sum(np.abs(neighbours2[1:]-neighbours2[:-1])) <= 2:
            histogram2[lbp] += 1
        else:
            histogram2[-1] += 1

    dkl_1 = 0.000001
    dkl_2 = 0.000001
    
    for k in histogram1.keys():
        dkl_1 += histogram1[k] * (np.log(histogram1[k]+1)-np.log(histogram2[k]+1))
    for k in histogram2.keys():
        dkl_2 += histogram2[k] * (np.log(histogram2[k]+1)-np.log(histogram1[k]+1))
        
    resistor_average_difference = 1/dkl_1 + 1/dkl_2

    return resistor_average_difference

def sliding_window(signal,winSize,step=1):
    """Returns a generator that will iterate through
    the defined chunks of input sequence.  Input sequence
    must be iterable."""

    try: it = iter(signal)
    except TypeError:
        raise Exception("**ERROR** sequence must be iterable.")
    if not ((type(winSize) == type(0)) and (type(step) == type(0))):
        raise Exception("**ERROR** type(winSize) and type(step) must be int.")
    if step > winSize:
        raise Exception("**ERROR** step must not be larger than winSize.")
    if winSize > len(signal):
        raise Exception("**ERROR** winSize must not be larger than sequence length.")
 
    # Pre-compute number of windows to emit
    num_windows = int(((len(signal)-winSize)/step)+1)
 
    # Do the work
    for i in range(0,num_windows*step,step):
        yield np.array(signal[i:i+winSize])


def calculate_feature(eegs, window_size, step_size, feature, sensor1=-1, sensor2=-1, seizure=-1):
    """
    This function apply a specific feature to a signal using windows.
    
    Parameters
    ----------
    eegs : list
        A list containing a list of seizure data. 
        Each list of seizure data contains a list of array for one 
        or several hours of records for this seizure
    sensor : int
        ID of the sensors we want to process
    window_size: int
        Size of the window for the sliding window processing
    step_size: int
        Step size to move the window onto the signal
    feature: function
        Function computing the feature on a window of data

    Returns
    -------
    list
        Returns a list with the same hierarchy as input eegs list with
        the computed features of the signals.
    """

    feature_signals = []
    signal2=None
    
    if seizure != -1:
        eegs = [eegs[seizure]]
        
    for seizure_eegs in eegs:
        seizure_feature_signals = []

        for eeg in seizure_eegs:

            if sensor1 != -1:
                signal1 = eeg[sensor1]
            else:
                signal1 = eeg
                
            if sensor2 != -1:
                signal2 = eeg[sensor2]

            windows1 = sliding_window(signal1, window_size, step_size)
            
            if sensor2 != -1:
                windows2 = sliding_window(signal2, window_size, step_size)

            feature_signal = []
            if sensor2 != -1:
                for window1, window2 in zip(windows1,windows2):
                        feature_signal.append(feature(window1, window2))
            else:
                for window1 in windows1:
                    feature_signal.append(feature(window1))

            seizure_feature_signals.append(feature_signal)
        feature_signals.append(seizure_feature_signals)

    return feature_signals