import numpy as np
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
    coeff = (n-1)/((n-2)*(n-3))
    kurtosis = np.sum(np.power(signal-mean,4))/std**2
    
    return coeff*((n+1)*kurtosis + 6)

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