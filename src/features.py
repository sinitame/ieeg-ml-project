import numpy as np

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


def calculate_feature(eegs, window_size, step_size, feature, sensor=-1, seizure=-1):
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

    energy_signals = []
    
    if seizure != -1:
        eegs = [eegs[seizure]]
        
    for seizure_eegs in eegs:
        seizure_energy_signals = []

        for eeg in seizure_eegs:

            if sensor != -1:
                signal = eeg[sensor]
            else:
                signal = eeg

            windows = sliding_window(signal, window_size, step_size)

            energy_signal = []
            for window in windows:
                energy_signal.append(feature(window))

            seizure_energy_signals.append(energy_signal)
        energy_signals.append(seizure_energy_signals)

    return energy_signals