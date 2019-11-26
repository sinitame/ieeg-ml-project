from scipy.signal import butter, lfilter, firwin, spectrogram, freqz, freqs

# Butterwoth filter

def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band', analog=True)
    return b, a


def calculate_butter_filtered_signal(signal, f_min, f_max, fs):
    b, a = butter_bandpass(f_min, f_max, fs, order=3)
    filtered_signal = lfilter(b, a, signal)
    return filtered_signal

# FIR filter

def calculate_fir_filtered_signal(signal, f_min, f_max, fs, order=15):
    taps = firwin(order, [f_min, f_max], pass_zero=False, nyq=0.5*fs)
    filtered_signal = lfilter(taps, 1.0, signal)
    return filtered_signal


def calculate_filtered_signal(eegs, sensor, f_min, f_max, ftype='fir',order=3, fs=512, seizure=-1):
    """
    This function apply a specific feature to a signal using windows.
    
    Parameters
    ----------
    eegs : list
        A list containing a list of seizure data. 
        Each list of seizure data contains a list of array for one 
        or several hours of records for this seizure.
        Each list contains a list with the data of each sensors.
    sensor : int
        ID of the sensors we want to process
    f_min: int
        Min frequency cut
    f_max: int
        Max frequency cut
    ftype: str
        Type of filter ('fir' or 'butter')
    order: int
        Order for the filter
    order: int
        Sampling frequency
    seizure: int
        Id of the seizure (default is all)

    Returns
    -------
    list
        Returns a list with the same hierarchy as input eegs list with
        the filtered signals.
    """

    filtered_signals = []
    
    if seizure != -1:
        eegs = [eegs[seizure]]
        
    for seizure_eegs in eegs:
        seizure_filtered_signals = []

        for eeg in seizure_eegs:
            signal = eeg[sensor]

            if ftype == 'fir':
                filtered_signal = calculate_fir_filtered_signal(signal, f_min, f_max, fs, order=order)
            elif ftype == 'butter':
                filtered_signal = calculate_butter_filtered_signal(signal, f_min, f_max, fs, order=order)
            else:
                return "Error, filter must be 'fir' or 'butter'."

            seizure_filtered_signals.append(filtered_signal)
        filtered_signals.append(seizure_filtered_signals)

    return filtered_signals