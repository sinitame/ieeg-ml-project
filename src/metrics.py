import numpy as np
from helpers import convert_to_seconds

def calculate_false_alarm(signal, indexes, threshold):
    """
    Calculate the False Alarm rate of a signal.
    
    Parameters
    ----------
    signal: array
        array of signal values
    indexes: array
        array of seizures indexes. (start,end) index for each seizures.
    threshold: float
        threshold value set for detection.
    Returns
    -------
    int
       return the number of FA for a given threshold.
    """

    # Delete values inside seizure
    index_to_delete = []
    for start, end in indexes:
        index_to_delete += list(np.arange(start, end, 1))

    signal_without_seizure = np.delete(signal, index_to_delete)

    # Rescale values
    signal_without_seizure = signal_without_seizure / np.max(signal_without_seizure)
    #Count the number of false alarm according to the threshold
    false_alarm = np.sum(np.where(signal_without_seizure >= threshold, 1, 0))

    return false_alarm


def calculate_delay(signal, indexes, threshold, win_size, step_size):
    """
    Calculate the Delay of detection for a given threshold.
    
    Parameters
    ----------
    signal: array
        array of signal values
    indexes: array
        array of seizures indexes. (start,end) index for each seizures.
    threshold: float
        threshold value set for detection.
    win_size : int
        size of the window.
    step_size: int
        step size used to generate the signal.
    
    Returns
    -------
    int
       return the number detection upon all seizures.
    """
    delays = []
    for seizure_start, seizure_end in indexes:
        seizure_signal = signal[seizure_start:seizure_end+1]
        delay_count = 0
        # Normalize values
        seizure_signal = seizure_signal/ np.max(seizure_signal)

        for e in seizure_signal:
            if e < threshold:
                delay_count +=1
            else:
                break
        delays.append(convert_to_seconds(delay_count, win_size, step_size))

    return delays

def calculate_precision(signal, indexes, threshold):
    """
    Calculate the Precision of a given threshold.
    
    Parameters
    ----------
    signal: array
        array of signal values
    indexes: array
        array of seizures indexes. (start,end) index for each seizures.
    threshold: float
        threshold value set for detection.

    Returns
    -------
    int
       return the number detection upon all seizures.
    """
    signal_of_interest = []
    for start, end in indexes:
        signal_of_interest += signal[start:end]
    max_energy = np.max(signal_of_interest)
    
    detection = 0
    for start, end in indexes:
            seizure_signal = signal[start:end+1]
            # Normalize values
            seizure_signal = seizure_signal/max_energy
            
            if np.max(seizure_signal) >= threshold:
                detection += 1
            
    return detection/len(indexes)


def compute_metrics(data, ranges, win_size, step_size):
    thresholds = np.arange(0,1,0.001)
    false_alarms = []
    delays = []
    precisions = []

    for t in list(thresholds):
        false_alarms.append(calculate_false_alarm(data, ranges, t))
        delays.append(calculate_delay(data, ranges, t, win_size, step_size))
        precisions.append(calculate_precision(data, ranges, t))
        
    return thresholds, false_alarms, delays, precisions