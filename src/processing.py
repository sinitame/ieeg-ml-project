import numpy as np

def calculate_signal_power(signal, start, end):
    """
    This function calculate the energy of a signal between a range
    """
    return np.sum(np.power(signal[start:end],2))