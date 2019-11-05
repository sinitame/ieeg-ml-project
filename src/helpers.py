import numpy as np
from processing import calculate_signal_power

def top_signals(eegs, seizure_range, top=1):
    """
    """
    
    sensors_power=[]
    
    for sensor_i in range(eegs[0].shape[0]):
        pow_sensor_i = 0
        for i, eeg_file in enumerate(eegs):
            seizure_start = seizure_range[i][0]
            seizure_end = seizure_range[i][1]
            pow_sensor_i += calculate_signal_power(eeg_file[sensor_i], seizure_start, seizure_end)
        
        sensors_power.append((sensor_i,pow_sensor_i))
    
    ordered_power = sorted(sensors_power, key = lambda x : x[1], reverse=True)
    
    return [id for id,power in ordered_power[:top]]


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
    print(num_windows)
 
    # Do the work
    for i in range(0,num_windows*step,step):
        yield signal[i:i+winSize]

