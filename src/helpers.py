import numpy as np
from processing import calculate_signal_power

def top_signals(eegs, seizure_range, top=1, id_seizure = None):
    """
    """
    
    sensors_power=[]
    num_files_per_seizure = len(eegs)
    seizure_file = num_files_per_seizure // 2
    
    if id_seizure:
        eegs = [eegs[id_seizure]]
    

    for i,eegs_seizure in enumerate(eegs):
        for sensor_i in range(eegs_seizure[0].shape[0]):
            pow_sensor_i = 0
            eeg_file = eegs_seizure[seizure_file]
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
        
def merge_seizures_data(energy_signals, indexes):
    """
    This function merge the different data obtain for each seizure together.
    """
    all_energy_signals = []
    new_indexes = []
    number_of_seizures = len(energy_signals)
    
    for i,seizure_signals in enumerate(energy_signals):
        merged_seizure_signals = []
        
        #Calculate index of seizure (always the middle table of data)
        index_seizure = len(seizure_signals)//2
        
        # Calculate the new index position after merge
        n = len(seizure_signals[0])
        offset = index_seizure * n
        new_indexes.append((indexes[i][0] + offset , indexes[i][1] + offset ))
        
        # Merge seizure signals
        for energy_signal in seizure_signals:
            merged_seizure_signals += energy_signal
        
        # Add merged seizure signals to output
        all_energy_signals.append(merged_seizure_signals)

    return new_indexes, all_energy_signals
        
def merge_all_data(energy_signals, indexes):
    """
    This function merge all seizure data together.
    """
    all_energy_signals = []
    new_indexes = []
    number_of_seizures = len(energy_signals)
    
    for i,seizure_signals in enumerate(energy_signals):
        merged_seizure_signals = []
        
        #Calculate index of seizure (always the middle table of data)
        number_of_files = len(seizure_signals)
        index_seizure = number_of_files//2
        
        # Calculate the new index position ofter merge
        n = len(seizure_signals[0])
        offset = index_seizure * n + i * n * number_of_files
        new_indexes.append((indexes[i][0] + offset , indexes[i][1] + offset ))
        
        for energy_signal in seizure_signals:
            merged_seizure_signals += energy_signal

        all_energy_signals += merged_seizure_signals
    return new_indexes, all_energy_signals


def scale_signal(signal, win_size, step_size):
    scaled_signal = [0]*step_size
    
    for point in signal:
        scaled_signal += [point]*step_size
    
    return scaled_signal