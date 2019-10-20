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


