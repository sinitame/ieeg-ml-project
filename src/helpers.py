import numpy as np
from processing import calculate_signal_power
from scipy.spatial import distance

def top_signals(eegs, seizure_range, top=1, id_seizure = None):
    """
    """
    
    sensors_power=[]
    num_files_per_seizure = len(eegs[0])
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
        
def merge_seizures_data(data, window_ranges):
    """
    This function merge the different data obtain for each seizure together.
    
    Parameters
    ----------
    data : list
        A list containing a list of seizure data. 
        Each list of seizure data contains a list of array for one 
        or several hours of records for this seizure
    window_ranges : list
        list of tupples containing the ranges of seizures expressed in
        window indexes.

    Returns
    -------
    list, list
        Return:
            - A list containing the merged data for each seizure (one array of data per seizure)
            - A list of the updated window ranges for each seizure with an offset resulting from the merge
    
    """
    all_energy_signals = []
    new_indexes = []
    number_of_seizures = len(data)
    
    for i,seizure_signals in enumerate(data):
        merged_seizure_signals = []
        
        #Calculate index of seizure (always the middle table of data)
        index_seizure = len(seizure_signals)//2
        
        # Calculate the new index position after merge
        n = len(seizure_signals[0])
        offset = index_seizure * n
        new_indexes.append((window_ranges[i][0] + offset , window_ranges[i][1] + offset ))
        
        # Merge seizure signals
        for energy_signal in seizure_signals:
            merged_seizure_signals += energy_signal
        
        # Add merged seizure signals to output
        all_energy_signals.append(merged_seizure_signals)

    return new_indexes, all_energy_signals
        
def merge_all_data(data, window_ranges):
    """
    This function merge all seizure data together.
    
    Parameters
    ----------
    data : list
        A list containing a list of seizure data. 
        Each list of seizure data contains a list of array for one 
        or several hours of records for this seizure
    window_ranges : list
        list of tupples containing the ranges of seizures expressed in
        window indexes.

    Returns
    -------
    list, list
        Return:
            - A list containing the merged data.
            - A list of the updated window ranges for each seizure with an offset resulting from the merge.
    
    """
    all_energy_signals = []
    new_indexes = []
    number_of_seizures = len(data)
    
    for i,seizure_signals in enumerate(data):
        merged_seizure_signals = []
        
        #Calculate index of seizure (always the middle table of data)
        number_of_files = len(seizure_signals)
        index_seizure = number_of_files//2
        
        # Calculate the new index position ofter merge
        n = len(seizure_signals[0])
        offset = index_seizure * n + i * n * number_of_files
        new_indexes.append((window_ranges[i][0] + offset , window_ranges[i][1] + offset ))
        
        for energy_signal in seizure_signals:
            merged_seizure_signals += energy_signal

        all_energy_signals += merged_seizure_signals
    return new_indexes, all_energy_signals


def scale_signal(signal, win_size, step_size):
    """
    This function scales the processed data (using windows) 
    to fit the dimension of the input signal.
    """
    scaled_signal = [0]*step_size
    
    for point in signal:
        scaled_signal += [point]*step_size
    
    return scaled_signal

def convert_to_seconds(index_value, win_size, step_size, n=1843200, fs=512.0):
    """
    Convert a given window index into it's corresponding time value.
    
    Parameters
    ----------
    index_value : int
        index_value in the window slices space.
    win_size : int
        size of the window.
    step_size: int
        step size used to generate the signal.

    Returns
    -------
    float
        Return the value in seconds of a given index_value.
    """

    # Convert the window index value into sample value
    number_of_slices = (n-win_size)/step_size + 1
    sample_index_value = (n*(index_value))/number_of_slices

    # Convert the sample index value into second
    time_value = sample_index_value / fs
    
    return time_value

def convert_sample_ranges_to_window_ranges(ranges, win_size, step_size,n):
    number_of_slices = (n-win_size)/step_size + 1
    window_ranges = []
    for (start_range, end_range) in ranges:
        start_window_range = int(start_range*(number_of_slices/n))
        end_window_range = int(end_range*(number_of_slices/n))
        window_ranges.append((start_window_range, end_window_range))
    return window_ranges

def compute_score_on_seizures(threshold, false_alarm, precision, delay, weights, lambda_=0.00001, d_max=10):
    all_scores = []
    w1, w2, w3 = weights
    
    # Compute min and max values of false alarm rate and delay
    min_false_alarm = min(false_alarm)
    max_false_alarm = max(false_alarm)
    
    delay_list = list(zip(*delay))
    min_delay = min([min(d) for d in delay_list])
    max_delay = max([max(d) for d in delay_list])
    
    for seizure in range(len(delay_list)):
        
        # Normalize delay and false alarms
        delays_i = np.array(delay_list[seizure])

        normalized_delays = (delays_i - min_delay) / (max_delay - min_delay)
        normalized_false_alarms = (np.array(false_alarm) - min_false_alarm) / (max_false_alarm - min_false_alarm)
        
        # Compute score
        scores = 1/(np.sqrt((w1*normalized_delays)**2 + (w2*normalized_false_alarms)**2 + 1/(w3*np.array(precision)**2) + lambda_))
        
        # Adding constrains
        scores = np.where(np.array(delays_i)>d_max, 0, scores)
        
        max_score = np.max(scores)
        index_max_score = np.argmax(scores)
        best_threshold = threshold[index_max_score]
        
        all_scores.append((max_score, best_threshold))
        
        # Printing best score threshold and metrics
        print("Max score for seizure {} is {} for t={}".format(seizure, max_score, best_threshold))
        print("FA:", false_alarm[index_max_score])
        print("Delay:", delay[index_max_score][seizure])
        print("Precision:", precision[index_max_score])
        print()
    return all_scores

def compute_overall_score(seizure_scores):
    feature_overall_score = 0
    
    matrix_distances = distance.cdist(seizure_scores, seizure_scores, 'euclidean')
    distance_score = np.triu(matrix_distances).sum()/matrix_distances.shape[0]
    
    # Compute seizures scores average
    seizure_scores_values = [score[0] for score in seizure_scores]
    n_values = len(seizure_scores_values)
    seizures_overall_score = sum(seizure_scores_values)/n_values
    
    feature_overall_score = seizures_overall_score / distance_score
    
    print("Distance score", distance_score)
    print("Averaged seizures score", seizures_overall_score)
    print("Overall score", feature_overall_score)
    
    return feature_overall_score