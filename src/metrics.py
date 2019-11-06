def false_alarm(signal, indexes, threshold):

    # Delete values inside seizure
    signal_without_seizure = np.delete(signal, np.arange(indexes[0][0], indexes[0][1], 1))
    # Rescale values
    signal_without_seizure = signal_without_seizure / np.max(signal_without_seizure)
    #Count the number of false alarm according to the threshold
    false_alarm = np.sum(np.where(signal_without_seizure >= threshold, 1, 0))

    return false_alarm


def delay(signal, indexes, threshold):
    """
    TODO: Change the delay value in time not in number of window
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
        delays.append(delay_count)
    return delays

def precision(signal, indexes, threshold):
    max_energy = np.max(signal[indexes[0][0]:indexes[0][1]]+signal[indexes[1][0]:indexes[1][1]])
    detection = 0
    for start, end in indexes:
            seizure_signal = signal[start:end+1]
            # Normalize values
            seizure_signal = seizure_signal/max_energy
            
            if np.max(seizure_signal) >= threshold:
                detection += 1
            
    return detection/len(indexes)