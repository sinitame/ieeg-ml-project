import matplotlib.pyplot as plt
import numpy as np

def plot_one_signal(data, seizure_ranges, plot_range, seizure_id = -1):

    if seizure_id != -1:
        seizure_ranges = [seizure_ranges[seizure_id]]

    start = max(int(seizure_ranges[0][0] - plot_range //2), 0)
    end = min(int(seizure_ranges[-1][1] + plot_range//2), 512*3600 - 1)
    time = np.arange(start,end,1)
    print("time", len(time))
    print("data", len(data))
    
    plt.figure(figsize=(20,10))
    plt.plot(time,data[start:end])
    for seizure_start, seizure_end in seizure_ranges:
        plt.axvspan(seizure_start, seizure_end, color='royalblue', alpha=0.5)
    plt.xlabel('Sample step')
    plt.ylabel('Voltage')
    eeg_plot = plt.plot()
    
    return eeg_plot


def plot_signal_on_ax(ax, data, seizure_ranges, plot_range, seizure_id = -1):
    
    if seizure_id != -1:
        seizure_ranges = [seizure_ranges[seizure_id]]
        
    start = max(int(seizure_ranges[0][0] - plot_range //2), 0)
    end = min(int(seizure_ranges[-1][1] + plot_range//2), 512*3600 - 1)
    time = np.arange(start,end,1)
    
    ax.plot(time,data[start:end] )
    for seizure_start, seizure_end in seizure_ranges:
        ax.axvspan(seizure_start, seizure_end, color='royalblue', alpha=0.5)
    ax.set_ylabel('Voltage')
    eeg_plot = plt.plot()
    
    return eeg_plot

def plot_multiple_signals(eegs, list_of_signals, seizure_ranges, plot_range, seizure_id = -1):
        
    fig, axs =  plt.subplots(len(list_of_signals), 1,figsize=(20, 10),sharex=True)
    fig.subplots_adjust(hspace=0)
    
    for i,signal in enumerate(list_of_signals):
        plot_signal_on_ax(axs[i], eegs[signal] , seizure_ranges, plot_range, seizure_id)
        

def plot_metrics(thresholds, false_alarms, delays, precision):
    fig, axs =  plt.subplots(2, 2,figsize=(20, 10))
    
    axs[0,0].plot(false_alarms, thresholds)
    axs[0,0].set_ylabel('Threshold')
    axs[0,0].set_xlabel('False Alarm')
    
    for i in range(len(delays[0])):
        axs[0,1].plot([pt[i] for pt in delays], thresholds, label = 'seizure %s'%i)
    axs[0,1].set_ylabel('Threhold')
    axs[0,1].set_xlabel('Delays')
    
    axs[1,0].plot(precision, thresholds)
    axs[1,0].set_ylabel('Threshold')
    axs[1,0].set_xlabel('Precision')
    
    for i in range(len(delays[0])):
        axs[1,1].plot([pt[i] for pt in delays], false_alarms, label = 'seizure %s'%i)
    axs[1,1].set_ylabel('False Alarm')
    axs[1,1].set_xlabel('Delays')
    
def plot_scores(thresholds, scores):
    plt.figure(figsize=(20,10))
    for i,score in enumerate(scores):
        plt.plot(thresholds,score, label = 'seizure {}'.format(i))
    plt.legend()
    plt.show()
    
def plot_feature(signal, feature, start, end, scale=True):
    
    if scale:
        normalization_term = np.max(np.abs(signal))/np.max(np.abs(feature))
    else:
        normalization_term = 1

    sub_signal = np.array(signal[start:end+1])
    sub_feature = np.array(feature[start:end+1])

    plt.figure(figsize=(20,10))
    plt.plot(sub_signal)
    plt.plot(sub_feature*normalization_term)
    plt.show()