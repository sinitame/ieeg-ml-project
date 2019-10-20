import matplotlib.pyplot as plt
import numpy as np

def plot_one_signal(data, seizure_start, seizure_end, plot_range):

    start = max(int(seizure_start - plot_range //2), 0)
    end = min(int(seizure_end + plot_range//2), 512*3600 - 1)
    time = np.arange(start,end,1)
    
    plt.figure(figsize=(20,10))
    plt.plot(time,data[start:end] )
    plt.axvspan(seizure_start, seizure_end, color='royalblue', alpha=0.5)
    plt.xlabel('Sample step')
    plt.ylabel('Voltage')
    eeg_plot = plt.plot()
    
    return eeg_plot


def plot_signal_on_ax(ax, data, seizure_start, seizure_end, plot_range):

    start = max(int(seizure_start - plot_range //2), 0)
    end = min(int(seizure_end + plot_range//2), 512*3600 - 1)
    time = np.arange(start,end,1)
    
    ax.plot(time,data[start:end] )
    ax.axvspan(seizure_start, seizure_end, color='royalblue', alpha=0.5)
    ax.set_ylabel('Voltage')
    eeg_plot = plt.plot()
    
    return eeg_plot

def plot_multiple_signals(eegs, list_of_signals, seizure_start, seizure_end, plot_range):
    fig, axs =  plt.subplots(len(list_of_signals), 1,figsize=(20, 10),sharex=True)
    fig.subplots_adjust(hspace=0)
    
    for i,signal in enumerate(list_of_signals):
        plot_signal_on_ax(axs[i], eegs[0][signal] , seizure_start, seizure_end, plot_range)