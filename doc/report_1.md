



# Toward a systematic way to evaluate feature selection for iEEG analyses

## Introduction

Epilepsy is a neuro-logical disorder characterized by a recurrence of a brief abnormal and uncontrollable electrical discharge of the brain called seizure. According to World Health Organization (WHO), approximately 50 million people world- wide have epilepsy. Upon all these patients, approximately 70% are responding to prevalent cure like medications and surgeries, while 30%  are untreated or poorly treated. 

One of the promising solution to help the patients who can not control their seizures using medications and surgeries is to use implantable electrical stimulators that can monitor their brain activity and generate an electrical stimulation to stop their seizure when it is detected by the device. Different studies have shown how stimulations can help stop an occurnig seizure but in order to do so in real time in an implanted device require an efficient way of detecting when a seizure is occurring.

It is know that machine learning can help achieve this goal, and many studies have presented methods that can detect a seizure with low computational cost. However, as seizure signature can vary a lot from a patient to another and as a consequence most of those methods lacks of generalization.

In order to overcome this weakness, we will focus on one of the key component of machine learning : feature selection. Fiding the relevant feature for a given problem can often drasticly improve the performance of a model wether it's a classification or a regression task. But without a way of quantifying the quality of a given feature for a model or a patient it is hard to know if the selected feature are the best for our problem. 

The aim of this project is to present a systematic way to evaluate feature selection for iEEG analyses and to apply this method to extract insights from the SWEC-ETHZ iEEG Database.

## Dataset

The SWEC-ETHZ iEEG Database is composed of two datasets:

- **Long-term Dataset**: 2656 hours of anonymized and continuous intracranial electroencephalography (iEEG) of 18 patients with pharmaco-resistant epilepsies.
- **Short-term Dataset**: 100 anonymized intracranially recorded electroencephalographic (iEEG) datasets of 16 patients with pharmaco-resistant epilepsy.

For this study we will only focus in the first dataset (**Long-term Dataset**). The iEEG signals were recorded intracranially by strip, grid, and depth electrodes. After 16-bit analog-to-digital conversion, the iEEG signals were median-referenced and digitally band-pass filtered between 0.5 and 120 Hz using a fourth-order Butterworth filter prior to analysis and written onto disk at a rate of 512 or 1024 Hz. Forward and backward filtering was applied to minimize phase distortions. All the iEEG recordings were visually inspected by an EEG board-certified and experienced epileptologist (K.S.) for identification of seizure onsets and endings and exclusion of channels continuously corrupted by artifacts.

### Exploratory data analysis

The iEEG recordings of the 18 patients are provided into `.mat` files. Each file contains one hour of recording and the data is stored in TxM array where T is the number of iEEG electrodes and M is the number of sampling points for one hour. An additional file provided for each patients also gives informations about the sampling frequency, the beginning and the end of the seizures (in seconds).

The following table gives an idea of the duration range of the seizure for each patients in seconds. 

**We can also notice with the following table that:**

- Some patients (8, 14) have much more seizures during the recording period than overs (1,2).

- Duration of seizures from a patient to another can vary a lot ranging from (6.22s for patient 5 to 613.74s  for patient 2).
- Duration of a seizure for a given patient can vary a lot (std of patient 18 is more than 100s)



| Patient | number of seizures | mean duration | std duration | min duration | max duration |
| :-----: | :----------------: | :-----------: | :----------: | :----------: | :----------: |
|    1    |         2          |    601.787    |   16.9381    |    589.81    |   613.764    |
|    2    |         2          |    88.0625    |   2.55547    |   86.2555    |   89.8695    |
|    3    |         4          |    64.6619    |   4.14865    |   60.5449    |    68.234    |
|    4    |         14         |    41.9404    |    13.779    |   7.77257    |   68.6968    |
|    5    |         4          |    16.6878    |   0.512328   |   15.9232    |    17.013    |
|    6    |         8          |    45.8905    |   32.8707    |   29.3444    |   126.882    |
|    7    |         4          |    69.5688    |   38.6222    |   14.1287    |   98.8148    |
|    8    |         70         |    21.9668    |    53.88     |   6.22336    |   413.385    |
|    9    |         27         |    42.377     |   35.5274    |   18.7427    |   148.283    |
|   10    |         17         |    70.8471    |   10.7102    |   61.2513    |   106.262    |
|   11    |         2          |    91.5471    |   11.9259    |   83.1142    |    99.98     |
|   12    |         9          |    146.461    |   33.0413    |   106.836    |   194.754    |
|   13    |         7          |    103.004    |   60.9422    |   40.1964    |    188.44    |
|   14    |         60         |    25.8067    |   24.3826    |   6.37323    |   100.775    |
|   15    |         2          |    94.5809    |   35.5882    |   69.4163    |   119.746    |
|   16    |         5          |    190.445    |   50.6856    |   120.293    |   245.196    |
|   17    |         2          |    97.9362    |   1.28925    |   97.0246    |   98.8479    |
|   18    |         5          |    199.132    |   100.565    |   71.4387    |   300.651    |

To get a better view of the repartition of seizure duration, and the repartition of the number of seizures for each patients, we can refer to the following figures.

**Seizure duration frequency**

![seizure_duration_frequency](img/seizure_duration_frequency.png)

**Number of seizure per patient frequency**

![number_of_seizure_frequency](img/number_of_seizure_frequency.png)



In addition to the heterogeneity of the seizures between patients and for a given patient, we can also observe heterogeneity within the recordings of a given seizure depending on the electrode on which we collect the signals. The following plot shows the plot of 5 electrodes of patient 2 during a period where a seizure occurs (the seizure is highlighted in blue).

**Example of seizure signal**

![patient_2_seizure_1_channels](img/patient_2_seizure_1_channels.png)



This first exploration of the data gives an idea of how hard it can be to find a generalized machine learning model which is able to detect a seizure with high accuracy, low false alarm rate and with a small delay for all patients. These observations motivates us to find a way to select relevant features for each patients according to their seizure episode history in order to design a specific machine learning model which could maximize the previous metrics.

## Methods

In this part, we will present the method that we designed in order  to evaluate feature selection for iEEG analyses according to each patient. In order to do so, we will define three metrics that we will focus on. After that, we will present how those metrics were computed with different features and finally we will present the results that we obtainned with our method.

As a simple start for our experiment, the classification will be based on a threshold. Seizures will be considered as detected when the value of the feature's signal is above the given threshold and as undetected if it's value is below the given threshold.

### Metrics

- **Precision:** The accuracy is defined as the number of detected seizure upon all seizures. A seizure is considered as detected if there is at least one positive classification (one of the feature value is above the given threshold) of the feature signal within the range of the seizure. This metric is important for our problem as we want to be sure that the system will detect all the seizures of the patient in order to generate a stimulation that will stop the seizure.
- **False alarm:** The False alarms are the number of points that are classified as `seizure` outside a real seizure. This metric is very important as we don't want the patient to receive a stimulation when no seizure occurs. It can be dangerous for him.
- **Delay:** The delay is defined as the number of seconds between the real beginning of a seizure and the first signal being classified as a seizure. This metric is also capital as the stimulation needs to occur as soon as possible in order to stop the seizure efficiently.

### Sliding window calculation



### Features

For now, the following features have been tested on patient 1 and 2 in order to ensure that all of the functions work correctly and are generic enough in order to be applied to the full dataset. All features are calculated using a sliding window of size  `sliding_window=128` and a step size of size `step_size=64` .

#### Min

$$
\min_{x \in [|x_0, x_0+N|]}\{y_x\}
$$

![](img/patient_2_seizure_1_min_scaled.png)



####  Max

$$
\max_{x \in [|x_0, x_0+N|]}\{y_x\}
$$

![](img/patient_2_seizure_1_max_scaled.png)



#### Energy

$$
Energy = \sum_{x=x_0}^{x_0+N} y_x^2
$$




![](img/patient_2_seizure_1_energy_scaled.png)

#### Line length

Line length is defined as the running sum of the absolute differences between all consecutive samples within a predefined window. The value of this feature grows as the data sequence magnitude or signal variance increases.
$$
LineLength = \sum_{x=x_0+1}^{x_0+N}|y_x - y_{x-1}|
$$




![patient_2_seizure_1_linelength_scaled](img/patient_2_seizure_1_linelength_scaled.png)

#### Moving Average

Moving average is commonly used with time series data to smooth out short-term fluctuations and highlight longer-term trends or cycles. 

**Feature calculation**
$$
MovingAverage = \frac{1}{N}\sum_{x=x_0}^{x_0+N}y_x
$$



![patient_2_seizure_1_movingaverage_scaled](img/patient_2_seizure_1_movingaverage_scaled.png)

#### Skewness

Skewness indicates the symmetry of the probability density function of the amplitude of a time series. It is a good indicator of the tendency of the time series amplitude in a given portion of time (here we look at this value during a window). 

![](img/skewness.png)

A window with many small values and few large values is positively skewed (right tail) and will have a positive skewness while a window with many large values and few small values is negatively skewed (left tail) and will have a negative skewness.

**Feature calculation**
$$
Skewness = \frac{\sqrt{N(N-1)}}{N(N-2)} g \text{  with   } g = \frac{\sum_{x=x_0}^{x_0+N} (y_x - \mu)^3}{\sigma^3}
$$


![](img/patient_2_seizure_1_skewness_scaled.png)







#### Kurtosis

Kurtosis measures the peakedness of the probability density function of the amplitude of a time series. A kurtosis value close to zero indicates a Gaussian-like peakedness. Probability density function with relatively sharp peaks have a positive kurtosis while probability density function  that have relatively flat peaks have a negative kurtosis.
$$
Kurtosis =  \frac{\sum_{x=x_0}^{x_0+N} (y_x - \mu)^4}{N\sigma^2} - 3
$$
![](img/patient_2_seizure_1_kurtosis_scaled.png)

#### Shannon Entropy

$$
ShannonEntropy = \sum_{x=x_0}^{x_0+N} \text{freq}(y_x) \text{log}(\text{freq}(y_x))
$$



#### Local Binary Patterns

Local binary patterns is a type of visual descriptor used for classification in computer vision. LBP was first described in 1994, it has since been found to be a powerful feature for texture classification. Even if we are not dealing with images in our case, an adapted version of the LBP for 1D dimentional signals can possibly be a good feature for the task of seizure classification. The choice of this feature is particularly motivated by a paper using this feature for voice signal segmentation and voice activity detection. More informations about this work can be found here : [Local binary patterns for 1-D signal processing](https://ieeexplore.ieee.org/document/7096717).

**Feature computation**

Inside each of our sliding window (here the window go from 1 to 21), we shift a window in order to extract the $P$ neighbours of a given data point $p_i$ (here $P$ is equal to 8). Then we substract to each of the neighbors of $i$ the value of $p_i$ and set their values to 1 if the result is equal or positive or 0 si the result is negative.

![](img/lbp_explained.jpg)
$$
LBP\_decimal\_value= \sum_{r=0}^{p/2^{-1}}\left\{S[x[i+r-P/_{2}]-x[i]]2^{r}+S[x[i+r+1]-x[i]]2^{r+P/2}\right\}
$$
Those values are then recorded in an histogram:

![](img/lbp_explained_histograms.jpg)



At the end, we can compare the similarity between two signals within a window by comparing the histograms obtained with the previous method. In order to do so, we use the Kullback– Leibler (KL) divergence  as described in [Quantitative Analysis of Facial Paralysis Using Local Binary Patterns in Biomedical Videos](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=4806065).





#### Phase synchrony



Neurons initiate electrical oscillations that are contained in multiple frequency bands such as alpha (8–12 Hz), beta (13–30 Hz) and gamma (40–80 Hz) and have been linked to a wide range of cognitive and perceptual processes. It has been shown that before and during a seizure the amount of synchrony between these oscillations from neurons located in different regions of the brain changes significantly. Thus, the amount of synchrony between multiple neural signals is a strong indicator in predicting or detecting seizures. To quantify the level of synchrony between two neural signals, a phase locking value (PLV) can be computed that accurately measures the phase-synchronization between two signal sites in the brain.

**Feature computation**
$$
Y_{0}=Re(Y_{0})+jIm(Y_{0}),\quad Y_{1}=Re(Y_{1})+jIm(Y_{1})
$$

$$
\phi_{k}=\arctan{Im(V_{k})\over Re(V_{k})}
$$

$$
\Delta\phi=\phi_{1}-\phi_{0}
$$


$$
PLV={1\over N}\sqrt{\left(\sum_{i=0}^{N-1}sin(\Delta \phi_{i})\right)^{2}+\left(\sum_{i=0}^{N-1}cos(\Delta \phi_{i})\right)^{2}}
$$
![](img/patient_2_seizure_1_phase_synchrony_scaled.png)



![](img/patient_2_seizure_1_preprocessed_phase_synchrony_scaled.png)



### Algorithm

The algorithm used to compute the Precision, the Delay and the False alarm rate is very simple. 

Given a threshold  `t`, we do the following:

- Precision: we check if there is a value of our feature which is superior to `t` within each seizure range (it means that the seizure have been detected by the system). We return the number of detection upon the number of seizures.
- Delay: we compute the number of values between the begining of the seizure and the first detection of a seizure (feature value >`t`). We convert the number of values (which is in the feature space a number of window slices) into a number of samples (number of values in the sampling space). We then convert the obtained value to time using the sampling frequency `fs`.
- False Alarm: we count the number of values of the feature signal that are above the threshold outside a seizure.

## Results

### Energy

|                          Patient 1                           |                          Patient 2                           |
| :----------------------------------------------------------: | :----------------------------------------------------------: |
| ![patient_1_all_metrics_energy](img/patient_1_all_metrics_energy.png) | ![patient_2_all_metrics_energy](img/patient_2_all_metrics_energy.png) |

### Line length 

|                          Patient 1                           |                          Patient 2                           |
| :----------------------------------------------------------: | :----------------------------------------------------------: |
| ![patient_1_all_metrics_linelength](img/patient_1_all_metrics_linelength.png) | ![patient_2_all_metrics_linelength](img/patient_2_all_metrics_linelength.png) |

### Moving average

|                          Patient 1                           |                          Patient 2                           |
| :----------------------------------------------------------: | :----------------------------------------------------------: |
| ![patient_1_all_metrics_movingaverage](img/patient_1_all_metrics_movingaverage.png) | ![patient_2_all_metrics_movingaverage](img/patient_2_all_metrics_movingaverage.png) |



## Discussion



## Conclusion







