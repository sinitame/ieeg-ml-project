



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

### Features

For now, the following features have been tested on patient 1 and 2 in order to ensure that all of the functions work correctly and are generic enough in order to be applied to the full dataset. All features are calculated using a sliding window of size  `sliding_window=128` and a step size of size `step_size=64` .

**Energy**
$$
Energy = \sum_{x=x_0}^{x_0+N} y_x^2
$$


![](img/patient_2_seizure_1_energy_scaled.png)

**Line length**
$$
LineLength = \sum_{x=x_0+1}^{x_0+N}|y_x - y_{x-1}|
$$




![patient_2_seizure_1_linelength_scaled](img/patient_2_seizure_1_linelength_scaled.png)

**Moving Average**
$$
MovingAverage = \frac{1}{N}\sum_{x=x_0}^{x_0+N}y_x
$$




![patient_2_seizure_1_movingaverage_scaled](img/patient_2_seizure_1_movingaverage_scaled.png)

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





