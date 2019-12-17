# 2019 semester project @EPFL

## Epileptic seizure prediction on iEEGs signals using Machine learning

#### Project description
Epilepsy is a neuro-logical disorder characterized by a recurrence of a brief abnormal and uncontrollable electrical discharge of the brain called seizure. According to World Health Organization (WHO), approximately 50 million people world- wide have epilepsy. Upon all these patients, approximately 70% are responding to prevalent cure like medications and surgeries, while 30%  are untreated or poorly treated. 

One of the promising solution to help the patients who can not control their seizures using medications and surgeries is to use implantable electrical stimulators that can monitor their brain activity and generate an electrical stimulation to stop their seizure when it is detected by the device. Different studies have shown how stimulations can help stop an occurnig seizure but in order to do so in real time in an implanted device require an efficient way of detecting when a seizure is occurring.

It is know that machine learning can help achieve this goal, and many studies have presented methods that can detect a seizure with low computational cost. However, as seizure signature can vary a lot from a patient to another and as a consequence most of those methods lacks of generalization.

In order to overcome this weakness, this project focus on one of the key component of machine learning : feature selection. Fiding the relevant feature for a given problem can often drasticly improve the performance of a model wether it's a classification or a regression task. But without a way of quantifying the quality of a given feature for a model or a patient it is hard to know if the selected feature are the best for our problem. 

The aim of this project is to present a systematic way to evaluate feature selection for iEEG analyses and to apply this method to extract insights from the SWEC-ETHZ iEEG Database.

## Code implementations

A set of tools have been implemented in order to facilitate experiments. All those helper function are located in the [source](src/) directory of this project. Example usage can be found in [notebooks](notebooks/).

**Feature processing (`features.py`)**
This file contains all the function used to compute features on iEEG signals.

**Filtering methods (`filters.py`)**     
This file contains all the function needed to filter signals.

**Helpers (`helpers.py`)**
This file contains a set of functions in order to perform the following:

- Selecting best sensor (according to the power signals during seizure)
- Rescale iEEG signals and features
- Merge signals
- Compute scores

**Data loading functions (`loading.py`)**    
This file contains all the functions to load and download patient data from the official ETHZ database.

**Metrics measurments (`metrics.py`)**   
This file contains all the functions used to compute the metrics used to score features.

**Visualization helpers (`visualization.py`)**
This file contains some helper function to plot iEEG signals easily.

### Notebooks

A set of notebooks have been developped in order to show the usage of the different helpers:

**Dataset analysis**
- data_loading_plotting.ipynb 
- dataset_exploration.ipynb
- iEGG_visualization.ipynb

**Features analysis**
- implemented_features.ipynb
- Feature_selection.ipynb
- Features_evaluation.ipynb
- Feature_selection_patient_1.ipynb
- Feature_selection_patient_2.ipynb

**Signal processing**
- Filtering.ipynb
- Fourrier_transform.ipynb

## Final report

The final report of this project can be found [here](doc/final_report.pdf)

## Final presentation

The slides of the final presentation for this project can be found [here](slides/final_presentation.pdf)
