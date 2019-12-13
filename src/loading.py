from tqdm import tqdm
import scipy.io
import numpy as np
from tqdm import tqdm
import urllib.request
import os



class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)



def download_patient_file(data_path, patient_id, file_name, verbose=False):
    
    
    database_url = "http://ieeg-swez.ethz.ch/long-term_dataset/"
    patient_path = "ID{value:0>{width}}".format(value=patient_id, width=2)
    
    file_url = os.path.join(database_url, patient_path, file_name)
    
    if verbose:
        print("\tCreating patient directory")

    patient_dir = os.path.join(data_path, patient_path)
    os.makedirs(patient_dir, exist_ok=True)
    
    if verbose:
        print("\tDirectory: ", patient_dir)
        print('\tBeginning file download with urllib2...')
        print("\tFile: ", file_url)

    save_name = os.path.join(patient_dir, file_name)
    with DownloadProgressBar(unit='B', unit_scale=True,
                             miniters=1, desc=file_url.split('/')[-1]) as t:
        urllib.request.urlretrieve(file_url, filename=save_name, reporthook=t.update_to)
    
    if verbose:
        print("File downloaded, saved at: ", save_name)
    
    
    
def compute_files_of_seizures(patient_id, seizures_start,seizures_end, all_seizures=False, delta = 0):
    """
    This function calculate in which file the seizure is occuring 
    depending on the start time and the end time in seconds given in
    the info file.
    
    inputs:
        - patient_id : ID of the patient studied
        - seizure_start: beginning of the seizure indicated in s
        - seizure_end: ending of the seizure inficated in s
    
    output:
        - returns the list of the files containning seizure data 
        (+ previous and next hours if delta != 0)
        
    TODO : Check the min and max possible hours
    """
    
    prefix = "../data/"
    hours = []
    files = []

    if not all_seizures:
        seizures_start = [seizures_start[0]]
        seizures_end = [seizures_end[0]]

    for s_start, s_end in zip(seizures_start,seizures_end):
        seizure_start, seizure_end = s_start[0], s_end[0]
        #Get the hour of starting
        start_hour = int(seizure_start/(60*60))+ 1 - delta
        #Get the hour of ending
        end_hour = int(seizure_end/(60*60)) + 1 + delta 
        h_range = np.arange(start_hour,end_hour + 1, 1)
        hours.append(h_range)
        files.append([prefix + "ID{value:0>{width}}/ID{value:0>{width}}_{hour}h.mat".format(value=patient_id,width=2,hour=h) for h in h_range])
    
    return hours, files

def compute_seizures_ranges(seizures_start,seizures_end,fs):
    """
    This function computes the range of each seizure in sampling time.
    """
    ranges = []
    for s_start, s_end in zip(seizures_start,seizures_end):
        seizure_start, seizure_end = s_start[0], s_end[0]
        start_hour = int(seizure_start/(60*60)) #Get the hour of starting
        end_hour = int(seizure_end/(60*60)) #Get the hour of ending
        hours = np.arange(start_hour,end_hour+1, 1)

        max_sample = fs*(60*60) - 1

        # Compute the indice of the first sample of seizure
        start_seizure_second = seizure_start%3600
        start_seizure_sample = int(start_seizure_second*fs)

        # Compute indice of the last sample of seizure
        end_seizure_second = seizure_end%3600
        end_seizure_sample = int(end_seizure_second*fs)

        if len(hours) > 1:
            for hour in hours:
                if hour == min(hours):
                    ranges.append((start_seizure_sample,max_sample))
                elif hour == max(hours):
                    ranges.append((0,end_seizure_sample))
                else:
                    ranges.append((0,max_sample))
        else:
            ranges.append((start_seizure_sample,end_seizure_sample))
    
    return ranges 

def load_patient_seizures(data_path, patient_id, all_seizures=False, delta=0, verbose=False):
    """
    Function that loads patient informations and relevant data (iEEG during a seizure).
    If data is not in the data directory, data is downloaded on the ETH iEEG database.
    
    TODO: Propagate all_seizure boolean to all functions
    """
    
    patient_path = "ID{value:0>{width}}".format(value=patient_id, width=2)
    info_file = "ID{value:0>{width}}_info.mat".format(value=patient_id, width=2)
    
    
    infos = {}
    
    # Load info file
    infos_file_path = os.path.join(data_path, patient_path, info_file)
    
    if not os.path.exists(infos_file_path):
        infos_file_name = os.path.basename(infos_file_path)
        print("\nInfo file not found, need to download it")
        download_patient_file(data_path, patient_id, infos_file_name)
        
    data_infos = scipy.io.loadmat(infos_file_path)
    
    # Get experiment informations
    seizure_start = data_infos['seizure_begin']
    seizure_end = data_infos['seizure_end']
    sf = data_infos['fs'][0][0]
    
    infos["seizure_start"] = seizure_start
    infos["seizure_end"] = seizure_end
    infos["sf"] = sf
    
    # Find the file of the seizure
    hours, files = compute_files_of_seizures(patient_id, seizure_start, seizure_end, all_seizures, delta)
    ranges = compute_seizures_ranges(seizure_start,seizure_end,sf)
    
    if verbose:
        print("Seizure starts (s): ", seizure_start)
        print("Seizure ends (s): ", seizure_end)
        print("Duration (s): ", seizure_end-seizure_start)
        print("Sampled frequency (Hz): ", sf)
        print("EEG files: ", files)
        print("Hour of seizure: ", hours)
        print("Samples ranges: ", ranges)
    
    # Load seizure data
    eegs = []
    
    if all_seizures:
        seizures_files = files
    else:
        seizures_files = [files[0]]

    for seizure_files_to_download in seizures_files:
        seizure_eegs = []
        for file_path in seizure_files_to_download:
            if not os.path.exists(file_path):
                file_name = os.path.basename(file_path)
                print("\n{} file not found, need to download it".format(file_name))
                download_patient_file(data_path, patient_id, file_name, verbose)
            data = scipy.io.loadmat(file_path)
            seizure_eegs.append(data['EEG'])
        eegs.append(seizure_eegs)
        
    return {"eegs":eegs, "ranges": ranges, "infos": infos}