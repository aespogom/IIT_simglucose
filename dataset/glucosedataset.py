from datetime import datetime, timedelta
import multiprocessing
import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

PATIENT_PARA_FILE = os.path.join('data', 'sampled_insilico_vparams.csv')
vparams = pd.read_csv(PATIENT_PARA_FILE)

CONTROL_QUEST = os.path.join('data','sampled_insilico_quest.csv')
quest = pd.read_csv(CONTROL_QUEST)

PATIENT_IDS = [pat_name for pat_name in quest['Name']]

def get_init_state(pat_name, vparams):
    params = vparams.loc[vparams.Name == pat_name].squeeze()
    initial_state = params.iloc[2:15]
    BW = params.BW
    initial_EGPb = params.EGPb
    return list(initial_state), BW, initial_EGPb

def get_quest(pat_name, quest):
    params = quest.loc[quest.Name == pat_name].squeeze()
    initial_quest = params.iloc[1:5]
    return list(initial_quest)

def get_cho_time(pat_name):
    ''' Get time at when the patient had the first meal'''
    PATIENT_VAL_FILE = os.path.join('data', 'insilico_validation', f'{pat_name}.csv')
    validation = pd.read_csv(PATIENT_VAL_FILE)
    val_cho = validation.loc[validation.CHO > 0].iloc[0].squeeze()
    return val_cho.Time

def get_gb_insulin(pat_name, time_at):
    PATIENT_VAL_FILE = os.path.join('data', 'insilico_validation', f'{pat_name}.csv')
    validation = pd.read_csv(PATIENT_VAL_FILE)
    val_gb = validation.loc[validation.Time == str(time_at)].squeeze()
    return val_gb.BG, val_gb.insulin, val_gb.CHO

def get_interval_datetimes(input_datetime_str):
    input_datetime = datetime.strptime(input_datetime_str, '%Y-%m-%d %H:%M:%S')

    # Initialize lists to store datetimes
    previous_datetimes = []
    posterior_datetimes = []

    # Calculate 30 minutes previous to input datetime
    previous_time = input_datetime - timedelta(minutes=30)

    # Generate datetimes in steps of 3 minutes, 30 minutes previous to input datetime
    while previous_time < input_datetime:
        previous_datetimes.append(str(previous_time))
        previous_time += timedelta(minutes=3)

    # Calculate 30 minutes posterior to input datetime
    posterior_time = input_datetime + timedelta(minutes=33)

    # Generate datetimes in steps of 3 minutes, 30 minutes posterior to input datetime
    while input_datetime < posterior_time:
        posterior_datetimes.append(str(input_datetime))
        input_datetime += timedelta(minutes=3)

    return previous_datetimes + posterior_datetimes

class GlucoseDataset(Dataset):
    def __init__(self, data, pat_ids):
        self.data = data
        self.pat_ids = pat_ids
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        # Get input sequence (30 minute pre-meal and during-meal) and output (30 min post-meal)
        input_sequence = self.data[index][:10, :]
        output = self.data[index][11:, :3]
        return input_sequence, output, self.pat_ids[index]

def create_dataset(vparams, quest):
    data_patients = {}
    for pat_name in PATIENT_IDS:
        data_temporal = []
        # Load static (non temporal) parameters
        initial_state, BW, initial_EGPb = get_init_state(pat_name, vparams)
        initial_quest = get_quest(pat_name, quest)
        cho_date = get_cho_time(pat_name=pat_name)
        datetime_intervals = get_interval_datetimes(cho_date)
        for time_at in datetime_intervals:
            gb, insulin, cho = get_gb_insulin(pat_name, time_at)
            # Add temporal and non temporal information
            data_temporal.append([gb, insulin, cho, *initial_state, BW, initial_EGPb, *initial_quest])
        
        # Append to the rest of patients
        data_patients[pat_name] = np.array(data_temporal)

    # split into train and test sets
    train_size = int(len(data_patients.keys()) * 0.6)
    val_size = int(len(data_patients.keys()) * 0.2)

    train_ids = PATIENT_IDS[0:train_size]
    selected_arrays = []
    for pat_id in train_ids:
        selected_arrays.append(data_patients[pat_id])
    train = np.stack(selected_arrays, axis=0)

    val_ids = PATIENT_IDS[train_size:train_size+val_size]
    selected_arrays = []
    for pat_id in val_ids:
        selected_arrays.append(data_patients[pat_id])
    val = np.stack(selected_arrays, axis=0)

    test_ids = PATIENT_IDS[train_size+val_size:]
    selected_arrays = []
    for pat_id in test_ids:
        selected_arrays.append(data_patients[pat_id])
    test = np.stack(selected_arrays, axis=0)
    
    return  torch.from_numpy(train).to(torch.float32), train_ids, \
            torch.from_numpy(val).to(torch.float32), val_ids, \
            torch.from_numpy(test).to(torch.float32), test_ids

def setup_loaders():
    AVAIL_GPUS = min(1, torch.cuda.device_count())
    AVAIL_CPUS = multiprocessing.cpu_count()
    torch.manual_seed(56)

    num_workers = (4 * AVAIL_GPUS) if (AVAIL_GPUS > 0) else AVAIL_CPUS
    batch_size = 2 # DUAL INPUT!

    train_data, train_ids, val_data, val_ids, test_data, test_ids = create_dataset(vparams=vparams, quest=quest)

    # Aumentar size para dataset completo
    train_loader = DataLoader(
        GlucoseDataset(train_data, train_ids),
        shuffle=True,
        pin_memory=True,
        num_workers=num_workers,
        batch_size=batch_size
    )

    val_loader = DataLoader(
        GlucoseDataset(val_data, val_ids),
        shuffle=True,
        pin_memory=True,
        num_workers=num_workers,
        batch_size=batch_size
    )
    
    test_loader = DataLoader(
        GlucoseDataset(test_data, test_ids),
        shuffle=True,
        pin_memory=True,
        num_workers=num_workers,
        batch_size=batch_size
    )

    return train_loader, val_loader, test_loader
