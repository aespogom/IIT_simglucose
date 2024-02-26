from datetime import datetime, timedelta
import multiprocessing
import os
import random
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import DataLoader, Dataset

random.seed(56)

PATIENT_PARA_FILE = os.path.join('data', 'sampled_insilico_vparams.csv')
vparams = pd.read_csv(PATIENT_PARA_FILE)

CONTROL_QUEST = os.path.join('data','sampled_insilico_quest.csv')
quest = pd.read_csv(CONTROL_QUEST)

PATIENT_PARA_FILE_TEST = os.path.join('data', 'insilico_vparams.csv')
vparams_test = pd.read_csv(PATIENT_PARA_FILE_TEST)

CONTROL_QUEST_TEST = os.path.join('data','insilico_quest.csv')
quest_test = pd.read_csv(CONTROL_QUEST_TEST)

PATIENT_IDS = [pat_name for pat_name in quest['Name']]
random.shuffle(PATIENT_IDS)
PATIENT_IDS_TEST = [pat_name for pat_name in quest_test['Name']]

def get_init_state(pat_name, vparams):
    ''' Get initial state of the patient'''
    params = vparams.loc[vparams.Name == pat_name].squeeze()
    return [params['x0_ 4'], params['x0_ 5'], params['x0_ 6'], params['x0_ 8'], params['x0_10'], params['x0_11'], params['x0_12'], params['x0_13']]

def get_meal_time(pat_name, test=False):
    ''' Get time at when the patient had the first meal'''
    if not test:
        PATIENT_VAL_FILE = os.path.join('data', 'insilico_validation', f'{pat_name}.csv')
    else:
        PATIENT_VAL_FILE = os.path.join('data', 'insilico_test', f'{pat_name}.csv')
    validation = pd.read_csv(PATIENT_VAL_FILE)
    val_cho = validation.loc[validation.CHO > 0].iloc[0].squeeze()
    return val_cho.Time

def get_gb_insulin_cho_at(pat_name, time_at, test=False):
    ''' Get the blood glucose level, insulin and CHO at time'''
    if not test:
        PATIENT_VAL_FILE = os.path.join('data', 'insilico_validation', f'{pat_name}.csv')
    else:
        PATIENT_VAL_FILE = os.path.join('data', 'insilico_test', f'{pat_name}.csv')
    validation = pd.read_csv(PATIENT_VAL_FILE)
    val_gb = validation.loc[validation.Time == str(time_at)].squeeze()

    insuline_time = datetime.strptime(time_at, '%Y-%m-%d %H:%M:%S') + timedelta(minutes=3)
    val_in = validation.loc[validation.Time == str(insuline_time)].squeeze()
    return val_gb.BG, val_in.insulin, val_gb.CHO

def get_pre_meal_datetimes(input_datetime_str):
    input_datetime = datetime.strptime(input_datetime_str, '%Y-%m-%d %H:%M:%S')

    # Initialize lists to store datetimes
    previous_datetimes = []

    # Calculate 30 minutes previous to input datetime
    previous_time = input_datetime - timedelta(minutes=30)

    # Generate datetimes in steps of 3 minutes, 30 minutes previous to input datetime
    while previous_time < input_datetime:
        previous_datetimes.append(str(previous_time))
        previous_time += timedelta(minutes=3)

    return previous_datetimes

class GlucoseDataset(Dataset):
    def __init__(self, data, pat_ids):
        self.data = data
        self.pat_ids = pat_ids
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        # Get input sequence (30 minute pre-meal and during-meal) and output (30 min post-meal)
        input_sequence = self.data[index,:-1]
        output = self.data[index,-1]
        return input_sequence, output, self.pat_ids[index]

def create_dataset(vparams, vparams_test):
    ''' Collect all information for input and output tensors '''
    # TRAIN AND VAL DATA
    data_patients = {}
    for pat_name in PATIENT_IDS:
        initial_state = get_init_state(pat_name, vparams)
        cho_date = get_meal_time(pat_name=pat_name)
        gb_meal, insulin, cho = get_gb_insulin_cho_at(pat_name, cho_date)
        datetime_intervals = get_pre_meal_datetimes(cho_date)
        temp_gb = []
        for time_at in datetime_intervals:
            gb, _, _ = get_gb_insulin_cho_at(pat_name, time_at)
            temp_gb.append(gb)
        # Append to the rest of patients
        # CHO is appended twice: first for input data for model to be scaled, second for input data for the simulator as raw
        data_patients[pat_name] = np.array([*initial_state, insulin, cho, cho, *temp_gb, gb_meal])

    # split into train and test sets
    train_size = int(len(data_patients.keys()) * 0.6)

    scaler = StandardScaler()

    train_ids = PATIENT_IDS[0:train_size]
    selected_arrays = []
    for pat_id in train_ids:
        selected_arrays.append(data_patients[pat_id])
    train = np.stack(selected_arrays, axis=0)
    # Fit the scaler only to the first part of the data (excluding the two last columns)
    scaler.fit(train[:, :-12])
    X_train_normalized_reshaped = scaler.transform(train[:, :-12])
    train[:, -11:] *= 0.01 #GB
    # train[:, -12] CHO
    X_train_normalized_reshaped = np.concatenate([X_train_normalized_reshaped, train[:, -12:]], axis=1)

    val_ids = PATIENT_IDS[train_size:]
    selected_arrays = []
    for pat_id in val_ids:
        selected_arrays.append(data_patients[pat_id])
    val = np.stack(selected_arrays, axis=0)
    # Fit the scaler only to the first part of the data (excluding the two last columns)
    X_val_normalized_reshaped = scaler.transform(val[:, :-12])
    val[:, -11:] *= 0.01 #GBs
    # val[:, -12] CHO without standarization
    X_val_normalized_reshaped = np.concatenate([X_val_normalized_reshaped, val[:, -12:]], axis=1)

    # TEST DATA
    data_patients = {}
    for pat_name in PATIENT_IDS_TEST:
        initial_state = get_init_state(pat_name, vparams_test)
        cho_date = get_meal_time(pat_name=pat_name, test=True)
        gb_meal, insulin, cho = get_gb_insulin_cho_at(pat_name, cho_date, test=True)
        datetime_intervals = get_pre_meal_datetimes(cho_date)
        temp_gb = []
        for time_at in datetime_intervals:
            gb, _, _ = get_gb_insulin_cho_at(pat_name, time_at, test=True)
            temp_gb.append(gb)
        # Append to the rest of patients
        # CHO is appended twice: first for input data for model to be scaled, second for input data for the simulator as raw
        data_patients[pat_name] = np.array([*initial_state, insulin, cho, cho, *temp_gb, gb_meal])

    selected_arrays = []
    for pat_id in PATIENT_IDS_TEST:
        selected_arrays.append(data_patients[pat_id])
    test = np.stack(selected_arrays, axis=0)
    # Fit the scaler only to the first part of the data (excluding the two last columns)
    X_test_normalized_reshaped = scaler.transform(test[:, :-12])
    test[:, -11:] *= 0.01 #GB
    # test[:, -12] CHO
    X_test_normalized_reshaped = np.concatenate([X_test_normalized_reshaped, test[:, -12:]], axis=1)

    return  torch.from_numpy(X_train_normalized_reshaped).to(torch.float32), train_ids, \
            torch.from_numpy(X_val_normalized_reshaped).to(torch.float32), val_ids, \
            torch.from_numpy(X_test_normalized_reshaped).to(torch.float32), PATIENT_IDS_TEST

def setup_loaders():
    AVAIL_GPUS = min(1, torch.cuda.device_count())
    AVAIL_CPUS = multiprocessing.cpu_count()
    torch.manual_seed(56)

    num_workers = (4 * AVAIL_GPUS) if (AVAIL_GPUS > 0) else AVAIL_CPUS
    batch_size = 2 # DUAL INPUT!

    train_data, train_ids, val_data, val_ids, test_data, test_ids = create_dataset(vparams=vparams, vparams_test=vparams_test)

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
