#!/usr/bin/env python

# Edit this script to add your team's code. Some functions are *required*, but you can edit most parts of the required functions,
# change or remove non-required functions, and add your own functions.

################################################################################
#
# Optional libraries, functions, and variables. You can change or remove them.
#
################################################################################

from helper_code import *
import numpy as np, os, sys
import mne
import pandas as pd
from sklearn.impute import SimpleImputer

import joblib

from sklearn.model_selection import GridSearchCV

import xgboost as xgb

from sklearn.preprocessing import StandardScaler

from tsfresh.feature_extraction import extract_features


################################################################################
#
# Required functions. Edit these functions to add your code, but do not change the arguments of the functions.
#
################################################################################

# Train your model.
def train_challenge_model(data_folder, model_folder, verbose):
    # Find data files.
    if verbose >= 1:
        print('Finding the Challenge data...')

    patient_ids = find_data_folders(data_folder)
    num_patients = len(patient_ids)

    if num_patients==0:
        raise FileNotFoundError('No data was provided.')

    # Create a folder for the model if it does not already exist.
    os.makedirs(model_folder, exist_ok=True)

    # Extract the features and labels.
    if verbose >= 1:
        print('Extracting features and labels from the Challenge data...')

    features = list()
    outcomes = list()
    cpcs = list()

    for i in range(num_patients):
        if verbose >= 2:
            print('    {}/{}...'.format(i+1, num_patients))

        current_features = get_features(data_folder, patient_ids[i])
        features.append(current_features)

        # Extract labels.
        patient_metadata = load_challenge_data(data_folder, patient_ids[i])
        current_outcome = get_outcome(patient_metadata)
        outcomes.append(current_outcome)
        current_cpc = get_cpc(patient_metadata)
        cpcs.append(current_cpc)
       

    features = np.vstack(features)
    outcomes = np.vstack(outcomes)
    cpcs = np.vstack(cpcs)
   
    df = pd.DataFrame(features)

   
    
   

    df.replace('nan', np.nan, inplace=True)

    df = df.apply(pd.to_numeric, errors='ignore')

    imputer = SimpleImputer(strategy='mean')
    
    df = imputer.fit_transform(df)
    
    df = pd.DataFrame(df)
    
    features= df.to_numpy()
    
    scaler = StandardScaler()
    
    features = scaler.fit_transform(features)
 

    # # Train the models.
    if verbose >= 1:
        print('Training the Challenge model on the Challenge data...')


    random_state = 42
    
    
    
    
    param_grid_xgb = {
    'max_depth': [3, 5, 7,9],
    'learning_rate': [0.01, 0.1,0.02],
    'n_estimators': [200,250,300,350,400,450,500],
    'base_score':[0.3,0.4,0.5,0.6],
 
}

    # Create XGBoost classifier and regressor
    xgb_classifier = xgb.XGBClassifier()
    xgb_regressor = xgb.XGBRegressor()
    
    # Perform grid search using cross-validation for classification
    grid_search_clf = GridSearchCV(xgb_classifier, param_grid_xgb, cv=5)
    grid_search_clf.fit(features,outcomes.ravel())
    best_clf = grid_search_clf.best_estimator_
    best_clf.fit(features,outcomes.ravel())
    
    # Perform grid search using cross-validation for regression
    grid_search_reg = GridSearchCV(xgb_regressor, param_grid_xgb, cv=5)
    grid_search_reg.fit(features,cpcs.ravel())
    best_reg = grid_search_reg.best_estimator_
    best_reg.fit(features,cpcs.ravel())

    # Save the models.
    save_challenge_model(model_folder, imputer ,scaler, best_clf, best_reg)

    if verbose >= 1:
        print('Done.')

# Load your trained models. This function is *required*. You should edit this function to add your code, but do *not* change the
# arguments of this function.
def load_challenge_models(model_folder, verbose):
    filename = os.path.join(model_folder, 'models.sav')
    return joblib.load(filename)

# Run your trained models. This function is *required*. You should edit this function to add your code, but do *not* change the
# arguments of this function.
def run_challenge_models(models, data_folder, patient_id, verbose):
    imputer = models['imputer']

    outcome_model = models['outcome_model']
    cpc_model = models['cpc_model']
    scaler = models["scaler"]

    # Extract features.
    features = get_features(data_folder, patient_id)
    features2 = features.reshape(1, -1)
   
    df= pd.DataFrame(features2)

    df.replace('nan', np.nan, inplace=True)

    # Convert columns to numeric (necessary for mean calculation)
    df = df.apply(pd.to_numeric, errors='ignore')
 
    
    df2 = imputer.transform(df)
    df2 = pd.DataFrame(df2)

    features1= df2.to_numpy()
    features = scaler.transform(features1)
  
    outcome = outcome_model.predict(features)[0]
    outcome_probability = outcome_model.predict_proba(features)[0, 1]
    cpc = cpc_model.predict(features)[0]

    # Ensure that the CPC score is between (or equal to) 1 and 5.
    cpc = np.clip(cpc, 1, 5)

    return outcome, outcome_probability, cpc 

################################################################################
#
# Optional functions. You can change or remove these functions and/or add new functions.
#
################################################################################

# Save your trained model.
def save_challenge_model(model_folder, imputer ,scaler , outcome_model, cpc_model):
    d = {'imputer': imputer, 'scaler' : scaler ,'outcome_model': outcome_model, 'cpc_model': cpc_model}
    filename = os.path.join(model_folder, 'models.sav')
    joblib.dump(d, filename, protocol=0)

# Preprocess data.
def preprocess_data(data, sampling_frequency, utility_frequency):
    # Define the bandpass frequencies.
    passband = [0.1, 30.0]

    # Promote the data to double precision because these libraries expect double precision.
    data = np.asarray(data, dtype=np.float64)

    # If the utility frequency is between bandpass frequencies, then apply a notch filter.
    if utility_frequency is not None and passband[0] <= utility_frequency <= passband[1]:
        data = mne.filter.notch_filter(data, sampling_frequency, utility_frequency, n_jobs=4, verbose='error')

    # Apply a bandpass filter.
    data = mne.filter.filter_data(data, sampling_frequency, passband[0], passband[1], n_jobs=4, verbose='error')

    # Resample the data.
    if sampling_frequency % 2 == 0:
        resampling_frequency = 128
    else:
        resampling_frequency = 125
    lcm = np.lcm(int(round(sampling_frequency)), int(round(resampling_frequency)))
    up = int(round(lcm / sampling_frequency))
    down = int(round(lcm / resampling_frequency))
    resampling_frequency = sampling_frequency * up / down
    data = scipy.signal.resample_poly(data, up, down, axis=1)

# #     # Scale the data to the interval [-1, 1].
    min_value = np.min(data)
    max_value = np.max(data)
    if min_value != max_value:
        data = 2.0 / (max_value - min_value) * (data - 0.5 * (min_value + max_value))
    else:
        data = 0 * data
    # min_value = np.min(data)
    # max_value = np.max(data)
    # if min_value != max_value:
    #     data2 = (data - min_value) / (max_value - min_value)
    # else:
    #     data2 =0* data
  

    return data, resampling_frequency 

# Extract features.
def get_features(data_folder, patient_id):
    # Load patient data.

    patient_metadata = load_challenge_data(data_folder,patient_id)
    recording_ids = find_recording_files(data_folder,patient_id)
    num_recordings = len(recording_ids)

    # Extract patient features.
    patient_features = get_patient_features(patient_metadata)

    # Extract EEG features.
    eeg_channels = ['F3', 'P3', 'F4', 'P4' , 'Fz' , 'Cz', 'C3' , 'C4' , 'Pz' , 'T3' , 'T4' ,'T6' ,'O1' ,'O2' , 'Fp1','Fp2','T5','F7','F8']
    group = 'EEG'
   


    eeg_list = []
    eeg_list1 = []
  




   
 
    if num_recordings > 0:
        recording_id = recording_ids[0]
       
       
       
        #for recording_id in recording_ids:
           
        recording_location = os.path.join(data_folder, patient_id, '{}_{}'.format(recording_id, group))
       
        if os.path.exists(recording_location + '.hea'):
            data, channels, sampling_frequency = load_recording_data(recording_location)
            utility_frequency = get_utility_frequency(recording_location + '.hea')
            #print(np.array(data).shape)

            if all(channel in channels for channel in eeg_channels):
                data, channels = reduce_channels(data, channels, eeg_channels)
                data , sampling_frequency  = preprocess_data(data, sampling_frequency, utility_frequency)
                data1 = np.array([data[0, :] - data[1, :], data[2, :] - data[3, :],data[14, :] - data[17, :],data[15, :] - data[18, :],data[4, :] - data[5, :],
                data[5, :] - data[8, :],data[9, :] - data[16, :],data[10, :] - data[11, :],data[12, :] - data[1, :] ,data[13, :] - data[3, :]  ,
                data[0, :] - data[6, :]  ,data[1, :] - data[7, :]  , data[16, :] - data[1, :],data[11, :] - data[3, :], data[0, :] - data[9, :],data[1, :] - data[10, :]
                ])
                
                
                
                # F3-P3, F4-P4 , Fp1-F7 , Fp2-F8 ,Fz-Cz , Cz-Pz ,T3-T5  , T4-T6 , O1-P3, O2-P4 , F3-C3 , F4-C4 ,T5-P3 , T6-P4 ,F3-T3 , F4-T4
                

                eeg_features1 = get_eeg_features(data1, sampling_frequency).flatten()
       
                eeg_features = get_tsfresh_features(data)
          

                eeg_list.append(eeg_features)
                eeg_list1.append(eeg_features1)
               
           
            else:
  
                eeg_list.append(float("nan") * np.ones(228))
                eeg_list1.append(float("nan") * np.ones(64))
  
           
        else:
            eeg_list.append(float("nan") * np.ones(228))
            eeg_list1.append(float("nan") * np.ones(64))
            
             
    else:
        eeg_list.append(float("nan") * np.ones(228))
        eeg_list1.append(float("nan") * np.ones(64))
    
   #if np.array(eeg_list).shape[0] != 1:
       
        # = np.mean(eeg_list,axis = 0)
     #   eeg_features2 = np.mean(eeg_list,axis = 0)
       # eeg_features1 = np.mean(eeg_list1,axis = 0)
    
    
    

    dfs = np.hstack((patient_features,np.array(eeg_list).reshape(228,),np.array(eeg_list1).reshape(64,)))
   
  

    return dfs
    

def get_tsfresh_features(data_rocket):
    data_rocket = pd.DataFrame(data_rocket)
    reshaped_data = pd.DataFrame({
        'time': np.tile(np.arange(data_rocket.shape[1]), data_rocket.shape[0]),
        'id': np.zeros(data_rocket.shape[0] * data_rocket.shape[1], dtype=int),
        'kind': np.repeat(np.arange(data_rocket.shape[0]), data_rocket.shape[1]),
        'signal_value': data_rocket.values.ravel()
    })
    
    # Print the reshaped DataFrame
    
    
    
    fc_parameters = {
        'sum_values':None,
        "absolute_maximum":None,
        "median":None,
        "mean":None,
        "standard_deviation":None,
        "variance":None,
        "length":None,
        "maximum":None,
        "minimum":None,
        "root_mean_square":None,
        "fourier_entropy":[{'bins': 20}],
        "kurtosis":None
    }
    
    extracted_features = extract_features(reshaped_data,column_id="id", column_sort='time', column_kind="kind", column_value="signal_value",default_fc_parameters=fc_parameters,n_jobs=6)
    extracted_features = extracted_features.to_numpy()
    eeg_features = extracted_features.flatten()
   # print(eeg_features.shape)
    return eeg_features
# Extract patient features from the data.
def get_patient_features(data):
    age = get_age(data)
    sex = get_sex(data)
    rosc = get_rosc(data)
    ohca = get_ohca(data)
    shockable_rhythm = get_shockable_rhythm(data)
    ttm = get_ttm(data)

    sex_features = np.zeros(2, dtype=int)
    if sex == 'Female':
        female = 1
        male   = 0
        other  = 0
    elif sex == 'Male':
        female = 0
        male   = 1
        other  = 0
    else:
        female = 0
        male   = 0
        other  = 1

    features = np.array((age, female, male, other,rosc, ohca, shockable_rhythm,ttm))

    return features

# Extract features from the EEG data.
def get_eeg_features(data, sampling_frequency):
    num_channels, num_samples = np.shape(data)

    if num_samples > 0:
        delta_psd, _ = mne.time_frequency.psd_array_welch(data, sfreq=sampling_frequency,  fmin=0.5,  fmax=8.0, verbose=False,n_fft = num_samples,n_per_seg = num_samples)
        theta_psd, _ = mne.time_frequency.psd_array_welch(data, sfreq=sampling_frequency,  fmin=4.0,  fmax=8.0, verbose=False,n_fft = num_samples,n_per_seg = num_samples)
        alpha_psd, _ = mne.time_frequency.psd_array_welch(data, sfreq=sampling_frequency,  fmin=8.0, fmax=12.0, verbose=False,n_fft = num_samples,n_per_seg = num_samples)
        beta_psd,  _ = mne.time_frequency.psd_array_welch(data, sfreq=sampling_frequency, fmin=12.0, fmax=30.0, verbose=False,n_fft = num_samples,n_per_seg = num_samples)

        delta_psd_mean = np.nanmean(delta_psd, axis=1)
        theta_psd_mean = np.nanmean(theta_psd, axis=1)
        alpha_psd_mean = np.nanmean(alpha_psd, axis=1)
        beta_psd_mean  = np.nanmean(beta_psd,  axis=1)
    else:
        delta_psd_mean = theta_psd_mean = alpha_psd_mean = beta_psd_mean = float('nan') * np.ones(num_channels)

    features = np.array((delta_psd_mean, theta_psd_mean, alpha_psd_mean, beta_psd_mean)).T

    return features

# Extract features from the ECG data.
# def get_ecg_features(data):
#     num_channels, num_samples = np.shape(data)

#     if num_samples > 0:
#         mean = np.mean(data, axis=1)
#         std  = np.std(data, axis=1)
#     elif num_samples == 1:
#         mean = np.mean(data, axis=1)
#         std  = float('nan') * np.ones(num_channels)
#     else:
#         mean = float('nan') * np.ones(num_channels)
#         std = float('nan') * np.ones(num_channels)
   

#     features = np.array((mean, std)).T

#     return features

