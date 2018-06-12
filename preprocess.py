import numpy as np
import math

# select the data based on a cutoff on the number of observations, default to 5
def cutoff_num_obs(data, cutoff):
    new_data = []
    for d in data:
        if d['obs_times'].shape[0] >= cutoff:
            new_data.append(d)
    return new_data

def get_new_time(old_time, starting_time):
    new_time = np.subtract(old_time, starting_time)
    # there might be some treatments before the first observation, discard them
    new_time = np.delete(new_time, np.where(new_time < 0))
    return new_time

# align the time series so that t=0 is where the observation starts
def align_time_series(data):
    new_data = np.copy(data)
    for i, d in enumerate(data):
        starting_time = d['obs_times'][0]
        names = ['obs', 'anticoagulant', 'aspirin', 'nsaid']
        for name in names:
            new_data[i][name + '_times'] = get_new_time(d[name + '_times'], starting_time)
        for j, t in enumerate(d['transfusion_times']):
            if t.shape[0] > 0:
                new_data[i]['transfusion_times'][j] = get_new_time(t, starting_time)
        new_data[i]['transfusion_plasma_times'] = new_data[i]['transfusion_times'][0]
        new_data[i]['transfusion_platelet_times'] = new_data[i]['transfusion_times'][1]                                                                        
        new_data[i]['event_time'] = d['event_time'] - starting_time
    return new_data

def get_bins(data, bin_size):
    max_obs_times = -np.inf
    for d in data:
        if d['obs_times'][-1] > max_obs_times:
            max_obs_times = d['obs_times'][-1]
    num_bins = math.ceil(max_obs_times / bin_size)
    bins = np.linspace(0, max_obs_times, num_bins)
    return bins

# data is the original data, y_times is obs_times after being aligned at the beginning
# if multiple values are in the same bin, take the average
def binning_y(data, bins):
    y_mtx = np.zeros((len(data), len(bins)-1))
    for i, d in enumerate(data):
        y_binned = np.full(len(bins)-1, np.nan)
        binned_index = np.digitize(d['obs_times'], bins)
        for j in range(1, y_binned.shape[0]):
            index = np.where(binned_index==j)[0]
            if index.shape[0] > 0:
                values = d['obs_y'][index]
                y_binned[j-1] = np.mean(values)
        y_mtx[i, :] = y_binned
    return y_mtx

def binning_X(data, bins):
    X_mtx = np.zeros((len(data), len(bins)-1, 5))
    names = ['nsaid', 'transfusion_plasma', 'transfusion_platelet', 'anticoagulant', 'aspirin']
    for i, d in enumerate(data):
        for k, name in enumerate(names):
            x_binned = np.zeros(len(bins)-1)
            binned_index = np.digitize(d[name + '_times'], bins)
            for j in range(1, x_binned.shape[0]):
                count = np.where(binned_index==j)[0].shape[0]   
                #x_binned[j-1] = count
                if count == 0:
                    x_binned[j-1] = 0
                else:
                    x_binned[j-1] = 1
            X_mtx[i, :, k] = x_binned
    return X_mtx

def get_missing_pct(y):
    y_cuttail = {}
    last_obs = np.zeros(y.shape[0], dtype=int)
    for i in range(y.shape[0]):
        last_obs[i] = np.where(np.invert(np.isnan(y[i, :])))[0][-1] + 1
        y_cuttail[i] = y[i, 0:last_obs[i]]
    # the missing percentage of each patient is calculated by dividing the total number of time points between the first
    # and last observations by the number of nans between the first and last observations 
    missing_pct = 0
    for _, obs in y_cuttail.items():
        pct = np.where(np.isnan(obs))[0].shape[0] / obs.shape[0]
        missing_pct += pct
    missing_pct /= len(y_cuttail)
    return missing_pct * 100

def get_c(data):
    c_mtx = np.zeros((len(data), 3))
    for i, d in enumerate(data):
        c = np.concatenate([d['chronic'], d['demographic']], axis=0)
        c_mtx[i, :] = c
    return c_mtx

def preprocess(data, cutoff, bin_size):
    data = cutoff_num_obs(data, cutoff)
    data = align_time_series(data)
    bins = get_bins(data, bin_size)
    y = binning_y(data, bins)
    X = binning_X(data, bins)
    c = get_c(data)
    return (y, X, c)