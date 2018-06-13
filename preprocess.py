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
# return missing_list, a list of index of data whose missing_pct is larger than a threshold
# use to inform binning_X and binning_c which indices not to include
def binning_y(data, bins, missing_pct):
    #y_mtx = np.zeros((len(data), len(bins)-1))
    y_list = []
    missing_list = []
    for i, d in enumerate(data):
        y_binned = np.full(len(bins)-1, np.nan)
        binned_index = np.digitize(d['obs_times'], bins)
        for j in range(1, y_binned.shape[0]):
            index = np.where(binned_index==j)[0]
            if index.shape[0] > 0:
                values = d['obs_y'][index]
                y_binned[j-1] = np.mean(values)
        if get_missing_pct_single(y_binned) < missing_pct:
            #y_mtx[i, :] = y_binned
            y_list.append(y_binned)
        else:
            missing_list.append(i)
    y_mtx = np.concatenate(y_list, axis=0)
    y_mtx = np.reshape(y_mtx, (len(y_list), len(bins)-1))
    return (y_mtx, missing_list)

def binning_X(data, bins, missing_list):
    #X_mtx = np.zeros((len(data), len(bins)-1, 5))
    X_list = []
    names = ['nsaid', 'transfusion_plasma', 'transfusion_platelet', 'anticoagulant', 'aspirin']
    for i, d in enumerate(data):
        if i in missing_list:
            continue
        X_i_list = []
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
            #X_mtx[i, :, k] = x_binned
            X_i_list.append(x_binned)
        X_i_mtx = np.transpose(np.stack(X_i_list, axis = 0))
        X_list.append(X_i_mtx)
    X_mtx = np.stack(X_list, axis=0)
    return X_mtx

def get_missing_pct_single(y):
    last_obs = np.where(np.invert(np.isnan(y)))[0][-1] + 1
    y_cuttail = y[0:last_obs]
    # the missing percentage of each patient is calculated by dividing the total number of time points between the first
    # and last observations by the number of nans between the first and last observations 
    missing_pct = np.where(np.isnan(y_cuttail))[0].shape[0] / y_cuttail.shape[0]
    return missing_pct * 100

def get_missing_pct_total(y):
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

def get_c(data, missing_list):
    #c_mtx = np.zeros((len(data), 3))
    c_list = []
    c_length = 3
    for i, d in enumerate(data):
        if i in missing_list:
            continue
        c = np.concatenate([d['chronic'], d['demographic']], axis=0)
        #c_mtx[i, :] = c
        c_list.append(c)
    c_mtx = np.concatenate(c_list)
    c_mtx = np.reshape(c_mtx, (len(c_list), c_length))
    return c_mtx

def preprocess(data, cutoff, bin_size, missing_pct=30):
    data = cutoff_num_obs(data, cutoff)
    data = align_time_series(data)
    bins = get_bins(data, bin_size)
    y, missing_list = binning_y(data, bins, missing_pct)
    X = binning_X(data, bins, missing_list)
    c = get_c(data, missing_list)
    return (y, X, c)