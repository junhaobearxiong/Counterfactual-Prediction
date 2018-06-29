from EM import EM
import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict

# em is an em object
# n is the index of patient
# bin_size is the size of interval used to bin the data
# true_model indicates if we have the true data generating model
# model is the the model generaing the data if we have it, default to None
def plot(em, n, time_unit, true_model=False, model=None):
    if n > 0:
        print('Patient {}'.format(n))
    times = [t * time_unit for t in range(em.last_obs[n])]
    fig = plt.figure()
    if em.train_pct < 1 and em.last_train_obs[n] < em.last_obs[n]:
        z, y = em.predict(n)
        upper = np.zeros(em.last_obs[n])
        lower = np.zeros(em.last_obs[n])
        upper[em.last_train_obs[n]:] = np.sqrt(em.sigma_filter[n, em.last_train_obs[n]:em.last_obs[n]] + em.sigma_2)
        lower[em.last_train_obs[n]:] = -np.sqrt(em.sigma_filter[n, em.last_train_obs[n]:em.last_obs[n]] + em.sigma_2)
        plt.fill_between(times, y+upper, y+lower, color='.8')
        #plt.plot(times, z, label = 'predicted state values')
        plt.plot(times, y, label = 'predicted observed values', color='g', linestyle='--')
    if true_model:
        plt.plot(times, model.z[n, 0:em.last_obs[n]], label = 'actual state values')
        plt.plot(times, em.mu_smooth[n, 0:em.last_obs[n]], color='m', label = 'predicted state values')
    plt.plot(times[0:em.last_train_obs[n]], em.y[n, 0:em.last_train_obs[n]], '.', label = 'actual observed values (for training)', color='b')
    plt.plot(times[em.last_train_obs[n]:em.last_obs[n]], em.y[n, em.last_train_obs[n]:em.last_obs[n]], '.', label = 'actual observed values (for testing)', color='r')
    colors = ['b', 'y', 'c', 'r', 'm']
    treatment_types = ['nsaid', 'transfusion_plasma', 'transfusion_platelet', 'anticoagulant', 'aspirin']
    for treatment in range(em.N):
        for t in np.nonzero(em.X[n, :, treatment])[0]:
            if t >= em.last_obs[n]:
                break
            plt.axvline(x=t * time_unit, linestyle=':', color=colors[treatment], label=treatment_types[treatment])
    #plt.plot(times[0:em.last_train_obs[n]], em.mu_filter[n, 0:em.last_train_obs[n]], label='filtered values')
    plt.xlabel('time (hrs)')
    plt.ylabel('INR')
    plt.title('Model Results')
    # To avoid duplicated labels when ploting verticle lines for treatments
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = OrderedDict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())
    fig.set_figheight(8)
    fig.set_figwidth(15)
    plt.show()