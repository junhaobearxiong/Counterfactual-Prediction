from EM import EM
import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict

# em is an em object
# n is the index of patient
# time_unit is the size of interval used to bin the data
# treatment_types is the list of treatments for this em object 
# true_model indicates if we have the true data generating model
# model is the the model generaing the data if we have it, default to None
def plot(em, n, time_unit, signal_name, treatment_types, true_model=False, model=None):
    if n > 0:
        print('Patient {}'.format(n))
    times = [t * time_unit for t in range(em.last_obs[n])]
    empty = np.zeros(0)
    if em.X_prev_given:
        times = [-j * time_unit for j in range(em.J, 0, -1)] + times
        # trajectory in negative time is nothing, but we need this as placeholder so everything
        # we plot is of the same length
        empty = np.full(em.J, np.nan) 
    fig = plt.figure()
    if em.train_pct < 1 and em.last_train_obs[n] < em.last_obs[n]:
        z, y = em.predict(n)
        upper = np.zeros(em.last_obs[n])
        lower = np.zeros(em.last_obs[n])
        upper[em.last_train_obs[n]:] = np.sqrt(em.sigma_filter[n, em.last_train_obs[n]:em.last_obs[n]] + em.sigma_2)
        lower[em.last_train_obs[n]:] = -np.sqrt(em.sigma_filter[n, em.last_train_obs[n]:em.last_obs[n]] + em.sigma_2)
        plt.fill_between(times, np.concatenate([empty, y+upper]), np.concatenate([empty, y+lower]), color='.8')
        plt.plot(times, np.concatenate([empty, z]), label = 'predicted state values')
        plt.plot(times, np.concatenate([empty, y]), label = 'predicted observed values', color='g', linestyle='--')
    else:
        plt.plot(times, np.concatenate([empty, em.mu_smooth[n, 0:em.last_train_obs[n]]]),
         label='predicted state values', color='g')
    if true_model:
        plt.plot(times, np.concatenate([empty, model.z[n, 0:em.last_obs[n]]]), label = 'actual state values')
    plt.plot(times[empty.shape[0]:empty.shape[0]+em.last_train_obs[n]], 
        em.y[n, 0:em.last_train_obs[n]], '.', 
        label = 'actual observed values (for training)', color='b')
    plt.plot(times[empty.shape[0]+em.last_train_obs[n]:empty.shape[0]+em.last_obs[n]], 
        em.y[n, em.last_train_obs[n]:em.last_obs[n]], '.', label = 'actual observed values (for testing)', color='r')
    colors = ['b', 'tab:orange', 'm', 'tab:brown', 'k', 'r']
    for treatment in range(em.N):
        for t in np.nonzero(em.X[n, :, treatment])[0]:
            if t >= em.last_obs[n]:
                break
            plt.axvline(x=t * time_unit, linestyle=':', color=colors[treatment], label=treatment_types[treatment])
        if em.X_prev_given:
            for t in np.nonzero(em.X_prev[n, :, treatment])[0]:
                plt.axvline(x=-(empty.shape[0]-t) * time_unit, linestyle=':', color=colors[treatment], label=treatment_types[treatment])
    #plt.plot(times[0:em.last_train_obs[n]], em.mu_filter[n, 0:em.last_train_obs[n]], label='filtered values', color='m')
    plt.xlabel('time (hrs)')
    plt.ylabel(signal_name)
    plt.title('Model Prediction')
    # To avoid duplicated labels when ploting verticle lines for treatments
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = OrderedDict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())
    fig.set_figheight(8)
    fig.set_figwidth(15)
    plt.show()