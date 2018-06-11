from EM import EM
import numpy as np
import matplotlib.pyplot as plt

# em is an em object
# n is the index of patient
# bin_size is the size of interval used to bin the data
# true model is a boolean variable indicating whether we have the model generaing
# the data 
def plot(em, n, bin_size, true_model=False):
    print('Patient {}'.format(n))
    time_unit = bin_size / 60
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
    plt.plot(times[0:em.last_train_obs[n]], em.y[n, 0:em.last_train_obs[n]], '.', label = 'actual observed values (for training)', color='b')
    plt.plot(times[em.last_train_obs[n]:em.last_obs[n]], em.y[n, em.last_train_obs[n]:em.last_obs[n]], '.', label = 'actual observed values (for testing)', color='r')
    #plt.axvline(x=em.last_train_obs[n]-1, color='m', linestyle='--')
    colors = ['r', 'y', 'm', 'c', 'b']
    for treatment in range(em.N):
        for t in np.nonzero(em.X[n, :, treatment])[0]:
            if t >= em.last_obs[n]:
                break
            plt.axvline(x=t * time_unit, linestyle=':', color=colors[treatment])
    plt.xlabel('time (hrs)')
    plt.ylabel('INR')
    plt.title('Model Results')
    plt.legend()
    fig.set_figheight(8)
    fig.set_figwidth(15)
    plt.show()