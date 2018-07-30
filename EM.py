import numpy as np
import scipy as sp
import pandas as pd
import scipy
import scipy.stats
from scipy.stats import norm
from scipy import integrate
import math
import random
import matplotlib.pyplot as plt
import pickle
import time

'''
EM object can perform the following:
1. Inference and parameters estimation
2. Predict ahead based on the learned parameters
3. Calculate prediction MSE

Derivation of the EM updates and the corresponding notations are in Ghahramani 1996
'''
class EM:
    '''
    Parameter list:
    y: observations, shape: [number of patients * number of time points]
    X: treatments after the first observation
        shape: [number of patients * number of time points * number of types of treatments]
    c: static conditions (chronics and demographics), shape: [number of patients * number of types of static conditions]
    J: number of past treatments to consider
    K: number of interaction effects to model (NOT implemented yet)
    train_pct: percentage of EACH observation time series to use for training
    X_prev: treatments prior to the first observation
        shape: [number of patients * number of past effects * number of types of treatments]
        smaller index in number of past effects corresponds to earlier time point
        so the last index has treatment that is closest to time zero
    single_effect: whether to consider only the effect of one treatment in the past
    init_A: the initial mean of coefficients in A (need to be dtype=float)
    init_b: the initial mean of coefficients in b (need to be dtype=float)
    '''
    def __init__(self, y, X, c, J, K, train_pct, X_prev_given=False, X_prev=None, single_effect=False, 
        init_A_given=False, init_A=None, init_b_given=False, init_b=None, init_0=False, init_1=False, 
        init_2=False, init_state=False):
        # Store inputs
        self.num_patients = np.shape(y)[0] 
        self.T = np.shape(y)[1] # length of the observed sequence
        self.y = y # observation matrix
        self.X = X # treatment matrix for treatments after (inclusive) the first observation
        self.X_prev = X_prev # treatment matrix for treatments before the first observation
        self.c = c # chronic conditions
        self.K = K  # number of interaction terms modeled
        self.J = J # number of past treatment effects to be considered
        self.single_effect = single_effect # if true only consider the Jth treatment prior to the current time point

        self.X_prev_given = X_prev_given
        self.train_pct = train_pct # percentage of each time series used for training
        # time of last observations for each patients plus one
        # plus one because last_obs and last_train_obs effectively
        # represents the position right after the end of 'valid' values in an array 
        self.last_obs = self.find_last_obs()
        # time of last observation for training plus one 
        self.last_train_obs = self.find_last_train_obs()
        # list of tuple: (array of index of non-nan inr value, array of non-nan inr value)
        self.valid_inr = self.find_valid_inr()
        
        # Other model parameters
        self.N = np.shape(self.X)[2] # number of treatments
        self.M = np.shape(self.c)[1] # number of chronic conditions
        self.Q = np.zeros((self.num_patients, self.T, self.K)) # interaction term
        
        # Model Parameters to be estimated
        if init_A_given:
            self.A = init_A #np.random.randn(init_A.shape[0], init_A.shape[1])
        else:
            if self.single_effect:
                self.A = np.zeros(self.N) + np.random.randn(self.N)*0.001
            else:
                self.A = np.zeros((self.J, self.N)) + np.random.randn(self.J, self.N)*0.001 # coefficients a_j's
        if init_b_given:
            self.b = init_b #+ np.random.randn(init_b.shape[0])
        else:
            self.b = np.zeros(self.M) + np.random.randn(self.M)*0.001

        self.d = np.zeros(self.K)
        # testing
        if init_0:
            self.sigma_0 = init_0
        else:
            self.sigma_0 = np.abs(np.random.randn()*.001) # initial state variance
        if init_1:
            self.sigma_1 = init_1
        else:
            self.sigma_1 = np.abs(np.random.randn()*.01) # transition variance
        if init_2:
            self.sigma_2 = init_2
        else:
            self.sigma_2 = np.abs(np.random.randn()*.01) # observation variance
        if init_state:
            self.init_z = init_state
        else: 
            self.init_z = np.random.normal(0, np.sqrt(self.sigma_0), size = 1)# initial state mean
        
        self.intercept = np.random.normal(0, 1, size=1)

        # used to debug
        self.init_0 = self.sigma_0
        self.init_1 = self.sigma_1
        self.init_2 = self.sigma_2
        self.init_state = self.init_z
        self.init_b = np.copy(self.b)

        # Intermediate values to stored for Kalman filter and smoother computations
        self.mu_filter = np.zeros((self.num_patients, self.T)) # mu_t|t
        self.sigma_filter = np.zeros((self.num_patients, self.T)) # sigma^2_t|t
        self.kgain = np.zeros((self.num_patients, self.T)) # K_t, kalman gain
        self.jgain = np.zeros((self.num_patients, self.T)) # J_t, backward kalman gain 
        self.mu_smooth = np.zeros((self.num_patients, self.T)) # mu_t|T
        self.sigma_smooth = np.zeros((self.num_patients, self.T)) # sigma^2_t|T
        self.mu_square_smooth = np.zeros((self.num_patients, self.T)) # E[z_t^2|{y}]
        self.mu_ahead_smooth = np.zeros((self.num_patients, self.T)) # E[z_t * z_{t-1}|{y}]
        self.sigma_ahead_smooth = np.zeros((self.num_patients, self.T)) # V[t, t-1 | T]
        self.mu_pred = np.zeros((self.num_patients, self.T)) # mu_t-1|t
        self.sigma_pred = np.zeros((self.num_patients, self.T)) # sigma^2_t-1|t
        
        # used to debug
        self.expected_log_lik = []
        self.obs_log_lik = []
        self.mse = []
        self.params = {}
    
    # find the last non-nan y value for training for each patient
    # this is necessary since the time series of each patient has different length
    # they are stored in one 2d array, so there would be nans after the last observation for each patient
    # return an array with the length num_patients
    def find_last_obs(self):
        last_non_nan = np.zeros(self.y.shape[0], dtype=int)
        for i in range(self.y.shape[0]):
            last_non_nan[i] = np.where(np.invert(np.isnan(self.y[i, :])))[0][-1] + 1
        return last_non_nan
    
    # find the position of the last observation used for training
    # return the index right after that position
    def find_last_train_obs(self):
        last_train_obs = np.zeros(self.y.shape[0], dtype=int)
        for i in range(self.y.shape[0]):
            non_nan_idx = np.where(np.invert(np.isnan(self.y[i, :])))[0]
            last_train_obs[i] = non_nan_idx[int(non_nan_idx.shape[0] * self.train_pct) - 1] + 1
        return last_train_obs

    # helper function to find the index of valid (not nan) inr measurement for 
    # patient n, return the indices and the corresponding inr values
    def find_valid_inr(self):
        valid_inr = []
        for n in range(self.num_patients):
            inr_index = np.where(np.invert(np.isnan(self.y[n, :self.last_train_obs[n]])))[0]
            inr = self.y[n, inr_index]
            valid_inr.append((inr_index, inr))
        return valid_inr
    
    # compute the added effect, denoted pi_t, at time t given the current parameter values 
    def added_effect(self, n, t):
        treatment_effect = 0
        if self.single_effect:
            if t >= self.J:
                treatment_effect = np.dot(self.A, self.X[n, t-self.J, :])
            elif self.X_prev_given:
                treatment_effect = np.dot(self.A, self.X_prev[n, -self.J, :])
        else:
            for j in range(self.J):
                if t >= j+1:
                    treatment_effect += np.dot(self.A[j, :], self.X[n, t-(j+1), :])
                elif self.X_prev_given:
                    treatment_effect += np.dot(self.A[j, :], self.X_prev[n, -(j+1), :])
        pi = treatment_effect + np.dot(self.b, self.c[n, :]) # total added effect
        return pi
    
    '''E Step Calculations'''
    # kalman filter update step
    def kfilter(self, n, t):
        self.mu_pred[n, t] = self.mu_filter[n, t-1] 
        self.sigma_pred[n, t] = self.sigma_filter[n, t-1] + self.sigma_1
        # updates for when observation is missing
        # taken from Durbin & Koopman textbook
        if np.isnan(self.y[n, t]):
            self.mu_filter[n, t] = self.mu_pred[n, t]
            self.sigma_filter[n, t] = self.sigma_pred[n, t]
        else:
            #sigma_pred = self.sigma_filter[n, t] + self.sigma_1
            self.kgain[n, t] = self.sigma_pred[n, t] / (self.sigma_pred[n, t] + self.sigma_2)
            self.mu_filter[n, t] = self.mu_pred[n, t] + self.kgain[n, t] * (self.y[n, t] - self.mu_pred[n, t] - self.added_effect(n, t))
            self.sigma_filter[n, t] = (1 - self.kgain[n, t]) * self.sigma_pred[n, t]
    
    # kalman filter for each time point, message passing forward
    def forward(self):
        for n in range(self.num_patients):
            # initialization and the first iteration
            self.mu_pred[n, 0] = self.init_z #self.mu_filter[n, 0] = self.init_z
            self.sigma_pred[n, 0] = self.sigma_0 #self.sigma_filter[n, 0] = self.sigma_0
            self.kgain[n, 0] = self.sigma_pred[n, 0] / (self.sigma_pred[n, 0] + self.sigma_2)
            self.mu_filter[n, 0] = self.mu_pred[n, 0] + self.kgain[n, 0] * (self.y[n, 0] - self.mu_pred[n, 0] - self.added_effect(n, 0))
            self.sigma_filter[n, 0] = (1 - self.kgain[n, 0]) * self.sigma_pred[n, 0]
            for t in range(1, self.last_train_obs[n]):
                self.kfilter(n, t)

    # kalman smoother update step
    def ksmoother(self, n, t):
        #sigma_pred = self.sigma_filter[n, t] + self.sigma_1 # sigma^2_t+1|t
        self.jgain[n, t] = self.sigma_filter[n, t] / self.sigma_pred[n, t+1]
        self.mu_smooth[n, t] = self.mu_filter[n, t] + self.jgain[n, t] * (self.mu_smooth[n, t+1] - self.mu_pred[n, t+1])
        self.sigma_smooth[n, t] = self.sigma_filter[n, t] + np.square(self.jgain[n, t]) * (self.sigma_smooth[n, t+1] - self.sigma_pred[n, t+1])
        self.mu_square_smooth[n, t] = self.sigma_smooth[n, t] + np.square(self.mu_smooth[n, t])
    
    # backwards message passing
    def backward(self):
        for n in range(self.num_patients):
            self.mu_smooth[n, self.last_train_obs[n]-1] = self.mu_filter[n, self.last_train_obs[n]-1]
            self.sigma_smooth[n, self.last_train_obs[n]-1] = self.sigma_filter[n, self.last_train_obs[n]-1]
            self.mu_square_smooth[n, self.last_train_obs[n]-1] = self.sigma_smooth[n, self.last_train_obs[n]-1] + \
                np.square(self.mu_smooth[n, self.last_train_obs[n]-1])
            for t in range(self.last_train_obs[n]-2, -1, -1):
                self.ksmoother(n, t)
    
    # backward recursion to compute sigma^2_{t, t-1}|T, which is necessary to compute mu_ahead_smooth
    def backward_sigma_ahead(self):
        for n in range(self.num_patients):
            initial_time = self.last_train_obs[n]-2
            self.sigma_ahead_smooth[n, initial_time] = (1 - self.kgain[n, initial_time+1]) * self.sigma_filter[n, initial_time]
            for t in range(self.last_train_obs[n]-3, -1, -1):
                self.sigma_ahead_smooth[n, t] = self.sigma_filter[n, t+1] * self.jgain[n, t] + \
                    self.jgain[n, t+1] * (self.sigma_ahead_smooth[n, t+1] - self.sigma_filter[n, t+1]) * self.jgain[n, t]
    
    # compute P_t_t-1
    def calc_mu_ahead_smooth(self):
        for n in range(self.num_patients):
            for t in range(self.last_train_obs[n]-1):
                self.mu_ahead_smooth[n, t] = self.sigma_ahead_smooth[n, t] + self.mu_smooth[n, t] * self.mu_smooth[n, t+1]
    
    def E_step(self):
        self.forward()
        self.backward()
        self.backward_sigma_ahead()
        self.calc_mu_ahead_smooth()
    
    '''M Step Calculations'''
    # M step updates for the initial state variance
    def sigma_0_mle(self):
        result = 0
        for n in range(self.num_patients):
            # written in this form instead of subbing in the M step updates for init_z
            # because for the population parameter init_z is not going to be the same as each patient's 
            # first smoothened state
            result += self.mu_square_smooth[n, 0] -2*self.mu_smooth[n, 0]*self.init_z + np.square(self.init_z)
        result /= self.num_patients
        self.sigma_0 = result

    # M step updates for the transition variance
    def sigma_1_mle(self):
        '''
        result = 0
        num_sum = 0
        '''
        numerator = 0
        denominator = 0
        for n in range(self.num_patients):
            # if a patient only has one observation, the transition term doesn't appear in its likelihood
            # so when calculating sigma 1, we should only include those who have more than one observations
            if self.last_train_obs[n] > 1:
                '''
                sum_result = np.sum(np.delete(self.mu_square_smooth[n, :self.last_train_obs[n]]
                    +np.roll(self.mu_square_smooth[n, :self.last_train_obs[n]], shift=-1), -1) \
                    - 2*self.mu_ahead_smooth[n, :self.last_train_obs[n]-1])
                result += sum_result / (self.last_train_obs[n]-1)
                num_sum += 1
                '''
        #result /= num_sum
                numerator += np.sum(np.delete(self.mu_square_smooth[n, :self.last_train_obs[n]]
                    +np.roll(self.mu_square_smooth[n, :self.last_train_obs[n]], shift=-1), -1) \
                    - 2*self.mu_ahead_smooth[n, :self.last_train_obs[n]-1])
                denominator += self.last_train_obs[n] - 1
        self.sigma_1 = numerator / denominator

    # M step update for the initial state mean
    def init_z_mle(self):
        result = 0
        for n in range(self.num_patients):
            result += self.mu_smooth[n, 0]
        result /= self.num_patients
        self.init_z = result


    # mle updates for the coefficients in A
    # derived by setting gradient wrt the coefficients to zero
    def A_mle(self):
        for j in range(self.J):
            for treatment in range(self.N):
                result = 0 # storing the result of the summation
                divisor = 0 # storing the value of the sum in the denominator
                for n in range(self.num_patients):
                    inr_index, inr = self.valid_inr[n]
                    extra = np.zeros_like(inr)
                    x_t = np.zeros_like(inr)
                    for i, t in enumerate(inr_index):
                        extra[i] = self.added_effect(n, t)
                        # for each time point w/ a measurement
                        # x_t[i] stores whether this treatment is given at the j+1 past time point
                        if t >= j+1:
                            x_t[i] = self.X[n, t-(j+1), treatment]
                        elif self.X_prev_given:
                            x_t[i] = self.X_prev[n, -(j+1), treatment]
                        # extra[i] stores the effect of this treatment at the j+1 past time point
                        extra[i] -= self.A[j, treatment] * x_t[i]
                    result += np.sum(np.multiply(inr-self.mu_smooth[n, inr_index]-extra, x_t))
                    divisor += np.sum(np.square(x_t))
                # none of the effect is contributed by this coefficient
                # then set it to zero
                if divisor == 0:
                    self.A[j, treatment] = 0
                else:
                    self.A[j, treatment] = result / divisor

    # mle updates for the coefficients in b
    # derived similarly as A_mle
    def b_mle(self):
        for m in range(self.M):
            result = 0
            divisor = 0 # number of times the term involving b_i is included in the sum
            for n in range(self.num_patients):
                inr_index, inr = self.valid_inr[n]
                extra = np.zeros_like(inr)
                for i, t in enumerate(inr_index):
                    extra[i] = self.added_effect(n, t)
                    extra[i] -= self.b[m] * self.c[n, m]
                result += self.c[n, m]*np.sum(inr-self.mu_smooth[n, inr_index]-extra)
                divisor += np.square(self.c[n, m]) * inr_index.shape[0]
            if divisor == 0:
                self.b[m] = 0
            else:
                self.b[m] = result / divisor

    # M step updates for the observation variance
    def sigma_2_mle(self):
        #result = 0
        numerator = 0
        denominator = 0
        for n in range(self.num_patients):
            inr_index, inr = self.valid_inr[n]
            pi = np.zeros_like(inr)
            for i, t in enumerate(inr_index):
                pi[i] = self.added_effect(n, t)
            '''  
            sum_result = np.sum(np.square(inr-pi)-2*np.multiply(inr-pi, self.mu_smooth[n, inr_index])+self.mu_square_smooth[n, inr_index]) 
            result += sum_result / inr_index.shape[0]
            '''
        #result /= self.num_patients
            #numerator += np.sum(np.square(inr-pi)-2*np.multiply(inr-pi, self.mu_smooth[n, inr_index])+self.mu_square_smooth[n, inr_index]) 
            numerator += np.sum(np.square(inr-pi-self.mu_smooth[n, inr_index])+self.sigma_smooth[n, inr_index])
            denominator += inr_index.shape[0]
        self.sigma_2 = numerator / denominator
    
    # testing
    def intercept_mle(self):
        numerator = 0
        denominator = 0
        for n in range(self.num_patients):
            inr_index, inr = self.valid_inr[n]
            pi = np.zeros_like(inr)
            for i, t in enumerate(inr_index):
                pi[i] = self.added_effect(n, t)      
            numerator += np.sum(inr-pi-self.mu_smooth[n, inr_index])
            denominator += inr_index.shape[0]
        self.intercept = numerator / denominator

    def M_step(self):
        self.init_z_mle()
        self.sigma_0_mle()
        self.sigma_1_mle()
        self.A_mle()
        self.b_mle()
        self.sigma_2_mle()
        #self.intercept_mle()
        
    '''Run EM for fixed iterations or until paramters converge'''
    def run_EM(self, max_num_iter, tol=.001):
        old_ll = -np.inf
        old_params = np.full(self.J*self.N+self.M+5, np.inf)
        for i in range(max_num_iter):
            print('iteration {}'.format(i+1))
            #t0 = time.time()
            self.E_step()
            #print('E step {}'.format(self.expected_complete_log_lik()))
            #t1 = time.time()
            self.M_step()
            #print('M step {}'.format(self.expected_complete_log_lik()))
            #self.expected_log_lik.append(self.expected_complete_log_lik())
            #t2 = time.time()
            new_ll = self.pykalman_log_lik()
            #t3 = time.time()
            '''
            print('E step took {}'.format(t1-t0))
            print('M step took {}'.format(t2-t1))
            print('calculating loglik took {}'.format(t3-t2))
            '''
            self.obs_log_lik.append(new_ll)
            #self.expected_log_lik.append(self.expected_complete_log_lik())

            if np.abs(new_ll - old_ll) < tol:
                print('{} iterations before loglik converges'.format(i+1))
                return i+1
            old_ll = new_ll

            # for faster training convergence, stop iterations when parameters stop changing
            new_params = np.concatenate([self.A.flatten(), self.b, np.array([self.init_z, self.sigma_0, self.sigma_1, self.sigma_2, self.intercept])])
            if np.max(np.absolute(new_params-old_params))<tol:
                print('{} iterations before params converge'.format(i+1))
                return i+1
            old_params = new_params
            
            # keep a list of values of each param for each iteration to debug mse 
            if i == 0:
                for j in range(len(new_params)):
                    self.params[j] = []
            for j, param in enumerate(new_params):
                self.params[j].append(param)
            
            self.mse.append(self.get_MSE())

        print('max iterations: {} reached'.format(max_num_iter))
        return max_num_iter

    # transition function used in prediction 
    def transition(self, prev, n, t):
        noise = self.sigma_filter[n, t-1] + self.sigma_1
        self.sigma_filter[n, t] = noise
        z = prev #np.random.normal(prev, np.sqrt(noise), 1)
        return z

    # emission function used in prediction
    def emission(self, z, n, t):
        mean = z + self.added_effect(n, t)
        y = mean #np.random.normal(mean, np.sqrt(self.sigma_2), 1)
        return y
    
    # given parameters and a sequence latent states up to the last training observation
    # predict the latent state (z) and observation (y) up to the last observation for a particular patient (n)
    def predict(self, n):
        y = np.zeros(self.last_obs[n])
        z = np.zeros(self.last_obs[n])
        y[:self.last_train_obs[n]] = self.y[n, 0:self.last_train_obs[n]]
        z[:self.last_train_obs[n]] = self.mu_smooth[n, 0:self.last_train_obs[n]]
        z[self.last_train_obs[n]] = self.transition(z[self.last_train_obs[n]-1], n, self.last_train_obs[n])
        y[self.last_train_obs[n]] = self.emission(z[self.last_train_obs[n]], n, self.last_train_obs[n])
        for t in range(self.last_train_obs[n]+1, self.last_obs[n]):
            z[t] = self.transition(z[t-1], n, t)
            y[t] = self.emission(z[t], n, t)
        return z, y

    # get prediction mean square error 
    def get_MSE(self):
        #self.sos = []
        sum_of_square = 0
        count = 0
        for n in range(self.num_patients):
            if self.last_train_obs[n] < self.last_obs[n]:
                y_true = self.y[n, self.last_train_obs[n]:self.last_obs[n]]
                y_pred = self.predict(n)[1][self.last_train_obs[n]:self.last_obs[n]]
                valid_index = np.where(np.invert(np.isnan(y_true)))[0]
                y_true_valid = y_true[valid_index]
                y_pred_valid = y_pred[valid_index]
                sum_of_square += np.sum(np.square(np.subtract(y_pred_valid, y_true_valid))) / y_pred_valid.shape[0]
                count += 1
                #self.sos.append(np.sum(np.square(np.subtract(y_pred_valid, y_true_valid))) / y_pred_valid.shape[0])
        if count > 0:
            return sum_of_square / count
        else:
            return 0

    # the log lik function used by pykalman
    # used to check for log likelihood convergence
    def pykalman_log_lik(self):
        total_log_lik = 0
        # the prediction mean and variance using the current (not previous) parameters
        # since kalman filter is run afresh every em iteration, this doesn't affect em outputs
        self.forward()
        for n in range(self.num_patients):
            inr_index, inr = self.valid_inr[n]
            log_lik = np.zeros_like(inr)
            for i, index in enumerate(inr_index):
                log_lik[i] = scipy.stats.norm.logpdf(self.y[n, index], self.mu_pred[n, index]+self.added_effect(n, index),
                    np.sqrt(self.sigma_pred[n, index]+self.sigma_2))
                #log_lik[i] = scipy.stats.norm.logpdf(self.y[n, index], self.mu_smooth[n, index], np.sqrt(self.sigma_smooth[n, index]+self.sigma_2))
            total_log_lik += np.sum(log_lik)
        return total_log_lik

    # calculate the expected complete data log likelihood 
    # used for testing
    def expected_complete_log_lik(self):
        log_lik = 0
        log_sigma_0 = -self.num_patients * np.log(self.sigma_0)/2
        log_lik += log_sigma_0
        for n in range(self.num_patients):
            inr_index, inr = self.valid_inr[n]
            log_sigma_1 = -(self.last_train_obs[n]-1)/2*np.log(self.sigma_1)
            log_sigma_2 = -inr_index.shape[0]/2*np.log(self.sigma_2)
            first_term = -1/(2*self.sigma_0)*(self.mu_square_smooth[n, 0]-2*self.init_z*self.mu_smooth[n, 0]+np.square(self.init_z))
            second_term = -1/(2*self.sigma_1)*np.sum(np.delete(self.mu_square_smooth[n, :self.last_train_obs[n]]+np.roll(self.mu_square_smooth[n, :self.last_train_obs[n]], shift=-1), -1) \
                - 2*self.mu_ahead_smooth[n, :self.last_train_obs[n]-1])
            pi = np.zeros_like(inr)
            for i, t in enumerate(inr_index):
                pi[i] = self.added_effect(n, t)
            third_term = -1/(2*self.sigma_2)*np.sum(np.square(inr-pi)-2*np.multiply(inr-pi, self.mu_smooth[n, inr_index])+self.mu_square_smooth[n, inr_index]) 
            log_lik += log_sigma_1 + log_sigma_2 + first_term + second_term + third_term
        return float(log_lik)