# Counterfactual-Prediction

Create counterfactual trajectory of a signal given past trajectory and some contexts.

### Modeling choices
* Cutoff (only consider the data point with at least certain number of observations): 5 --> 2214 data points, 215 bins
* Bin size (length of time interval used to bin the time series): 18 hrs
* Parameters initialization
    * Both treatment and static coefficients: Gaussian with mean 0.1 plus small white noise
* Missingness
    * After binning the data, only consider the data points with less than 30% missing observations
* Chronic conditions and age
    * Set to zero
