# Counterfactual-Prediction

Create counterfactual trajectory of a signal given past trajectory and some contexts.

### Files Description
* PDF file
    * Detailed write up of the model
* Python files
    * EM.py: perform inference and parameter estimation
    * plot.py: plot predicted states and observations
    * EM_individual_params: EM for learning individual parameters (incomplete)
    * preprocess.py: preprocess data into format usable for EM.py (more general version is in Pipeline_Preprocessing.py)
* Notebook files
    * Pipeline_Preprocessing: extract relevant data from database files; preprocess; perform inference, parameter estimation and prediction
    * Pipeline_Training: given preprocessed file, perform training (use to experiment w/ training procedure) 
    * DLM: contain simulated data to evaluate model's performance
    * Population Level Analysis: assume the same set of parameters for all patients; fit to real data
    * Individual Level Analysis: assume a set of parameters for each patients; fit to real data
    * Source Data Analysis: some analysis on the real data set before and after preprocessing
