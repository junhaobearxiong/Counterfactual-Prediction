# Counterfactual-Prediction

Create counterfactual trajectory of a signal given past trajectory and some contexts (Please see the pdf file for a more detailed description of the model and the data pipeline)

### Files Description
* PDF file
    * Detailed write up of the model and results
* Python files
    * EM.py: perform inference and parameter estimation
    * plot.py: plot predicted states and observations
* Notebook files
    * Pipeline_Preprocessing: extract relevant data from database files; preprocess; perform inference, parameter estimation and prediction
    * Pipeline_Training: given preprocessed file, perform training, can perform hyperparameter validation and training on multiple signals through parallel computing (using ipyparallel)
    * DLM: simulation to evaluate model's performance
 * Old dev files
    * EM_individual_params: EM for learning individual parameters (incomplete)
    * preprocess.py: preprocess data into format usable for EM.py (more general version is in Pipeline_Preprocessing.py)
     * Population Level Analysis: assume the same set of parameters for all patients; fit to real data
    * Individual Level Analysis: assume a set of parameters for each patients; fit to real data
    * Source Data Analysis: some analysis on the real data set before and after preprocessing
    * Playground: place to do small analysis and testing
