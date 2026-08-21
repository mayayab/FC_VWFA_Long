#!/usr/bin/env python
# coding: utf-8

# Hyperparameter tuning for classifier models

from nilearn import surface
import numpy as np
import pandas as pd
import nibabel as nib
from os.path import exists
import matplotlib.pyplot as plt
import matplotlib
import glob
import os
import sys
import seaborn as sns
import re
import time
from scipy.stats import randint, uniform
import joblib
import platform


# import sklearn
from sklearn import metrics

# XGBoost
from xgboost import XGBClassifier
# optimize hyperparameter search
from skopt import BayesSearchCV
from skopt.space import Real, Integer, Categorical



# Set parameters:
wPredictor = 'Atlas' # Options are 'IFC' or 'Atlas'
wModel = 'XGB' # 'XGB' for XGBoost
# number of iterations and folds for cross-validation within training data
num_iter = 50; num_cv = 3
doScale = False # should predictors be scaled
zScore = True # should data be zscored within subject/session

# _____Set all paths ________#

basepath = '<PROJECT_PATH/>'

space = 'fsaverage' # 'fsaverage' or 'fsnative'
workpath = f'{basepath}FC_analysis/'
inputdir = f'{workpath}cleaned_{space}_GSR/'
outputdir = inputdir + 'Predict/' 
if not os.path.exists(outputdir):
    os.makedirs(outputdir)
print(inputdir) 

sessions = ['ses-1','ses-2','ses-3','ses-4','ses-5']

# Load data

# Load data for classifer - y is binary
train_y = np.load(f'{outputdir}train_y.npy')
test_y_sessions = np.load(f'{outputdir}test_y_sessions.npy',allow_pickle = True)
    
if zScore:
    train_x = np.load(f'{outputdir}train_x_feat_201_zscored.npy')
    test_x_sessions = np.load(f'{outputdir}test_x_feat_201_sessions_zscored.npy',allow_pickle = True)
else:
    train_x = np.load(f'{outputdir}train_x_feat_201.npy')
    test_x_sessions = np.load(f'{outputdir}test_x_feat_201_sessions.npy',allow_pickle = True)


# filter to a single predictor
if wPredictor == 'IFC':
    train_x = train_x[:,200]
    train_x = train_x.reshape(-1, 1)
    for ses_id, ses in enumerate(sessions):
        test_x_sessions[ses_id] = test_x_sessions[ses_id][:,200]
        test_x_sessions[ses_id] = test_x_sessions[ses_id].reshape(-1, 1)



# Create initial model and search space
    
if wModel == 'XGB':
# calculate ratio of ones / zeros - since the data is highly imbalanced we need to scale by that ratio
    positive_rate = train_y.mean()
    print(positive_rate)
    scale_pos_weight = (1 - positive_rate) / positive_rate
    print(f'Scale positive weights: {scale_pos_weight}')

    # Baseline XGBoost model
    clf = XGBClassifier(
        objective='binary:logistic',
        eval_metric='aucpr',
        n_estimators=1000,
        learning_rate=0.03,
        max_depth=10,
        min_child_weight=3,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        reg_alpha=0.1,
        reg_lambda=2.0,
        tree_method='hist',
        n_jobs=-1)
    
    # define search space for XGBoost hyper parameters
    param_dist = {

    # boosting
    'learning_rate': Real(0.01, 0.2, prior='log-uniform'),
    'n_estimators': Integer(300, 2000),
    # tree complexity
    'max_depth': Integer(3, 10),
    'min_child_weight': Integer(1, 10),
    # row/feature sampling
    'subsample': Real(0.6, 1.0),
    'colsample_bytree': Real(0.6, 1.0),
    # regularization
    'reg_alpha': Real(1e-3, 10.0, prior='log-uniform'),
    'reg_lambda': Real(1e-2, 10.0, prior='log-uniform'),
    # weights
    'scale_pos_weight': Real(scale_pos_weight / 4, scale_pos_weight * 4,prior='log-uniform'),
    # split regularization
    'gamma': Real(0, 10)}


# Bayesian search
# Search hyperparameters

start_time = time.time()

if wModel == 'XGB'
    # Bayesian search
    rand_search = BayesSearchCV(
      clf, search_spaces=param_dist,
      n_iter=num_iter, cv=num_cv, scoring='f1',
      n_jobs=-1, random_state=0,verbose=2,refit='f1')

    # fit the model to search for hyperparameters:
    rand_search.fit(train_x, train_y)

    print("Best params (by F1):", rand_search.best_params_)
    print("Best F1 score:", rand_search.best_score_)

    
end_time = time.time()
print(f"Execution time: {end_time - start_time} sec")

# Get the results
results = pd.DataFrame(rand_search.cv_results_)

# Create a variable for the best model
best_model = rand_search.best_estimator_

# # Print the best hyperparameters
print('Best hyperparameters:',  rand_search.best_params_)
print('Best Cross Validated Score:', rand_search.best_score_)

# save
fname = f'BayesSearch_{wModel}_{wPredictor}_iter{num_iter}_cv{num_cv}'
if zScore:
    fname = f'{fname}_z'

search_fname = f'{outputdir}{fname}_Results.pkl'
best_model_fname = f'{outputdir}{fname}_Best_Model.pkl' 
res_fname = f'{outputdir}{fname}_Best_Model.csv' 
print(res_fname)

results.to_csv(res_fname)
joblib.dump(best_model, best_model_fname) 
joblib.dump(rand_search, search_fname) 

