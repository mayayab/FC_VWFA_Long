#!/usr/bin/env python
# coding: utf-8

# Run models to predict VWFA2
# Run permutation test to evaluate prediction significance

# General imports
from nilearn import surface
import numpy as np
import pandas as pd
import nibabel as nib
from os.path import exists
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
from scipy import stats
from scipy.stats import randint, uniform


# import sklearn
from sklearn import metrics
from sklearn.pipeline import Pipeline


# Set parameters:
wPredictor = 'IFC' # Options are 'IFC' or 'Atlas'
model_type = 'XGB' # 'XGB' for XGBoost
num_iter = 50; num_cv = 3
perm_iter = 100000 # iterations for random permutations
threshold = 0.55
zScore = True
wPermute = 'subwise' # 'all'- permute vector vertex-wise. 'subwise'- permute by subjects, more restrictive
    
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
if zScore:
    train_x = np.load(f'{outputdir}train_x_feat_201_zscored.npy')
    train_y = np.load(f'{outputdir}train_y.npy')
    
    test_x_sessions = np.load(f'{outputdir}test_x_feat_201_sessions_zscored.npy',allow_pickle = True)
    test_y_sessions = np.load(f'{outputdir}test_y_sessions.npy',allow_pickle = True)
else:
    train_x = np.load(f'{outputdir}train_x_feat_201.npy')
    train_y = np.load(f'{outputdir}train_y.npy')
    
    test_x_sessions = np.load(f'{outputdir}test_x_feat_201_sessions.npy',allow_pickle = True)
    test_y_sessions = np.load(f'{outputdir}test_y_sessions.npy',allow_pickle = True)


if wPredictor == 'IFC':
    # For the IFC alone select the corresponding IFC vector from the matrix
    train_x = train_x[:,200]
    train_x = train_x.reshape(-1, 1)
    for ses_id, ses in enumerate(sessions):
        test_x_sessions[ses_id] = test_x_sessions[ses_id][:,200]
        test_x_sessions[ses_id] = test_x_sessions[ses_id].reshape(-1, 1)


# Load model:
# This is the model that was found to be the best using the Bayesian search
model_name = f'BayesSearch_{model_type}_{wPredictor}_iter{num_iter}_cv{num_cv}'
if zScore:
    model_name = f'{model_name}_z'
    
model_fname = f'{outputdir}{model_name}_Best_Model.pkl'
print(f"Loading optimized model for {wPredictor}")
print(model_fname)
clf = joblib.load(model_fname)
search_res_fname = f'{outputdir}{model_name}_Results.pkl'
print(search_res_fname)
search_res = joblib.load(search_res_fname)

# Fit the model

start_time = time.time()

trained_model_name = f'{model_fname}_trained_model.pkl'
if os.path.exists(trained_model_name):
    print(f"Loading trained model: {trained_model_name}")
    clf = joblib.load(trained_model_name)
else:
    print(f"Training the model: {model_fname}")
    clf.fit(train_x, train_y)

end_time = time.time()
print(f"Execution time: {end_time - start_time} sec")
print(f"{(end_time - start_time)/60} min")

# Get the search space dimensions for the permutations
probmap_path = f'{workpath}Group_FuncMaps/'
probmap_file = f'{probmap_path}Group_all_VWFA2_allses_overlap.curv'
probmap = surface.load_surf_data(probmap_file)
#vwfa_space is all the vertices that could potentially be vwfa2
vwfa_space = np.where(probmap>0)[0]
print(len(vwfa_space))

# Random shuffle test_x for 1000 iterations
start_time = time.time()
# We are interested in predicting from ses-2 data (pre-intervention)
ses_id = 1; ses = 'ses-2'

# create null distribution of predictions, compare the actual prediction to the permuted distribution
# shuffle vertex wise - oblivious to subjects
if wPermute == 'all':

    test_x = test_x_sessions[ses_id]
    test_y = test_y_sessions[ses_id]

    f_dist_px = np.empty(perm_iter)
    prauc_dist_px = np.empty(perm_iter)
    
    # Run the model to get the prediction
    prob_y = clf.predict_proba(test_x)[:, 1]
    pred_y = (prob_y > threshold).astype(int)

    # calculate actual metrics
    f_score = metrics.f1_score(test_y,pred_y)       
    PRAUC_score = metrics.average_precision_score(test_y,prob_y)

    for iter_id in range(perm_iter):
        permuted_y = np.random.permutation(test_y)

        if len(permuted_y.shape) ==1 :
            permuted_y = permuted_y.reshape(-1, 1)

        #prob_permuted_y = clf.predict_proba(permuted_x)[:, 1]
        #pred_permuted_y = (prob_permuted_y > threshold).astype(int)
        #f_score = metrics.f1_score(test_y,pred_permuted_y)
        #PRAUC_score = metrics.average_precision_score(test_y,prob_permuted_y)

        f_score_iter = metrics.f1_score(permuted_y,pred_y)
        PRAUC_score_iter = metrics.average_precision_score(permuted_y,prob_y)

        f_dist_px[iter_id] = f_score_iter
        prauc_dist_px[iter_id] = PRAUC_score_iter
        #print(f_score)
    


# Random shuffle test_x for 1000 iterations
# create null distribution of predictions, compare the actual prediction to the permuted distribution
# Here we shuffle the participants, without shuffling the values within each participant

if wPermute == 'subwise':
    
    test_x = test_x_sessions[ses_id]
    test_y = test_y_sessions[ses_id]

    # Run the model to get the prediction
    prob_y = clf.predict_proba(test_x)[:, 1]
    pred_y = (prob_y > threshold).astype(int)

    # calculate actual metrics
    f_score = metrics.f1_score(test_y,pred_y)       
    PRAUC_score = metrics.average_precision_score(test_y,prob_y)
    
    print(ses)
    print(f'F score: {f_score}')
    print(f'PRAUC score: {PRAUC_score}')

    f_dist_px = np.empty(perm_iter)
    prauc_dist_px = np.empty(perm_iter)

    # how many subjects in this session- needed to permute subjects
    n_subs = test_y.shape[0]/vwfa_space.shape[0]
    n_subs = int(n_subs)

    # split the y - vector into subject blocks
    test_y_subs = test_y.reshape(n_subs, vwfa_space.shape[0])
    print(test_y_subs.shape)

    for iter_id in range(perm_iter):

        # permute the subjects
        perm = np.random.permutation(n_subs) # random order
        test_y_subs_perm = test_y_subs[perm] # reorganize by random order

        # flatten back into a vector
        perm_test_y = test_y_subs_perm.ravel()

        # we want to see that the f1 score between the predicted y and the real test y 
        # is greater than the f1 between the predicted y and any other shuffled y
        # this means that each subject predicts themselved more than they do any other subject
        f_score_iter = metrics.f1_score(perm_test_y,pred_y)       
        PRAUC_score_iter = metrics.average_precision_score(perm_test_y,prob_y)

        f_dist_px[iter_id] = f_score_iter
        prauc_dist_px[iter_id] = PRAUC_score_iter
        #print(f_score)
    
        

end_time = time.time()
print(f'Run time: {(end_time - start_time)/60} min')

my_score = f_score

# calcuate p
print("p-val:")
print(len(prauc_dist_px[prauc_dist_px>my_score])/len(prauc_dist_px))

# save for future plotting without rerunning the permutation test
res_name = f'{model_type}_{wPredictor}_{threshold}_{wPermute}'
if zScore:
    res_name = f'{res_name}_z'
print(res_name)
print(perm_iter)
np.save(f'{outputdir}{res_name}_{perm_iter}_permutations_f1',f_dist_px)
np.save(f'{outputdir}{res_name}_{perm_iter}_permutations_prauc',prauc_dist_px)