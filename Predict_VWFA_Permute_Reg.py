#!/usr/bin/env python
# coding: utf-8

# Run regresstion models to predict text-selectivity
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

# Regression
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import ElasticNet, ElasticNetCV, LinearRegression


# Set parameters:
wPredictor = 'Atlas' # Options are 'IFC' or 'Atlas'
model_type = 'ENet' # ElasticNet
num_iter = 50; num_cv = 3
perm_iter = 100000 # iterations for random permutations
zScore = True # work on zscored or raw data
wPermute = 'subwise' # 'all' or 'subwise'

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

probmap_path = f'{workpath}Group_FuncMaps/'
probmap_file = f'{probmap_path}Group_all_VWFA2_allses_overlap.curv'
outputdir = inputdir + 'Predict/' 
probmap = surface.load_surf_data(probmap_file)
#vwfa_space is all the vertices that could potentially be vwfa2
vwfa_space = np.where(probmap>0)[0]
print(len(vwfa_space))


# Load data
if zScore:
    train_x = np.load(f'{outputdir}train_xt_feat_201_zscored.npy')
    train_y = np.load(f'{outputdir}train_yt_zscored.npy')
    
    test_x_sessions = np.load(f'{outputdir}test_xt_feat_201_sessions_zscored.npy',allow_pickle = True)
    test_y_sessions = np.load(f'{outputdir}test_yt_sessions_zscored.npy',allow_pickle = True)

else: 
    train_x = np.load(f'{outputdir}train_xt_feat_201.npy')
    train_y = np.load(f'{outputdir}train_yt.npy')
    
    test_x_sessions = np.load(f'{outputdir}test_xt_feat_201_sessions.npy',allow_pickle = True)
    test_y_sessions = np.load(f'{outputdir}test_yt_sessions.npy',allow_pickle = True)

if wPredictor == 'IFC':
    # For the IFC alone select the corresponding IFC vector from the matrix
    train_x = train_x[:,200]
    train_x = train_x.reshape(-1, 1)
    for ses_id, ses in enumerate(sessions):
        test_x_sessions[ses_id] = test_x_sessions[ses_id][:,200]
        test_x_sessions[ses_id] = test_x_sessions[ses_id].reshape(-1, 1)


# Regression model:

if model_type == 'ENet':
    clf = Pipeline([
        ("scaler", StandardScaler()),
        ("regressor", ElasticNetCV(
            alphas=[0.01, 0.1, 1.0, 10.0, 100.0]
        ))
    ])


# Fit the model
start_time = time.time()

clf.fit(train_x, train_y)

end_time = time.time()
print(f"Execution time: {end_time - start_time} sec")
print(f"{(end_time - start_time)/60} min")

if model_type == 'ENet':
    print("Chosen alpha:", clf.named_steps["regressor"].alpha_)
    print(f'L1 ratio: {clf.named_steps["regressor"].l1_ratio_}')
    

# Random shuffle test_x for 1000 iterations
# create null distribution of predictions, compare the actual prediction to the permuted distribution
# sort to see where actual value is

# take the same fitted model
start_time = time.time()
# We are interested in predicting from ses-2 data (pre-intervention)
ses_id = 1; ses = 'ses-2'

if wPermute == 'all': 
    

    test_x = test_x_sessions[ses_id]
    test_y = test_y_sessions[ses_id]

    r2_dist_px = np.empty(perm_iter)
    r_dist_px = np.empty(perm_iter)
    RMSE_dist_px = np.empty(perm_iter)

    # Run the model to get the prediction
    pred_y = clf.predict(test_x)

    # calculate actual metrics
    r2_score = metrics.r2_score(test_y,pred_y)       
    RMSE_score = metrics.root_mean_squared_error(test_y,pred_y)
    rval = stats.pearsonr(test_y,pred_y)
    print(ses)
    print(f'R2 score: {r2_score}')
    print(f'RMSE score: {RMSE_score}')
    print(f'r value: {rval}')
    
    for iter_id in range(perm_iter):
        permuted_y = np.random.permutation(test_y)

        cur_r2_score = metrics.r2_score(pred_y,permuted_y)
        cur_r_val = stats.pearsonr(pred_y,permuted_y)[0]
        cur_RMSE_score = metrics.root_mean_squared_error(pred_y,permuted_y)

        r2_dist_px[iter_id] = cur_r2_score
        r_dist_px[iter_id] = cur_r_val
        RMSE_dist_px[iter_id] = cur_RMSE_score
    

end_time = time.time()
print(f'Run time: {(end_time - start_time)/60} min')

# Random shuffle test_x for 1000 iterations
# create null distribution of predictions, compare the actual prediction to the permuted distribution
# shuffle by participants

# take the same fitted model
start_time = time.time()

if wPermute == 'subwise': 

    test_x = test_x_sessions[ses_id]
    test_y = test_y_sessions[ses_id]

    r2_dist_px = np.empty(perm_iter)
    r_dist_px = np.empty(perm_iter)
    RMSE_dist_px = np.empty(perm_iter)

    # Run the model to get the prediction
    pred_y = clf.predict(test_x)

    # calculate actual metrics
    r2_score = metrics.r2_score(test_y,pred_y)       
    RMSE_score = metrics.root_mean_squared_error(test_y,pred_y)
    rval = stats.pearsonr(test_y,pred_y)
    print(ses)
    print(f'R2 score: {r2_score}')
    print(f'RMSE score: {RMSE_score}')
    print(f'r value: {rval}')

    
    n_subs = test_y.shape[0]/vwfa_space.shape[0]
    n_subs = int(n_subs)
    # split y- vector into subject blocks
    test_y_subs = test_y.reshape(n_subs, vwfa_space.shape[0])
    print(test_y_subs.shape)
        
    for iter_id in range(perm_iter):

        # shuffle the subjects
        perm = np.random.permutation(n_subs)
        test_y_subs_perm = test_y_subs[perm]

        # flatten back into a vector
        perm_test_y = test_y_subs_perm.ravel()

        cur_r2_score = metrics.r2_score(pred_y,perm_test_y)   
        cur_r_val = stats.pearsonr(pred_y,perm_test_y)[0]
        cur_RMSE_score = metrics.root_mean_squared_error(pred_y,perm_test_y)

        r2_dist_px[iter_id] = cur_r2_score
        r_dist_px[iter_id] = cur_r_val
        RMSE_dist_px[iter_id] = cur_RMSE_score


end_time = time.time()
print(f'Run time: {(end_time - start_time)/60} min')


my_score = rval

# calcuate p
print("p-val:")
print(len(r_dist_px[r_dist_px>my_score])/len(r_dist_px))

# save so we can later load this and plot without rerunning the permutation test

res_name = f'{model_type}_{wPredictor}_{wPermute}'
if zScore:
    res_name = f'{res_name}_z'
print(res_name)
np.save(f'{outputdir}{res_name}_{perm_iter}_permutations_r2score',r2_dist_px)
np.save(f'{outputdir}{res_name}_{perm_iter}_permutations_rval',r_dist_px)
np.save(f'{outputdir}{res_name}_{perm_iter}_permutations_RMSE',RMSE_dist_px)

