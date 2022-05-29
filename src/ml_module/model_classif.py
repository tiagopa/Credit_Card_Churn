# -*- coding: utf-8 -*-

import pickle
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import roc_auc_score, classification_report


    
def fit_model(parameters, X_train, y_train, X_val=None, y_val=None):
    
    # create booster with defined parameters
    booster = xgb.XGBClassifier(**parameters)

    # fit model
    if X_val is not None:
        model = booster.fit(X_train, y_train, \
                            eval_set=[(X_train, y_train), (X_val, y_val)], \
                            eval_metric='auc', \
                            early_stopping_rounds=30,
                            verbose=0)
    else:
        model = booster.fit(X_train, y_train, \
                            eval_metric='auc', \
                            eval_set=[(X_train, y_train)], \
                            verbose=0)

    return model



def predict_results(model, X, y_true):
    
    # predict probabilities
    estimated_prob = model.predict_proba(X)[:,1]
    
    # auc score
    auc_score = roc_auc_score(y_true,estimated_prob)
    
    return estimated_prob, auc_score



def predict_test(root_folder, model, X, y_true):
    
    print('Predicting churn flag for test dataset...')
    
    output_file = 'xgboost_test_results.csv'
    
    # predict results
    estimated_prob, auc_score = predict_results(model, X, y_true)
    y_pred = np.where(estimated_prob >= 0.5, 1, 0)
    
    # save results
    output_probabilities_df = y_true.reset_index()
    output_probabilities_df['estimated_prob'] = estimated_prob
    output_probabilities_df = output_probabilities_df.rename(columns={'Churn_Flag': 'y_true'})
    output_probabilities_df.to_csv(f'{root_folder}\\results\\{output_file}')
    
    # print results
    print_auc = round(auc_score,3)
    print(f'AUC score: {print_auc}')
    print(classification_report(y_true, y_pred, target_names=['Existing Customer','Attrited Customer']))
    
    # auc score
    auc_score = roc_auc_score(y_true,estimated_prob)
    
    return None



def grid_search_xgboost(X_train,y_train,X_val,y_val,param_grid):
    
    print('Performing gridsearch...')
    
    results = []
    
    # create sklearn parameter grid
    grid = ParameterGrid(param_grid)
    
    # grid search
    for params in grid:
    
        # fit model
        model = fit_model(params, X_train, y_train, X_val, y_val)
        
        # predict and score
        _, auc_train = predict_results(model, X_train,y_train)
        _, auc_val = predict_results(model,X_val,y_val)
        
        # append results
        results.append((auc_train,auc_val,params))
    
    return results
        


def train_model(root_folder,X_train,y_train,X_val,y_val):
    
    gridsearch_output_file = 'grid_search_results.csv'
    model_pickle_file = 'xgboost_classifier.sav'
    
    # define model parameter grid to search
    param_grid = {
                    'min_child_weight': [5, 15, 30],
                    'gamma': [0.5, 1, 1.5, 2, 5],
                    'subsample': [0.6, 0.8, 1.0],
                    'colsample_bytree': [0.6, 0.8, 1.0],
                    'max_depth': [3, 4, 5],
                    'use_label_encoder':[False]
                }

    # perform gridsearch
    gridsearch_results = grid_search_xgboost(X_train,y_train, \
                                             X_val,y_val, \
                                             param_grid)
        
    # create dataframe with results and save
    df_results = pd.DataFrame(gridsearch_results,columns=['train_score','validation_score','hyperparameters'])  
    df_results.to_csv(f'{root_folder}\\results\\{gridsearch_output_file}')
    
    # get parameters that maximize validation score
    best_results = df_results.loc[df_results['validation_score'].idxmax()]
    best_params = best_results.hyperparameters
    auc_train = best_results.train_score
    auc_val = best_results.validation_score
    
    print(f'best parameters found:  {best_params}')
    print(f'AUC score - training:   {auc_train}')
    print(f'AUC score - validation: {auc_val}\n')
    
    print('Training model...\n')
    
    # train model
    model = fit_model(best_params, X_train, y_train, X_val, y_val)
    
    # save model
    pickle.dump(model, open(f'{root_folder}\\models\\{model_pickle_file}', 'wb'))
    
    return model
