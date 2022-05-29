# -*- coding: utf-8 -*-


import pandas as pd
import numpy as np
import json
import pickle
from sklearn.impute import KNNImputer
from sklearn.ensemble import IsolationForest



def encode_categorical(root_folder,dataset):
   
    # load values to imput
    with open(root_folder+'\\conf\\data_processing\\encode_categorical.json', 'r') as fp:
        missing_imputation = json.load(fp)
    
    cols_to_imputate = list(missing_imputation)
    
    # encode categorical cols with values from dict
    dataset.update(dataset[cols_to_imputate].apply(lambda col: col.map(missing_imputation[col.name])))
    dataset[cols_to_imputate] = dataset[cols_to_imputate].astype('float')
    
    # encode other categorical cols, except index cols
    cat_cols = [col for col in dataset.columns if dataset[col].dtype == 'object' and col not in ['CUSTOMER_ID','YEAR_MONTH']]
    dataset = pd.get_dummies(data=dataset, columns=cat_cols, dummy_na=False)
    
    return dataset
    


def preprocessing(root_folder,path_to_dataset):
    
    dataset = pd.read_csv(root_folder+path_to_dataset)
    
    print('Loading data...\n')
    
    # encode target
    dataset['Churn_Flag'] = np.where(dataset['Churn_Flag'].str.lower() == 'attrited customer', 1, 0)
    
    # replace invalid strings with null
    dataset = dataset.replace(['Unknown'], np.nan)
    
    # encode categorical variables
    dataset = encode_categorical(root_folder,dataset)
    
    return dataset



def knn_fillnan(root_folder,trainset, validset, testset):
    
    # copy datasets
    trainset_temp = trainset.copy()
    validset_temp = validset.copy()
    testset_temp = testset.copy()
    
    # initializing KNNImputer
    knn = KNNImputer(n_neighbors = 10)
    
    # fit model
    knn.fit(trainset_temp)
    
    # save model for future imputation
    pickle.dump(knn, open(root_folder+'\\models\\data_processing\\knnImputer.sav', 'wb'))
    
    # transform datasets
    X_train = knn.transform(trainset_temp)
    X_valid = knn.transform(validset_temp)
    X_test = knn.transform(testset_temp)
    X_train = pd.DataFrame(X_train, index=trainset.index, columns = trainset.columns.values)
    X_valid = pd.DataFrame(X_valid, index=validset.index, columns = validset.columns.values)
    X_test = pd.DataFrame(X_test, index=testset.index, columns = testset.columns.values)
    
    # identify numerical columns (less than 10 unique values)
    num_cols = list(trainset.loc[:, trainset.nunique() < 10].columns.values)
    
    # fillna only for numerical columns
    trainset_temp[num_cols] = trainset_temp[num_cols].fillna(X_train[num_cols])
    validset_temp[num_cols] = validset_temp[num_cols].fillna(X_valid[num_cols])
    testset_temp[num_cols] = testset_temp[num_cols].fillna(X_test[num_cols])
    
    return trainset_temp, validset_temp, testset_temp



def fill_missing_values(root_folder,trainset, validset, testset):
    
    # fillna for numerical/continuous columns
    trainset_temp, validset_temp, testset_temp = knn_fillnan(root_folder,trainset, validset, testset)
    
    # identify discrete columns
    discrete_cols = list(trainset.loc[:, trainset.nunique() >= 10].columns.values)
    
    # compute mode based on trainset
    mode_values = trainset.dropna().mode().iloc[0].to_dict()
    
    # save values for future imputation
    with open(root_folder+'\\conf\\data_processing\\missing_imputation_discrete_cols.json', 'w') as fp:
        json.dump(mode_values, fp) 
    
    # fillna with mode for discrete columns
    for col in discrete_cols:
        trainset_temp.loc[trainset_temp[col].isna(), col] = mode_values[col]
        validset_temp.loc[validset_temp[col].isna(), col] = mode_values[col]
        testset_temp.loc[testset_temp[col].isna(), col] = mode_values[col]
    
    print('Filling missing values...\n')
    
    return trainset_temp, validset_temp, testset_temp
    
    

def remove_outliers(dataset, contamination=0.05):
    
    iso = IsolationForest(contamination=contamination)
    yhat = iso.fit_predict(dataset.values)
    
    print('Removing outliers...\n')
    
    return yhat == 1