# -*- coding: utf-8 -*-

import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline
from collections import Counter


def dataset_sampling(X,y):
    """Apply oversampling using SMOTE and then random undersampling.
   
    Args:
        X: Dataset with independent variables.
        y: Dependent variable.
    Returns:
        Independent and dependent variables after sampling.
    """
    
    # check proportion of class 1
    proport_class1 = round(y.mean()*100,1)
    print('Sampling train dataset...')
    print(f'Proportion of class 1 before sampling: {proport_class1}%')
    
    # use SMOTE for oversampling (30% class 1)
    over = SMOTE(sampling_strategy=0.3, random_state=1)
    
    # use random undersampler for undersampling (40% class 1)
    under = RandomUnderSampler(sampling_strategy=0.6)
    
    # create pipeline with both samplers
    steps = [('o', over), ('u', under)]
    pipeline = Pipeline(steps=steps)
    
    # transform the dataset
    X_sampled, y_sampled = pipeline.fit_resample(X, y)
    
    # check final proportion of class 1
    proport_class1_sampled = round(y_sampled.mean()*100,1)
    print(f'Proportion of class 1 after sampling:  {proport_class1_sampled}%\n')
    
    return X_sampled, y_sampled


def train_test_split(dataset):
    """Split dataset into train, validation and test based on year and month.
   
    Args:
        dataset: Pandas dataset.
    Returns:
        Train, validation and test datasets.
    """
    
    # define dates to split dataset
    train_dates = [202101,202102,202103,202104]
    valid_dates = [202105]
    test_dates = [202106]
    print('Split based on reference dates:')
    print(f'trainset: {train_dates}')
    print(f'validset: {valid_dates}')
    print(f'testset:  {test_dates}')
    
    # split into train, validation and test
    train_dataset = dataset[dataset['YEAR_MONTH'].isin(train_dates)]
    valid_dataset = dataset[dataset['YEAR_MONTH'].isin(valid_dates)]
    test_dataset = dataset[dataset['YEAR_MONTH'].isin(test_dates)]
    
    # set index
    train_dataset = train_dataset.set_index(['CUSTOMER_ID','YEAR_MONTH'])
    valid_dataset = valid_dataset.set_index(['CUSTOMER_ID','YEAR_MONTH'])
    test_dataset = test_dataset.set_index(['CUSTOMER_ID','YEAR_MONTH'])
    
    # split dependent and indepent variables
    y_train = train_dataset.pop('Churn_Flag')
    y_valid = valid_dataset.pop('Churn_Flag')
    y_test = test_dataset.pop('Churn_Flag')
    
    # check splits
    train_per = round(len(train_dataset)*100/len(dataset), 2)
    valid_per = round(len(valid_dataset)*100/len(dataset), 2)
    test_per = round(len(test_dataset)*100/len(dataset), 2)
    train_class1_per = round(y_train.mean()*100, 2)
    valid_class1_per = round(y_valid.mean()*100, 2)
    test_class1_per = round(y_test.mean()*100, 2)
    print(f'trainset: {train_per}% of dataset, {train_class1_per}% of class 1')
    print(f'validset: {valid_per}% of dataset, {valid_class1_per}% of class 1')
    print(f'testset:  {test_per}% of dataset, {test_class1_per}% of class 1\n')
    
    return train_dataset, y_train, valid_dataset, y_valid, test_dataset, y_test
