# -*- coding: utf-8 -*-

from pathlib import Path
import os

from ml_module.preprocessing import preprocessing, fill_missing_values, remove_outliers
from ml_module.data_prep import train_test_split, dataset_sampling
from ml_module.feature_selection import select_features
from ml_module.model_classif import train_model, predict_test


# get project root directory
directory = os.getcwd()
root_folder = str(Path(directory).absolute())

# load dataset
dataset = preprocessing(root_folder,'\\data\\01_raw\\dataset.csv')

# split dataset
X_train, y_train, X_valid, y_valid, X_test, y_test = train_test_split(dataset)

# fill missing values
X_train, X_valid, X_test = fill_missing_values(root_folder, X_train, X_valid, X_test)

# remove outliers from trainset
outliers_mask = remove_outliers(X_train)
X_train = X_train[outliers_mask]
y_train = y_train[outliers_mask]

# feature selection
final_predictors = select_features(X_train, y_train)
X_train = X_train[final_predictors]
X_valid = X_valid[final_predictors]
X_test = X_test[final_predictors]

# sampling train data
X_train_sampled, y_train_sampled = dataset_sampling(X_train, y_train)

# train model
model = train_model(root_folder,X_train_sampled, y_train_sampled, X_valid, y_valid)

# predict test data
predict_test(root_folder,model, X_test, y_test)

print('\n Done!')
