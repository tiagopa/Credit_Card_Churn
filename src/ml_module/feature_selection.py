# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from boruta import BorutaPy


def quasi_constant_features(dataset, threshold=0.99):
    """Identification of columns in the dataset that are constant or quasi-constant.
   
    Args:
        dataset: Pandas dataset.
        threshold: If any value of a given column represents more than this percentage
                   of datapoints the column is constant.
    Returns:
        List of constant or quasi-constant columns.
    """
    
    list_columns = dataset.columns
    quasi_constant_columns = []
    
    for column in list_columns:
        res=dataset[column].value_counts(normalize=True)
        if res.iloc[0]>=threshold:
            quasi_constant_columns.append(column)
            
    return quasi_constant_columns


class MultiCollinearityEliminator():
    """ Identify highly correlated features and drops the one that is least correlated with the target.
    code from Joseph Jacob: https://stackoverflow.com/questions/29294983/how-to-calculate-correlation-between-all-columns-and-remove-highly-correlated-on
    """   
    
    #Class Constructor
    def __init__(self, df, target, threshold=0.7, corr_method='pearson'):
        self.df = df
        self.target = target
        self.threshold = threshold
        self.corr_method = corr_method

    #Method to create and return the feature correlation matrix dataframe
    def createCorrMatrix(self, include_target = False):
        #Checking if we should include the target in the correlation matrix
        if (include_target == False):
            df_temp = self.df.drop([self.target], axis =1)
            
            #Setting method to Pearson to prevent issues in case the default method for df.corr() gets changed
            #Setting min_period to 30 for the sample size to be statistically significant (normal) according to 
            #central limit theorem
            corrMatrix = df_temp.corr(method=self.corr_method, min_periods=30).abs()
        #Target is included for creating the series of feature to target correlation - Please refer the notes under the 
        #print statement to understand why we create the series of feature to target correlation
        elif (include_target == True):
            corrMatrix = self.df.corr(method=self.corr_method, min_periods=30).abs()
        return corrMatrix

    #Method to create and return the feature to target correlation matrix dataframe
    def createCorrMatrixWithTarget(self):
        #After obtaining the list of correlated features, this method will help to view which variables 
        #(in the list of correlated features) are least correlated with the target
        #This way, out the list of correlated features, we can ensure to elimate the feature that is 
        #least correlated with the target
        #This not only helps to sustain the predictive power of the model but also helps in reducing model complexity
        
        #Obtaining the correlation matrix of the dataframe (along with the target)
        corrMatrix = self.createCorrMatrix(include_target = True)                           
        #Creating the required dataframe, then dropping the target row 
        #and sorting by the value of correlation with target (in asceding order)
        corrWithTarget = pd.DataFrame(corrMatrix.loc[:,self.target]).drop([self.target], axis = 0).sort_values(by = self.target)                    
        #print(corrWithTarget, '\n')
        return corrWithTarget

    #Method to create and return the list of correlated features
    def createCorrelatedFeaturesList(self):
        #Obtaining the correlation matrix of the dataframe (without the target)
        corrMatrix = self.createCorrMatrix(include_target = False)
        #print(corrMatrix, '\n')                     
        colCorr = []
        #Iterating through the columns of the correlation matrix dataframe
        for column in corrMatrix.columns:
            #Iterating through the values (row wise) of the correlation matrix dataframe
            for idx, row in corrMatrix.iterrows():                                            
                if(row[column]>self.threshold) and (row[column]<1):
                    #Adding the features that are not already in the list of correlated features
                    if (idx not in colCorr):
                        colCorr.append(idx)
                    if (column not in colCorr):
                        colCorr.append(column)
        #print(colCorr, '\n')
        return colCorr

    #Method to eliminate the least important features from the list of correlated features
    def deleteFeatures(self, colCorr):
        #Obtaining the feature to target correlation matrix dataframe
        corrWithTarget = self.createCorrMatrixWithTarget()
        for idx, row in corrWithTarget.iterrows():
            #print(idx, '\n')
            if (idx in colCorr):
                self.df = self.df.drop(idx, axis=1)
                break
        return self.df

    #Method to run automatically eliminate multicollinearity
    def autoEliminateMulticollinearity(self):
        #Obtaining the list of correlated features
        colCorr = self.createCorrelatedFeaturesList()                                       
        while colCorr != []:
            #Obtaining the dataframe after deleting the feature (from the list of correlated features) 
            #that is least correlated with the taregt
            self.df = self.deleteFeatures(colCorr)
            #Obtaining the list of correlated features
            colCorr = self.createCorrelatedFeaturesList()                                     
        return list(self.df)


def boruta_feature_selection(X,y):
    """Apply boruta pyfeature selection algorithm, based on Random Forest Classifier.
   
    Args:
        X: Dataset with independent variables.
        y: Dependent variable.
    Returns:
        Lists with accepted and rejected features.
    """
    
    # RandomForestClassifier as the estimator
    rf_class = RandomForestClassifier(random_state=1, n_estimators=1000, max_depth=5)
    
    # BorutaPy object created with RandomForestClassifier as the estimator
    boruta_selector = BorutaPy(rf_class, n_estimators='auto', random_state=1)
    
    # fit boruta
    boruta_selector.fit(np.array(X), np.array(y))  
    
    # list accepted and rejected features
    accepted_feats = X.columns[boruta_selector.support_].to_list()
    rejected_feats = [elem for elem in X.columns.values if elem not in accepted_feats]
    
    return accepted_feats, rejected_feats


def select_features(X,y):
    """Performs feature selection:
        1) drop constant or quasi-constant features
        2) drop gender columns (not ethic)
        3) remove multicollinearity
        4) boruta feature selection
   
    Args:
        X: Dataset with independent variables.
        y: Dependent variable.
    Returns:
        List of features to keep in dataset.
    """
    
    print('Selecting features...')
    
    # drop constant and quasi-constant cols
    drop_constant = quasi_constant_features(X, threshold=0.99)
    fs_dataset = X.drop(drop_constant, axis=1)
    if drop_constant:
        print(f'Droppping constant and quasi-constant features: {drop_constant}')
    
    # drop Gender columns
    gender_feats = ['Gender_M','Gender_F']
    fs_dataset = fs_dataset.drop(gender_feats, axis=1)
    if gender_feats:
        print(f'Dropping gender features: {gender_feats}')
    
    # remove multicollinearity
    multicol_eliminator = MultiCollinearityEliminator(fs_dataset.join(y), 'Churn_Flag')
    tokeep = multicol_eliminator.autoEliminateMulticollinearity() # list feats to keep
    drop_multicol = [elem for elem in list(fs_dataset) if elem not in tokeep] # feats to drop
    if drop_multicol:
        print(f'Dropping highly correlated features: {drop_multicol}')
    
    # boruta feature selection
    fs_dataset = fs_dataset.drop(drop_constant+drop_multicol, axis=1)
    feats_keep, feats_drop = boruta_feature_selection(fs_dataset,y)
    if feats_drop:
        print(f'Dropping features rejected by boruta: {feats_drop}\n')
        
    return feats_keep
