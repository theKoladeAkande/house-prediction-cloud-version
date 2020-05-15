from __future__ import print_function

import time
import sys
from io import StringIO
import os
import shutil

import argparse
import csv
import json
from sklearn.externals import joblib

import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline


from sagemaker_containers.beta.framework import (
    content_types, encoders, env, modules, transformer, worker)


import transformers as trf

class BaseError(Exception):
    """ Base package error"""

class InvalidModelInput(BaseError):
    """Invalid input for model"""



class CategoricalNaNImputer(BaseEstimator, TransformerMixin):
    """ Creates another category in categorical features by filling
        the NaN values with a new category """

    def __init__(self, *, variables: list = None, category: str = None):
        if not isinstance(variables, list):
            self.variables = [variables]
        else:
            self.variables = variables

        self.category = category

    def fit(self, X, y=None):
        """Nothing to be estimated from trainset,
           method implemented to accomodate sklearn pipeline"""

        return self

    def transform(self, X: pd.DataFrame, y=None):
        """fills the missing value of categotrical features
           with the specified category"""

        X = X.copy()
        for feature in self.variables:
            X[feature] = X[feature].fillna(self.category)
        return X


class NumeriaclNaNImputer(BaseEstimator, TransformerMixin):
    """Fills the missing values in numerical features with the estimated median 
       from training data"""
    def __init__(self, *, variables: list = None):
        if not isinstance(variables, list):
            self.variables = [variables]
        else:
            self.variables = variables

    def fit(self, X: pd.DataFrame, y=None):
        """Estimates the median from the training data"""
        self.median_dict = {}

        for feature in self.variables:
            self.median_dict[feature] = X[feature].median() 
        return self

    def transform(self, X: pd.DataFrame, y=None):
        """fills the missing value with the estimated median from the training data"""
        X = X.copy()

        for feature in self.variables:
            median = self.median_dict[feature]
            X[feature] = X[feature].fillna(median)
        return X


class RareLabelCategoryImputer(BaseEstimator, TransformerMixin):
    """Replaces observations in categorical feature with the value "rare", with occurences 
       less than a certain percentage """

    def __init__(self, *, variables: list = None, tol: float = 0.01):
        if not isinstance(variables, list):
            self.variables = [variables]
        else:
            self.variables = variables
        
        self.tol = tol
    
    def fit(self, X: pd.DataFrame, y=None):
        """gets frequent labels from train data"""
        self.frequent_labels_dict = {}

        for feature in self.variables:
           temp = pd.Series(X[feature].value_counts()/ len(X))
           frequent_labels = list(temp[temp > self.tol].index)
           self.frequent_labels_dict[feature] = frequent_labels
 
        return self

    def transform(self, X: pd.DataFrame, y=None):
        """creates new category label rare for observations not conatined in
           frequent_labels_dict"""

        X = X.copy()

        for feature in self.variables:
            X[feature] = np.where(X[feature].isin(
                self.frequent_labels_dict[feature]),X[feature], 'Rare')
        return X
        

class CategoricalMonotonicEncoder(BaseEstimator, TransformerMixin):
    """Encode categorical variable via monotonic ordering"""

    def __init__(self, *, variables: list = None):
        if not isinstance(variables, list):
            self.variables = [variables]
        else:
            self.variables = variables

    def fit(self, X: pd.DataFrame, y=None):
        """estimates monotonic ordering on the training data"""
        temp = pd.concat([X, y], axis=1)
        temp.columns = list(X.columns)+['target']

        self.monotonic_labels_dict = {}

        for feature in self.variables:
            order = temp[[feature, 'target']].groupby(feature).median().sort_values('target').index
            monotonic_label = {keys:idx for idx, keys in enumerate(order, 0)}

            self.monotonic_labels_dict[feature] = monotonic_label

        return self

    def transform(self, X: pd.DataFrame, y=None):
        """" encodes categorical features with monotonic label estimated from
             train data """
        X = X.copy()
        for feature in self.variables:
            X[feature] = X[feature].map(self.monotonic_labels_dict[feature])

        if X[self.variables].isnull().any().any():
            counts_null = X[self.variables].isnull().any()
            vars_na = {key: vars for key, vars in counts_null.items() if vars is True}
            raise InvalidModelInput(
                            f'Error in encoding features NaN Values introduced for'
                            f'features {vars_na.keys()}')
        return X


class DropFeatures(BaseEstimator, TransformerMixin):
    """Drop Features not needed"""
    def __init__(self, *, drop_features: list = None):
        if not isinstance(drop_features, list):
            self.drop_features = [drop_features]
        else:
            self.drop_features = drop_features

    def fit(self, X, y=None):
        """Nothing to be estimated from training data, 
            fit implemented to accomodate sklearn pipeline"""

        return self


    def transform(self, X: pd.DataFrame, y=None):
        """ drops specified features"""
        X = X.copy()

        X = X.drop(self.drop_features, axis=1)

        return X

class BinaryFeatureGenerator(BaseEstimator, TransformerMixin):
    """generates new a new feature specifying the exixtence of a property as 0
       and non-existence as 1 """

    def __init__(self, variables: list = None):
        if not isinstance(variables, list):
            self.variables = [variables]
        else:
            self.variables = variables

    def fit(self, X, y=None):
        """Nothing to be estimated from trainset,
           method implemented to accomodate sklearn pipeline"""

        return self

    def transform(self, X: pd.DataFrame, y=None):
        """generates a binary feature indicator """

        X = X.copy()

        for feature in self.variables:
            X['has'+feature] = X[feature].apply(lambda x: 1 if x > 0 else 0)
        return X



categorical_features = ['ExterQual',
                        'BsmtQual',
                        'Neighborhood',
                        'KitchenQual',
                        'GarageCond',
                        'GarageQual',
                        'GarageFinish',
                        'CentralAir',
                        'FireplaceQu',
                        'LandContour']

#categorical features with na
categorical_features_with_na = ['BsmtQual',
                                'GarageCond',
                                'GarageQual',
                                'GarageFinish',
                                'FireplaceQu']

categorical_na_not_allowed = [ 'CentralAir',
                               'ExterQual',
                               'KitchenQual',
                               'LandContour',
                               'Neighborhood' ]

# Numerical features with na
numerical_features_with_na = ['LotFrontage']

numerical_na_not_allowed = ['1stFlrSF',
                            '2ndFlrSF',
                            'BsmtFinSF1',
                            'FullBath',
                            'GarageCars',
                            'GrLivArea',
                            'KitchenAbvGr',
                            'OverallQual',
                            'PoolArea',
                            'TotRmsAbvGrd',
                            'TotalBsmtSF']

features_for_feature_generation = ['Fireplaces']

drop_features = ['Fireplaces']

target = 'SalePrice'

house_price_preprocessing_pipeline = Pipeline(
    [
    ('categorical_nan_imputer',
    trf.CategoricalNaNImputer(variables=categorical_features_with_na,
                                            category='MissingValue')),
    ('numerical_na_imputer',
    trf.NumeriaclNaNImputer(variables=numerical_features_with_na)),
    ('rare_label_encoder',
    trf.RareLabelCategoryImputer(variables=categorical_features, tol=0.01)),
    ('cateogrical_encoder',
    trf.CategoricalMonotonicEncoder(variables=categorical_features)),
    ('binary_feature_generator',
    trf.BinaryFeatureGenerator(variables=features_for_feature_generation)),
    ('drop_features',
    trf.DropFeatures(drop_features=drop_features))])


def input_fn(input_data, content_type):
    if content_type == 'text/csv':
        raw_data = pd.read_csv(StringIO(input_data))

        return raw_data
    else:
        raise ValueError('This script only takes csv')


def output_fn(prediction, accept):
        if accept == "application/json":
            instances = []
            for row in prediction.tolist():
                instances.append({"features": row})

            json_output = {"instances": instances}

            return worker.Response(json.dumps(json_output), accept, mimetype=accept)
        elif accept == 'text/csv':
            return worker.Response(encoders.encode(prediction, accept), accept, mimetype=accept)
        else:
            raise RuntimeException("{} accept type is not supported by this script.".format(accept))


def predict_fn(input_data, model):
        features =  model.transform(input_data)
        return features



def model_fn(model_dir):
    preprocessor = joblib.load(os.path.join(model_dir, "model.joblib"))
    return preprocessor



if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Sagemaker specific arguments. Defaults are set in the environment variables.
    parser.add_argument('--model-dir', type=str, default=os.environ.get('SM_MODEL_DIR'))
    parser.add_argument('--output-data-dir', type=str, default=os.environ['SM_OUTPUT_DATA_DIR'])
    parser.add_argument('--train', type=str, default=os.environ.get('SM_CHANNEL_TRAIN'))
    args, _ = parser.parse_known_args()
    print('reading data...')

    input_files = [os.path.join(args.train, file) for file in os.listdir(args.train)]
    if not input_files:
        raise ValueError(('There are no files in {}.\n' +
                            'This usually indicates that the channel ({}) was incorrectly specified,\n' +
                            'the data specification in S3 was incorrectly specified or the role specified\n' +
                            'does not have permission to access the data.').format(training_path, channel_name))
    if len(input_files) >= 2:
        raw_data_list = [ pd.read_csv(file, header=None) for file in input_files ]
        raw_data = pd.concat(raw_data_list)
    else:
        raw_data = pd.read_csv(input_files[0])

    raw_data_X = raw_data.iloc[:, :-1]
    raw_data_y = raw_data[[target]]

    print('fitting transformers....')
    house_price_preprocessing_pipeline.fit(raw_data_X, raw_data_y)

    joblib.dump(house_price_preprocessing_pipeline,
                os.path.join(args.model_dir, "model.joblib"))
    print("Saving model....")
