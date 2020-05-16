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



features = ['OverallQual',
            'GarageCars',
            'ExterQual',
            'BsmtQual',
            'GrLivArea',
            'Neighborhood',
            'FullBath',
            'KitchenQual',
            'KitchenAbvGr',
            '2ndFlrSF',
            'PoolArea',
            '1stFlrSF',
            'GarageCond',
            'TotRmsAbvGrd',
            'GarageQual',
            'BsmtFinSF1',
            'TotalBsmtSF',
            'GarageFinish',
            'CentralAir',
            'FireplaceQu',
            'LandContour',
            'LotFrontage',
            'Fireplaces']


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
    """
    Function required by sagemaker to parse  the input data,
    herea subset of the required feature is selected.
    """
    if content_type == 'text/csv':
        raw_data = pd.read_csv(StringIO(input_data))
        raw_data = raw_data[features].copy()
        return raw_data
    else:
        raise ValueError('This script only takes csv')


def output_fn(prediction, accept):
    """
    Required by sagemaker to write the output from predict_fn,
    two choices are supported for this script json and csv.
    """

    if accept == "application/json":
        instances = []
        for row in prediction.values.tolist():
            instances.append({"features": row})

        json_output = {"instances": instances}

        return worker.Response(json.dumps(json_output), accept, mimetype=accept)
    elif accept == 'text/csv':
        return worker.Response(encoders.encode(prediction, accept), accept, mimetype=accept)
    else:
        raise RuntimeError("{} accept type is not supported by this script.".format(accept))


def predict_fn(input_data, model):
    """
    Required by sagemaker to predict with model,
    """

    features = model.transform(input_data)#for preprocessing transform is used instead of predict
    return features



def model_fn(model_dir):
    """
    Loads existing  model
    """
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

    #try to read in multiple file
    input_files = [os.path.join(args.train, file) for file in os.listdir(args.train)]

    if not input_files:
        raise ValueError(('There are no files in {}.\n' +
                            'This usually indicates that the channel ({}) was incorrectly specified,\n' +
                            'the data specification in S3 was incorrectly specified or the role specified\n' +
                            'does not have permission to access the data.').format(args.train, "train"))

    if len(input_files) >= 2:
        raw_data_list = [ pd.read_csv(file, header=None) for file in input_files ]
        raw_data_main = pd.concat(raw_data_list)
    else:
        raw_data_main = pd.read_csv(input_files[0])


    raw_data_X = raw_data_main[features].copy()
    raw_data_y = raw_data_main[[target]]

    print('fitting transformers....')
    house_price_preprocessing_pipeline.fit(raw_data_X, raw_data_y)


    joblib.dump(house_price_preprocessing_pipeline,
                os.path.join(args.model_dir, "model.joblib"))
    print("Saving model....")
