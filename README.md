# Building and deployment of an end-end machine learning pipeline with aws services.

A variant of this project is the [house-price-prediction](https://github.com/theKoladeAkande/house-price-prediction)
which a machine learning pipeline is built on prem. and is deployed via a rest api to serve the predictions of the model.
However, this project is an alternative which leverages AWS services in building the entire pipeline which involves the 
following:

1. Building a custom model for use in sagemaker.
2. Building an inference pipeline for preprocessing and feature engineering in sagemaker.
3. Training and deploying the model.
4. Invoking model predictions with aws lambda.
5. Integrating lambda with aws api gateway to serve model predictions via RestApi

## Building a custom model for use in aws sagemaker

Sagemaker plays host to a lot of machine learning model that could be used out of the box but also offers the flexibility of
using a custom model. The custom built here is the same  xgboost model with that of the 
[house-price-prediction](https://github.com/theKoladeAkande/house-price-prediction) to use this model in sagemaker the model must
comply with the sagemaker's architecture more on this [here](https://sagemaker-workshop.com/custom/containers.html), 
this was achieved through the following steps

1. Structuring the project's folder to comply with sagemaker's model architecture 
2. Writing a custom train script that trains the model and saves the model in a specified location for sagemaker to access.
3. Implemet a flask server to serve model's prediction
4. Wrapping the model training pipeline in a SageMaker-compatible Docker image
5. Ship image to Amazon ECR

Image repo: 249021303942.dkr.ecr.us-west-2.amazonaws.com/sagemaker-custom-models 

To train with this model, all thatis needed is to specify the location in ECR.


## Building an inference pipeline for preprocessing and feature engineering in sagemaker training and deploying the model

For data preprocessing and feature engineering there are alot of options during deployment, 
one way is to specify transformations in aws lambda but rather than having a bulk of functions and layers in aws lambda an
inference pipeline was built which seemed to be cleaner. 

Building an inference pipeline involved the following steps:

1. Custom transformer script:
Writing a script subclassing sklearn.base TransformerMxn and Base classes to build custom transformers,
leveraging sklearn.pipeline to connect this transformers in a pipeline. Including the required functions for sagemaker,
*input_fn*: to parse input data, *predict_fn*: make transformation with the model, *output_fn*: write and encode the output data, 
*model_fn*: load model.
Creating an entry point in the script to train and save the transformer pipeline.
To avoid a global variable conflict in sklearn's container on aws  the transformers were written in a different module and 
specified as a dependency file.

2. Training the transformer pipeline setting the script as the entry point.

3. Performing a Batch transform with trained transformer pipeline and saving the location of the output file.

4. Training machine Learning model:
The custom xgboost model image is loaded up using its location in ECR and training it on the transformed data

5. Building the pipeline:
The transformer pipeline is connected to the trained xgboost model with sagemaker PipelineModel function, which is deployed 
with the endpoint name: inference-pipeline-ep-2020-05-15-21-52-39

This makes inference easy as data is transformed and predictions are made with the same model endpoint inference-pipeline-ep-2020-05-15-21-52-39.
Predictions could be made by passing in the raw data.

## Invoking model predictions with aws lambda
To invoke predictions with the model the architecture used is 

client >>>>> api-gateway >>>>>> lambda >>>>>> model_endpoint

The lambda microservice transform the data to a csv format and gets prediction from the model.

![](https://github.com/theKoladeAkande/house-prediction-cloud-version/blob/master/img/house_price_lambda%20-%20Lambda.png)


## Integrating lambda with aws api gateway to serve model predictions 

For easy integration with clients the lambda is integrated with aws api gateway to serve model predictions via Rest API.
API for this project:  https://i73xeinese.execute-api.us-west-2.amazonaws.com/beta


![](https://github.com/theKoladeAkande/house-prediction-cloud-version/blob/master/img/postman_api.png)




## Project Structure.
* /houseprice-prediction-inference-pipeline.ipynb: SageMaker notebook, Modelling and Inference Pipeline.
* /preprocessor.py: Custom transformer script, entry point for bulding inference pipeline.
* /transformers.py: Depedency script.
* /sagemaker-custom-model/Dockerfile: Custom sagemaker image.
* /sagemaker-custom-model/xgboost/train: Custom model train script 
* /sagemaker-custom-model/xgboost/predictor.py: Serve Prediction
