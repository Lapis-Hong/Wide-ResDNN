# Wide and ResDNN Learning (Wide&ResDNN)

## Content
* [Overview](#overview)
* [Dataset](#dataset)
* [Model](#model)
* [Usage](#usage)
* [Experiments](#experiments)


## Overview
Here, we develop a brandly new framework, called **Wide and ResDNN** for general structural data classification tasks, such as CTR prediction, recommend system, etc.
The model extend the DNN part of Wide and Deep model with arbitrary connections between layers, including connection mode similar to ResNet and DenseNet, which is widely used in CV.

This work is inspired by [wide and deep model](https://research.googleblog.com/2016/06/wide-deep-learning-better-together-with.html) 
and [ResNet](https://arxiv.org/pdf/1512.03385v1.pdf) [DenseNet](https://arxiv.org/pdf/1608.06993.pdf)
The **wide model** is able to memorize interactions with data with a large number of features but not able to generalize these learned interactions on new data. The **deep model** generalizes well but is unable to learn exceptions within the data. The **wide and deep model** combines the two models and is able to generalize while learning exceptions.

The code is based on TensorFlow wide and deep tutorial with high level `tf.estimator.Estimator` API. 
We use Kaggle Criteo and Avazu Dataset as examples.

This project is still on progress...
### Requirements
- Python 2.7
- TensorFlow >= 1.4
- NumPy
- pyyaml

## Dataset

### 1. Criteo
Kaggle Criteo Dataset [Display Advertising Challenge](https://www.kaggle.com/c/criteo-display-ad-challenge)

#### Data descriptions
- train.csv - The training set consists of a portion of Criteo's traffic over a period of 7 days. Each row corresponds to a display ad served by Criteo. Positive (clicked) and negatives (non-clicked) examples have both been subsampled at different rates in order to reduce the dataset size. The examples are chronologically ordered.
- test.csv - The test set is computed in the same way as the training set but for events on the day following the training period.
Note: the test.csv file label is unreleased, here we randomly split train.csv into train, dev, test set.

#### Data fields
- Label - Target variable that indicates if an ad was clicked (1) or not (0).
- I1-I13 - A total of 13 columns of integer features (mostly count features).
- C1-C26 - A total of 26 columns of categorical features. The values of these features have been hashed onto 32 bits for anonymization purposes. 

The semantic of the features is undisclosed.
When a value is missing, the field is empty.

### 2. Avazu
Kaggle Avazu Dataset [Click-Through Rate Prediction](https://www.kaggle.com/c/avazu-ctr-prediction)

#### Data descriptions
- train - Training set. 10 days of click-through data, ordered chronologically. Non-clicks and clicks are subsampled according to different strategies.
- test - Test set. 1 day of ads to for testing your model predictions. 
Note: the test file label is unreleased, here we randomly split train.csv into train, dev, test set.

#### Data fields
- id: ad identifier
- click: 0/1 for non-click/click
- hour: format is YYMMDDHH, so 14091123 means 23:00 on Sept. 11, 2014 UTC.
- C1 -- anonymized categorical variable
- banner_pos
- site_id
- site_domain
- site_category
- app_id
- app_domain
- app_category
- device_id
- device_ip
- device_model
- device_type
- device_conn_type
- C14-C21 -- anonymized categorical variables

## Model
1. provide flexible feature configuration and train configuration.
2. support custom dnn network (arbitrary connections between layers)
3. support distributed tensorflow  

## Usage
### Setup
```
cd conf
vim feature.yaml
vim model.yaml
vim train.yaml
...
```

### Training
You can run the code locally as follows:

```
python train.py
```
### Testing
```
python test.py
```

### TensorBoard

Run TensorBoard to inspect the details about the graph and training progression.

```
tensorboard --logdir=./model/wide_deep
```

## Experiments
First, we evaluate the base model `wide_deep` to chose best hyper-parameters.  

wide_deep|1024-512-256 | 512-512-512 | 256-256-256 |
-------- | ----------- | ----------- | ----------- | 
auc      | 0.7763      | 0.7762      | 0.7808      |
logloss  | 0.4700      | 0.4709      | 0.4662      |


From the result we found that `256-256-256` architecture works best,
we also found that dropout decrease the performance.**


Then, we evaluate our `wide_resdnn` model with different number of layers using fixed 256 hidden units.

model             | 3 layers    | 4 layers    | 5 layers    | 6 layers    |
------            | ---------   | ---------   | ---------   | --------    |
------            | auc loss    | auc loss    | auc loss    | auc loss    |
wide_deep         |0.7808 0.4662|0.7798 0.4674|0.7783 0.4680|0.7695 0.4746|
first_dense/concat|0.7548 0.5397|             |             |             |
first_dense/add   |             |             |             |             |
last_dense/concat |             |             |             |             |
last_dense/add    |0.7840 0.4636|             |             |             |             
dense/concat      |             |             |             |             |
dense/add         |             |             |             |             |
resnet/concat     |             |             |             |             |
resnet/add        |             |             |             |             |





