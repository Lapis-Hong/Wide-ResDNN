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
`wide_resdnn` is a simple but powerful variant of `wide_deep`, the main difference is the connection mode of deep part (DNN).

We hope to figure out the best kind of skip connections for large scale sparse data tasks.
Here we provide 5 types connection mode (arbitrary connection is supported also) and 2 types residual mode as follows: 

connect_mode:
- `normal`: use normal DNN with no residual connections
- `first_dense`: add addition connections from first input layer to all hidden layers.
- `last_dense`: add addition connections from all previous layers to last layer.
- `dense`: add addition connections between all layers, similar to DenseNet.
- `resnet`: add addition connections between adjacent layers, similar to ResNet.

residual_mode: 
- `add`: add the previous output, can only used for same hidden size architecture.
- `concat`: concat the previous layers output


## Usage
### Setup
```
cd conf
vim feature.yaml
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
### settings
For simplicity, we do not use cross features, it is highly dataset dependent.
We only do some basic feature engineering for generalization.
For continuous features, we use standard normalization transform as input,  
for category features, we set hash_bucket_size according to its values size,  
and we use embed category features for deep and not use dicretize continuous features for wide.
The specific parameters setting see `conf/*/train.yaml`

### criteo dataset
First, we evaluate the base model `wide_deep` to chose best network architecture.  

network  |1024-512-256 | 512-512-512 | 256-256-256 | 128-128-128 |
-------- | :---------: | :---------: | :---------: | :---------: |
auc      | 0.7763      | 0.7762      | 0.7808      |0.7776       |
logloss  | 0.4700      | 0.4709      | 0.4662      |0.4687       |



From the result we found that `256-256-256` architecture works best,
we also found that dropout decrease the performance.


Then, we evaluate our `wide_resdnn` model with connect mode and residual mode using fixed `256-256-256` architecture.

model             | auc logloss | 
------            | ---------   |             
wide_deep         |0.7808 0.4662|
first_dense/concat|0.7548 0.5397|        
first_dense/add   |**0.7843 0.4636**|            
last_dense/concat |0.7751 0.4830|             
last_dense/add    |0.7840 **0.4636**|   
dense/concat      |0.7258 1.2312|             
dense/add         |0.7839 0.4640|          
resnet/concat     |0.7638 0.5023|            
resnet/add        |0.7841 0.4637|             

We found that `add` is consistently much better than `concat`, all the four connect mode result in similar results and our `wide_resdnn` significantly better than `wide_deep`.

Then, we evaluate `wide_deep`, `wide_resdnn` model with different number of layers.

model   |   wide_deep  |first_dense/add|last_dense/add| dense/add| resnet/add  |
---     | ---          | ---         | ---         | ---         | ---         |
layers  |auc    logloss|auc   logloss|auc   logloss|auc   logloss|auc   logloss|
3       |0.7808 0.4662 |0.7843 0.4636|0.7840 0.4636|0.7839 0.4640|0.7841 0.4637|
5       |0.7783 0.4680 |             |0.7313 0.6651
7       |||||
9       |||||

Finally, we need to evaluate model variance, we run each model 10 times to calculate related auc and logloss statics.
The network setting is `256-256-256`.
model | wide_deep | wide_resdnn |
---   | ---       | ---         |
1     |0.7808 0.4662|0.7843 0.4636|
2     |0.7798 0.4672|0.7838 0.4652|
3     |0.7783 0.4685|0.7821 0.4677|
4     |0.7828 0.4653|0.7818 0.4670|
5     |0.7767 0.4695|0.7841 0.4638| 
6     |0.7826 0.4651|0.7823 0.4653|
7     |0.7783 0.4685|0.7831 0.4648|
8     |0.7767 0.4699|0.7841 0.4638|
9     |0.7775 0.4689|0.7827 0.4654|
10    |0.7821 0.4655|0.7831 0.4647|
**mean**|0.7796 0.4675|0.7831 0.4651|
**std** |0.0023 0.0017|0.0009 0.0013|
We found `wide_resdnn` is significantly better than `wide_deep` and has lower variance.

### avazu dataset
First, we evaluate the base model `wide_deep` to chose best network architecture. 
wide_deep| 512-512-512 | 256-256-256 | 128-128-128 |
-------- | ----------- | ----------- | ----------- | 
auc      |  0.7528     | 0.7529      | 0.7528      |
logloss  |  0.3950     | 0.3950      | 0.3950      |
We found `hidden size` has little influence on performance.

Then, we evaluate our `wide_resdnn` model with connect mode and residual mode using fixed `256-256-256` architecture.

model             | auc logloss | 
------            | ---------   |             
wide_deep         |0.7529 0.3950|
first_dense/concat||        
first_dense/add   ||            
last_dense/concat ||             
last_dense/add    ||   
dense/concat      ||             
dense/add         ||          
resnet/concat     ||            
resnet/add        || 
    
    
Then, we evaluate `wide_deep`, `wide_resdnn` model with different number of layers.

model   |   wide_deep  |first_dense/add|last_dense/add| dense/add| resnet/add  |
---     | ---          | ---         | ---         | ---         | ---         |
layers  |auc    logloss|auc   logloss|auc   logloss|auc   logloss|auc   logloss|
3       |0.7529 0.3950 |||||
5       | |             |
7       |||||
9       |||||

Finally, we need to evaluate model variance, we run each model 10 times to calculate related auc and logloss statics.
The network setting is `256-256-256`.
model | wide_deep | wide_resdnn |
---   | ---       | ---         |
1     |||
2     |||
3     |||
4     |||
5     ||| 
6     |||
7     |||
8     |||
9     |||
10    |||
**mean**|||
**std** |||