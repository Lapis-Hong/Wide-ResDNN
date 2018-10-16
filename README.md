# Wide and ResDNN Learning (Wide&ResDNN)

## Directory
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


## Dataset
1. Kaggle Criteo Dataset for ctr prediction. 
2. Kaggle Avazu Dataset for ctr prediction

# Data fields
Label - Target variable that indicates if an ad was clicked (1) or not (0).
I1-I13 - A total of 13 columns of integer features (mostly count features).
C1-C26 - A total of 26 columns of categorical features. The values of these features have been hashed onto 32 bits for anonymization purposes. 

The semantic of the features is undisclosed.
When a value is missing, the field is empty.

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







