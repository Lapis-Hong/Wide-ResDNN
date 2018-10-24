# Configuration in Yaml
For each dataset, we need three config files: 
- schema.yaml
- feature.yaml
- train.yaml

## Schema config
This should be consistent with the **data fields**, the order matters.  
Field index start from 1, and the target variable should be named with **`label`**.

### Examples:
```
# Field index: field name
1: xxx
2: xxx
3: xxx

```

## Feature config
Each feature consists 2 attributes **`type`**, **`parameter`**.  
The **`parameter`** are differ according to two **`type`**, **continuous** or **category**.  
The feature name should be consistent with `schema.yaml`.

### Examples:
```
# For category feature, using `tf.feature.categorical_column_with_hash_bucket` 
f1:                 
    type: category    
    parameter: 
         hash_bucket_size: 1000  # required
         embedding_dim: 8        # optional, set empty to not use category feature for deep input;
                                             set positive integer to embedding category feature for deep using tf.feature.embedding_column;
                                             set `auto` to use empirical formula to calculate embedding dim.
                                             
# For continuous feature, using `tf.feature_column.numeric_column`
f2:                 
    type: continuous    
    parameter:      
        mean: 3.5                # optional, set both mean and std to do standard normalization.
        std: 2.5                 # optional
        boundaries: [0., 1., 2.] # optional, set empty to not use continuous feature for wide input;
                                             set boundaries to discretize continuous feature for wide input using tf.feature_column.bucketized_column
```  
### how to set `hash_bucket_size` ?
If category size=1000, how much should hash_bucket_size be ?  
   An interesting discovery is that randomly chose N number a_i between 1~N, i=1,...N  
     let b_i = a_i % N, the distinct b_i from all N number is about 0.633.  
     in other words, a random hash func chose N as hash_bucket_size collision rate is 0.633.  
   Recommend `hash_bucket_size` to be 2~3*category size.  
     larger `hash_bucket_size` require more memory and complexity, but smaller cause more collision  
   
   Here use the strategy that
   -  for low sparsity category, set `hash_bucket_size` 3~4*category size to reduce collision  
   -  for high sparsity category, set 1.5~2*category size to save memory.  

### Notes:
For feature.basic config:
1. Use static hash_bucket_size=1000 and use category features for deep with static embedding_dim=8.
2. No standard normalization and not use continuous feature for wide.

For feature.advance config:
1. Use dynamic hash_bucket_size and use category features for deep with dynamic embedding_dim.
2. Do standard normalization and use continuous feature for wide.

## Train config
All train config divided into following four part: 
- train 
- model
- runconfig
- distributed

Optional parameters set empty to use default.
Note this configuration set defaults to argparser. Same params can be overrided by using command line.  
For example:   
`python train.py --model_dir ./model_new`

### Examples
```
# Train & Test Parameters
train:
  model_dir: model                  # model base directory            
  model_type: wide_deep             # model type, one of `deep`, `wide`, `wide_deep`
  train_data: data/criteo/train.csv # train data file path
  dev_data: data/criteo/dev.csv     # validation data file path 
  test_data: data/criteo/test.csv   # test data file path
  batch_size: 256                   # batch size
  train_epochs: 5                   # train epochs
  epochs_per_eval: 1                # evaluation every epochs
  keep_train: 0                     # bool, set true or 1 to keep train from ckpt
  num_samples: 50000000             # train sample size for shuffle buffer size
  checkpoint_path:                  # optional, checkpoint path used for testing  
  verbose: 1                        # bool, Set 0 for tf log level INFO, 1 for ERROR 
# Model Parameters
model:
  # Wide Parameters                  
  wide_learning_rate: 0.1           # wide part initial learning rate
  wide_lr_decay: true               # wide part whether to use learning rate decay or not
  wide_l1: 0.5                      # optional, wide part l1 regularization lambda
  wide_l2: 1                        # optional, wide part l2 regularization lambda

  # Deep Parameters
  # connect_mode: one of {`normal`, `first_dense`, `last_dense`, `dense`, `resnet`} or arbitrary connections
  #   1. `normal`: normal DNN with no residual connections.
  #   2. `first_dense`: add addition connections from first input layer to all hidden layers.
  #   3. `last_dense`: add addition connections from all previous layers to last layer.
  #   4. `dense`: add addition connections between all layers, similar to DenseNet.
  #   5. `resnet`: add addition connections between adjacent layers, similar to ResNet.
  #   6. arbitrary connections string: add addition connections between layer0 to layer1 like '01', separated by comma
  #      eg: '01,03,12' index start from zero(input_layer), max index is len(hidden_units), smaller index first.
  
  # To use multi dnn model, set nested hidden_units, list connect_mode, list residual_mode
  # Examples:
  # hidden_units: [[1024, 12,256], [512,256]] 
  # connect_mode: [normal, dense]
  # residual_mode: [add, concat]

  hidden_units: [1024, 512, 256]    # hidden_units: List of each hidden layer units, set nested list for Multi DNN. 
  connect_mode: normal              # see above
  residual_mode:                    # one of `add` or `concat`, must be set when connect_mode is not `normal`

  deep_learning_rate: 0.1           # deep part initial learning rate
  deep_lr_decay: false              # deep part whether to use learning rate decay or not
  activation_function: tf.nn.relu   # activation function, must use tf API format
  deep_l1: 0.01                     # optional, deep part l1 regularization lambda
  deep_l2: 0.01                     # optional, deep part l2 regularization lambda
  dropout:                          # optional, dropout rate, 0.1 for drop 10%
  batch_normalization: false        # optional, bool, set true or 1 for use batch normalization
  
# Saving Parameters (Optional)
# Defined in tf.estimator.RunConfig. See details in https://www.tensorflow.org/api_docs/python/tf/estimator/RunConfig
runconfig:
  tf_random_seed: 12345
  save_summary_steps: 100           # Defaults to 100
  save_checkpoints_steps:           # Set either save_checkpoints_steps or save_checkpoints_secs
  save_checkpoints_secs: 1500       # Defaults to 600 (10 minutes)
  keep_checkpoint_max: 5            # Defaults to 5
  keep_checkpoint_every_n_hours: 1  # Defaults to 10000
  log_step_count_steps: 100         # Defaults to 100
  
# Distributed Parameters
distributed:
  is_distributed: 0                 # bool, whether to use distributed 
  cluster:
    ps: ['10.172.110.162:3333']     # ps ip list
    chief: ['10.120.180.212:3333']  # chief ip list
    worker: ['10.120.180.213:3333'] # worker ip list
  job_name: ps                      # job_name: one of `ps`, `chief`, `worker`                     
  task_index: 0                     # task_index: the host index, start from 0
```
  

