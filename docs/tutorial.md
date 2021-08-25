# Tutorial

Let's go through the different ways of interacting with the dataset. Through
the Python API, it's possible to pull dataframe files of the trials, `scikit-optimize`
objects which detail the Gaussian process-based search history, 
and the resulting models themselves.

For the purpose of this tutorial, we'll be working with VGG portion of the dataset.
However, the steps are the same for ResNet and DenseNet.

Let's start by loading and initializing the dataset class. 

```
>>> from crossedwires.cifar10 import ResNet50Dataset
>>> dataset = ResNet50Dataset()
```

## Trial Logs
There are multiple logs available as part of the dataset. They are returned as 
Pandas dataframe objects. 

One attribute is the Weights and Biases export. This is the preferred dataframe as it contains
all information in one spot. *Use this dataframe in order to lookup the name of 
the trial to pull a model. You can filter it as desired to isolate the names you 
need in order to load specific models.*
```
>>> dataset.wandb_dataframe()
                    Name  accuracy_diff  pt_test_acc  tf_test_acc     State Notes  ... time_since_restore  time_this_iter_s time_total_s  timestamp  timesteps_since_restore  training_iteration
0    dual_train_85721cc8       0.452007     0.570107       0.1181  finished     -  ...                NaN               NaN          NaN        NaN                      NaN                 NaN
1    dual_train_f7099482       0.435161     0.551261       0.1161  finished     -  ...                NaN               NaN          NaN        NaN                      NaN                 NaN
2    dual_train_fcdc666a       0.430048     0.562548       0.1325  finished     -  ...                NaN               NaN          NaN        NaN                      NaN                 NaN
3    dual_train_21b65dba       0.422199     0.522199       0.1000  finished     -  ...                NaN               NaN          NaN        NaN                      NaN                 NaN
4    dual_train_807e7310       0.409086     0.559586       0.1505  finished     -  ...                NaN               NaN          NaN        NaN                      NaN                 NaN
..                   ...            ...          ...          ...       ...   ...  ...                ...               ...          ...        ...                      ...                 ...
395  dual_train_5ac692ce       0.000800     0.098300       0.0991  finished     -  ...                NaN               NaN          NaN        NaN                      NaN                 NaN
396  dual_train_f416de3a       0.000732     0.574932       0.5742  finished     -  ...                NaN               NaN          NaN        NaN                      NaN                 NaN
397  dual_train_54dc1862       0.000569     0.436069       0.4355  finished     -  ...                NaN               NaN          NaN        NaN                      NaN                 NaN
398  dual_train_5efd9218       0.000316     0.559484       0.5598  finished     -  ...                NaN               NaN          NaN        NaN                      NaN                 NaN
399  dual_train_98577868       0.000137     0.605537       0.6054  finished     -  ...                NaN               NaN          NaN        NaN                      NaN                 NaN

[400 rows x 37 columns]
```

The Ray Tune integration also generates log files, however they are split up
into the separate spaces searched. These are accessible using this command:
```
# specify a num argument to return a particular space, if num=None all spaces are returned 
>>> dataset.ray_tune_dataframes(num=None)
```

## Search Objects

As part of the integration with `scikit-optimize`, each overlapping space searched
generates an `OptimizeResult` object. These track the Gaussian processes which
are responsible for the optimization, and have
helpful utilities to understand the trajectory of the search. In addition, the
surrogate models used to guide the search are contained here. 

```
# specify a num argument to return a particular space, if num=None all spaces are returned 
>>> opt_results = dataset.optimizer_results(num=None)
```

The plotting utilities for these objects can be found [here.](https://scikit-optimize.github.io/stable/auto_examples/plots/visualizing-results.html#sphx-glr-auto-examples-plots-visualizing-results-py)

Here's an example of a plot we can generate, showing the partial dependency plots 
(generated using `skopt.plots.plot_objective`):
![partial dependency](images/partial_dependency.png)

## Trained Models

The key piece of the dataset! Let's interact with the actual trained models that
were generated through the hyperparameter searches. In order to pull a model, you need to know
what the name of the trial is. This can be found in the `wandb_dataframe` attribute
that is part of the main Dataset class. 

Once you have that, you can load models of either framework, and interact with them
from there. 

### PyTorch Example
```
>>> torch_model = dataset.get_pytorch_model('dual_train_85721cc8')
>>> torch_model
ResNet(
  (conv1): Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
  (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (relu): ReLU(inplace=True)
  (maxpool): MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
  (layer1): Sequential(
    (0): Bottleneck(
      (conv1): Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (conv3): Conv2d(64, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn3): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace=True)
      (downsample): Sequential(
        (0): Conv2d(64, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
........
```
### TensorFlow Example

```
>>> tensorflow_model = dataset.get_tensorflow_model('dual_train_85721cc8')
>>> tensorflow_model.summary()
Model: "resnet50"
__________________________________________________________________________________________________
Layer (type)                    Output Shape         Param #     Connected to
==================================================================================================
input_1 (InputLayer)            [(None, 3, 32, 32)]  0
__________________________________________________________________________________________________
conv1_pad (ZeroPadding2D)       (None, 3, 38, 38)    0           input_1[0][0]
__________________________________________________________________________________________________
conv1_conv (Conv2D)             (None, 64, 16, 16)   9472        conv1_pad[0][0]
__________________________________________________________________________________________________
conv1_bn (BatchNormalization)   (None, 64, 16, 16)   256         conv1_conv[0][0]
__________________________________________________________________________________________________
conv1_relu (Activation)         (None, 64, 16, 16)   0           conv1_bn[0][0]
__________________________________________________________________________________________________
........
```