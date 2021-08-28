# Overview

## Motivation

It can be of great benefit to machine learning practitioners to have multiple 
frameworks available for implementation and training of neural networks. However,
it also comes with challenges, the main of which being *switching* between these different
libraries while maintaining existing performance levels. Critically, there can 
be vast accuracy differences between these frameworks,
even when every possible effort is made to maintain consistency. This phenomenon
can be greatly influenced by the hyperparameters selected, however, there is not
currently an understanding as to why this is the case.
There are plenty of [examples](https://stackoverflow.com/search?q=pytorch+and+tensorflow+accuracy)
of this on forums and support sites, 
but there hasn't been any formal study or data collected to our knowledge. 
The **CrossedWires** dataset is intended to close this gap, collecting extensive
model information using hyperparameter search methods. Both the search trajectories
and the resulting trained models are included in the dataset. We invite the machine
learning community to utilize the CrossedWires dataset to explore the potential 
mechanisms behind these inconsistencies, further the collective understanding of 
hyperparameter spaces for improved training stability, and build tools for greater
reliability and reproducibility in deep learning that eliminate these discrepancies.

## Dataset Details

The dataset currently consists of PyTorch and TensorFlow models using 
three different computer vision architectures on the CIFAR10 dataset across a 
wide hyperparameter space. Using hyperparameter optimization (HPO), models are 
trained on 400 sets of hyperparameters suggested by a search algorithm. These 
results showcase a wide range of benchmark accuracy divergence on the test set 
split. The 390 GB dataset and benchmarks presented here include the performance 
statistics, training curves, and model weights for all 1200 trials, resulting in 
2400 total models. The hyperparameter searches were orchestrated using the 
[SpaceRay](https://github.com/maxzvyagin/spaceray/) package. All data is currently
hosted publicly on Google Cloud Storage.

### *Note:*

*If you want a more in-depth understanding of the dataset and how it was created, 
check out our [paper](https://google.com) on the subject.*


