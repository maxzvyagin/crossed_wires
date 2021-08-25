# Overview

## Motivation

It can be of great benefit to machine learning practicioners to have multiple 
frameworks available for implementation and training of neural networks. However,
it also comes with challenges, the main of which being *switching* between these different
libraries. Critically, there can be vast accuracy differences between these frameworks,
even when every possible effort is made to maintain consistency. This phenomenon
can be greatly influenced by the hyperparameters selected, however, there is not
currently an understanding as to why this is the case.
There are plenty of [examples] 
(https://stackoverflow.com/search?q=pytorch+and+tensorflow+accuracy) 
of this on forums and support sites, 
but there hasn't been any formal study or data collected to our knowledge. 
The **CrossedWires** dataset is intended to close this gap, collecting extensive
model information using hyperparameter search methods. Both the search trajectories
and the resulting trained models are included in the dataset. We invite the machine
learning community to utilize the CrossedWires dataset to explore the potential 
mechanisms behind these inconsistencies, further the collective understanding of 
hyperparameter spaces for improved training stability, and build tools for greater
reliability and reproducibility in deep learning that eliminate these discrepancies.
The dataset is freely available at (https://github.com/maxzvyagin/crossedwires).

## Dataset Details

**If you want a more in-depth understanding of the dataset and how it was created, 
check out our [paper]() on the subject.**