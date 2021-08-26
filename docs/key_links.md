# Key Links

## Full Dataset Download
If you wish to download the dataset in its entirety, you are welcome to do so
using this link: <https://storage.googleapis.com/crossed-wires-dataset/full_cifar10_results.zip>. 
The full dataset size is approximately 340 GB expanded, and 324 GB compressed.
It is licensed under CC-BY-4.0.

## Weights and Biases - Dataset Trial Logs
All experiments were tracked using [Weights and Biases](https://wandb.ai/). These
links include some extra information that can be helpful, such as training loss curves.

You can find the raw logging information for each trial on its respective project page:
- [VGG16](https://wandb.ai/mzvyagin/vgg_cifar10_lambda_comparison?workspace=user-mzvyagin)
- [ResNet50](https://wandb.ai/mzvyagin/resnet_cifar10_lambda_comparison?workspace=user-mzvyagin)
- [DenseNet121](https://wandb.ai/mzvyagin/densenet_cifar10_lambda_comparison?workspace=user-mzvyagin)

## Supporting Libraries and Research
This project would not have been possible without extensive previous libraries and research. 
In particular, we relied heavily upon these resources:
- [Ray Tune](https://docs.ray.io/en/latest/tune/index.html): a highly effective 
hyperparameter optimization package which automated our search and provided
  extremely helpful resource and trial management.
- [HyperSpace](https://hyperspace.readthedocs.io/en/latest/): a search algorithm
  implemented by Todd Young for more effective hyperparameter searches. This 
  original approach is integrated with Ray Tune in the 
  [SpaceRay](https://github.com/maxzvyagin/spaceray) package.
- ANYTHING ELSE HERE??