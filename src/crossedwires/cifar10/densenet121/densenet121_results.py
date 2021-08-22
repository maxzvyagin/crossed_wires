from crossedwires.base_class import ModelWeightDataset
import torchvision.models as models
import tensorflow as tf
import requests
from os.path import exists
import torch
import os


class DenseNet121Dataset(ModelWeightDataset):
    # inheriting from base class, specialized to
    def __init__(
        self, filename="densenet121_cifar10_wandb_export.csv", num_spaces_searched=16
    ):
        super().__init__(self, filename, num_spaces_searched)
        self.baseline_url = "https://storage.googleapis.com/crossed-wires-dataset/cifar10/densenet_lambda"

    def trial_lookup(self, epochs, learning_rate, batch_size, adam_epsilon) -> list:
        # returns the name(s) of the trial which fits those criteria
        df = self.wandb_dataframe
        # filter the dataframe based off the attributes selected
        df = df[df["epochs"] == epochs]
        df = df[df["learning_rate"] == learning_rate]
        df = df[df["batch_size"] == batch_size]
        df = df[df["adam_epsilon"] == adam_epsilon]
        return list(df.Name)

    def get_pytorch_weights(self, trial_name):
        """Return state dict of specific trial's pytorch model"""
        weights_file_name = "/tmp/{}_{}.pt_model.pt".format(
            "densenet_lambda", trial_name
        )
        # get the weights either from cache or google cloud
        if exists(weights_file_name):
            weights = torch.load(weights_file_name)
        else:
            weights = requests.get(
                self.baseline_url + "/model_weights/{}.pt_model.pt".format(trial_name)
            ).content
            with open(weights_file_name, "w") as f:
                f.write(weights)
        # return the weights
        return weights

    def get_tensorflow_weights(self, trial_name):
        """Get the model weights/definitions from the google cloud storage, save to folder"""
        weights_file_name = "/tmp/{}_{}tf_model/".format("densenet_lambda", trial_name)
        if exists(weights_file_name):
            pass
        else:
            os.mkdir(weights_file_name)
            os.mkdir(weights_file_name + "/variables")
            # get each kind of file needed
            with open(weights_file_name + "/saved_model.pb", "w") as f:
                saved_model = requests.get(
                    self.baseline_url
                    + "/model_weights/{}tf_model/saved_model.pb".format(trial_name)
                ).content
                f.write(saved_model)
            with open(
                weights_file_name + "/variables/variables.data-00000-of-00001"
            ) as f:
                variables_data = requests.get(
                    self.baseline_url
                    + "/model_weights/{}tf_model/variables/variables.data-00000-of-00001".format(
                        trial_name
                    )
                ).content
                f.write(variables_data)
            with open(weights_file_name + "/variables/variables.index") as f:
                variables_index = requests.get(
                    self.baseline_url
                    + "/model_weights/{}tf_model/variables/variables.index".format(
                        trial_name
                    )
                ).content
                f.write(variables_index)
        print(
            "Tensorflow Model definition has been saved to {}. Feel free to call get_tensorflow_model(trial_name).".format(
                weights_file_name
            )
        )
        return

    def get_pytorch_model(self, trial_name):
        """Returns pretrained torch model in eval mode"""
        # define the pytorch model
        model = models.densenet121(pretrained=False, num_classes=10)
        weights = self.get_pytorch_weights(trial_name)
        model.load_state_dict(weights)
        model.eval()
        return model

    def get_tensorflow_model(self, trial_name):
        weights_file_name = "/tmp/{}_{}tf_model/".format("densenet_lambda", trial_name)
        if not exists(weights_file_name):
            self.get_tensorflow_weights(trial_name)
        model = tf.keras.models.load_model(weights_file_name)
        return model
