from crossedwires.base_class import ModelWeightDataset
import torchvision.models as models
import tensorflow as tf
import requests


class DenseNet121Dataset(ModelWeightDataset):
    # inheriting from base class, specialized to
    def __init__(
        self, filename="densenet121_cifar10_wandb_export.csv", num_spaces_searched=16
    ):
        super().__init__(self, filename, num_spaces_searched)
        self.most_recent_pt_weights = None
        self.most_recent_tf_weights = None
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

    def get_pytorch_model(self, trial_name):
        """Returns pretrained torch model in eval mode"""
        # define the pytorch model
        model = models.densenet121(pretrained=False, num_classes=10)
        # get the weights either from cache or google cloud
        if trial_name == self.most_recent_pt_weights[0]:
            weights = self.most_recent_pt_weights[1]
        else:
            weights = requests.get(
                self.baseline_url + "/model_weights/{}.pt_model.pt"
            ).content
            self.most_recent_pt_weights = (trial_name, weights)
        model.load_state_dict(weights)
        model.eval()
        return model

    def get_pytorch_weights(self, trial_name):
        # return just the state dict of the model
        if trial_name == self.most_recent_pt_weights[0]:
            weights = self.most_recent_pt_weights[1]
        else:
            weights = requests.get(
                self.baseline_url + "/model_weights/{}.pt_model.pt"
            ).content
            self.most_recent_pt_weights = (trial_name, weights)
        return weights

    def get_tensorflow_weights(self, trial_name):
        """Returns pretrained torch model in eval mode"""
        # define the tensorflow model
        model = tf.keras.applications.densenet.DenseNet121(
            weights=None, input_shape=(3, 32, 32), classes=100
        )
        # get the weights either from cache or google cloud
        if trial_name == self.most_recent_pt_weights[0]:
            weights = self.most_recent_pt_weights[1]
        else:
            weights = requests.get(
                self.baseline_url + "/model_weights/{}.pt_model.pt"
            ).content
            self.most_recent_pt_weights = (trial_name, weights)
        model.load_state_dict(weights)
        model.eval()
        return model

    def get_tensorflow_model(self, trial_name):
        raise NotImplementedError
