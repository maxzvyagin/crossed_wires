import requests
import pickle
from io import BytesIO
import pandas as pd


class ModelWeightDataset:
    def __init__(self, filename, num_spaces_searched):
        self.file_name = filename
        self.num_spaces_searched = num_spaces_searched
        self.baseline_url = None
        self.optimizer_results = None
        self.ray_tune_dfs = None

    @property
    def dataframe(self):
        if not hasattr(self, "dataframe"):
            self.dataframe = pd.read_csv(self.file_name)
        return self.dataframe

    @property
    def optimizer_result(self, num=None):
        if hasattr(self.optimizer_results):
            if not num:
                # return all
                return self.optimizer_results
            else:
                return self.optimizer_results[num]
        else:
            self.optimizer_results = []
            # get each byte stream of optimizer results from google cloud storage
            for i in range(self.num_spaces_searched):
                optimizer_bytes = requests.get(
                    self.baseline_url + "/optimizer_result{}.pkl".format(i)
                ).content
                optimizer = pickle.loads(optimizer_bytes)
                self.optimizer_results.append(optimizer)
            if not num:
                # return all
                return self.optimizer_results
            else:
                return self.optimizer_results[num]

    @property
    def ray_tune_dataframes(self, num=None):
        if hasattr(self.ray_tune_dfs):
            if not num:
                # return all
                return self.ray_tune_dfs
            else:
                return self.optimizer_results[num]
        else:
            self.ray_tune_dfs = []
            # get the csvs
            for i in range(self.num_spaces_searched):
                df_bytes = requests.get(
                    self.baseline_url + "/space{}.csv".format(i)
                ).content
                df = pd.read_csv(BytesIO(df_bytes))
                self.ray_tune_dfs.append(df)
            if not num:
                # return all
                return self.ray_tune_dfs
            else:
                return self.ray_tune_dfs[num]


class TrialResult:
    def __init__(self, name):
        self.name = name
        self.url = name

    # @property
    # def pytorch_weights(self):
    #     if not hasattr(self, "pytorch_weights"):
    #         # load the pt model from google cloud storage
    #
