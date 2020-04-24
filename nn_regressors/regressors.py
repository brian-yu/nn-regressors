import os
from enum import Enum
import pkg_resources

import pandas as pd
import numpy as np
from joblib import load, dump
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

from .utils import get_layer_features, preprocess
from .benchmark import get_benchmark_data, benchmark_model


class Architecture(Enum):
    NONE = 0
    CNN = 1
    RNN = 2


class RegressorType(Enum):
    NONE = 0
    CPU = 1
    MEM = 2


class RegressorState(Enum):
    DEFAULT = 1
    CUSTOM = 2


class Regressor:
    def __init__(self, pretrained, save_file, arch, reg_type):
        self.save_file = save_file
        self.regressor = None
        self.train_df = None
        self.arch = arch
        self.type = reg_type

        self.state = RegressorState.DEFAULT

        if pretrained:
            self.load_saved_regressor()

    def predict(self, model):
        layer_features = get_layer_features(model)
        cleaned_features = preprocess(layer_features, inference=True)

        X = cleaned_features.drop(
            ["name", "input_shape", "output_shape", "strides"], axis=1
        )  # Features

        predicted = self.regressor.predict(X)

        return pd.DataFrame(
            {"name": cleaned_features["name"], f"pred_{self.type.name}": predicted}
        )

    def fit(self, model=None, n_estimators=1000, seed=42):

        if self.type == RegressorType.CPU:
            target_column = "[avg ms]"
        elif self.type == RegressorType.MEM:
            target_column = "[mem KB]"
        else:
            raise Exception("Invalid Regressor Type")

        X = self.train_df.drop(
            ["name", "[avg ms]", "[mem KB]", "input_shape", "output_shape", "strides"],
            axis=1,
        )  # Features
        y = self.train_df[target_column]  # Labels

        # Split dataset into training set and test set
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=seed
        )  # 70% training and 30% test

        # Use supplied model if available.
        if model != None:
            self.regressor = model
        # Default to RF if no current model and no supplied model.
        elif not self.regressor:
            self.regressor = RandomForestRegressor(
                n_estimators=n_estimators, random_state=seed
            )
        # Train the model on training data
        self.regressor.fit(X_train, y_train)

        return X_test, y_test

    def _get_eval_data(self, model):
        pred = self.predict(model)
        benchmark = benchmark_model(model)

        pred_col = f"pred_{self.type.name}"

        if self.type == RegressorType.CPU:
            actual_col = "[avg ms]"
            get_col = lambda x: x.sum()
        elif self.type == RegressorType.MEM:
            actual_col = "[mem KB]"
            get_col = lambda x: x.max()

        actual = get_col(benchmark[["name", actual_col]].groupby("name"))

        return pred_col, actual_col, actual.merge(pred, on="name")

    def evaluate_mse(self, model):
        pred_col, actual_col, df = self._get_eval_data(model)
        return mean_squared_error(df[pred_col], df[actual_col])

    def compare(self, models, model_to_predict):
        old_model = self.regressor
        mse = []
        mase = []

        model_names = [type(model).__name__ for model in models]
        for model in models:
            self.fit(model=model)
            mse.append(self.evaluate_mse(model_to_predict))
            mase.append(self.evaluate_mase(model_to_predict))
        self.regressor = old_model

        df = pd.DataFrame({"Model": model_names, "MSE": mse, "MASE": mase,})

        print(f"Comparing regression models for {model_to_predict.name} {self.type}")
        print(df.head(len(models)))

    # Alias for evalute_mse
    def evaluate(self, model):
        return self.evaluate_mse(model)

    # See https://stats.stackexchange.com/a/108963
    def evaluate_mase(self, model):
        pred_col, actual_col, df = self._get_eval_data(model)
        errors = np.abs(df[actual_col] - df[pred_col])
        train_mean = self.train_df[actual_col].mean()
        train_diff = np.abs(self.train_df[actual_col] - train_mean)
        den = train_diff.sum() / self.train_df.shape[0]
        return np.abs(errors / den).mean()

    def add_model_data(self, model):
        new_data = get_benchmark_data(model)
        if self.state == RegressorState.DEFAULT:
            self.state = RegressorState.CUSTOM
        self.train_df = pd.concat([self.train_df, new_data], sort=False)

    def save(self, file=None):
        if self.state == RegressorState.DEFAULT:
            raise Exception("You don't need to save the default regressor.")

        if not file:
            file = self.save_file

        dump({"model": self.regressor, "train_data": self.train_df,}, file)

    def load_saved_regressor(self):
        if not self.save_file:
            raise Exception("Save file must not be None.")

        file = self.save_file
        if os.path.exists(file):
            loaded = load(self.save_file)
        else:
            reg_file = pkg_resources.resource_filename("nn_regressors", self.save_file)
            loaded = load(reg_file)

        self.regressor = loaded["model"]
        self.train_df = loaded["train_data"]


class CNN:
    @staticmethod
    def CPURegressor(pretrained=False, save_file="cnn_cpu.joblib"):
        return Regressor(pretrained, save_file, Architecture.CNN, RegressorType.CPU)

    @staticmethod
    def MemoryRegressor(pretrained=False, save_file="cnn_mem.joblib"):
        return Regressor(pretrained, save_file, Architecture.CNN, RegressorType.MEM)


class RNN:
    @staticmethod
    def CPURegressor(pretrained=False, save_file="rnn_cpu.joblib"):
        return Regressor(pretrained, save_file, Architecture.RNN, RegressorType.CPU)

    @staticmethod
    def MemoryRegressor(pretrained=False, save_file="rnn_mem.joblib"):
        return Regressor(pretrained, save_file, Architecture.RNN, RegressorType.MEM)
