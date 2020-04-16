from enum import Enum
import pkg_resources

import pandas as pd
from joblib import load

from .utils import get_layer_features, preprocess
from .benchmark import get_benchmark_data

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

class Regressor():
    def __init__(self, save_file, arch, reg_type):
        self.save_file = save_file
        self.regressor = self.load_saved_regressor()
        self.arch = arch
        self.type = reg_type

        self.state = RegressorState.DEFAULT
        self.train_df = None

    def predict(self, model):
        layer_features = get_layer_features(model)
        cleaned_features = preprocess(layer_features, inference=True)
        
        X = cleaned_features.drop(['name', 'input_shape', 'output_shape', 'strides'], axis=1)  # Features
        
        predicted = self.regressor.predict(X)

        return pd.DataFrame({'name': cleaned_features['name'], f'pred_{self.type}': predicted})
    
    def fit(self, n_estimators=1000, seed=42):

        if self.type == RegressorType.CPU:
            target_column = '[avg ms]'
        elif self.type == RegressorType.MEM:
            target_column = '[mem KB]'
        else:
            raise Exception("Invalud Regerssor Type")

        X = self.train_df.drop(['name', '[avg ms]', '[mem KB]', 'input_shape', 'output_shape', 'strides'], axis=1)  # Features
        y = self.train_df[target_column]  # Labels

        # Split dataset into training set and test set
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state = RANDOM_SEED) # 70% training and 30% test

        # Instantiate model with 1000 decision trees
        self.regerssor = RandomForestRegressor(n_estimators = n_estimators, random_state = seed)
        # Train the model on training data
        self.regressor.fit(X_train, y_train);

        return X_test, y_test
    
    def add_model_data(self, model):
        new_data = get_benchmark_data(model)

        if not self.train_df:
            self.state = RegressorState.CUSTOM
            self.train_df = new_data
        else:
            self.train_df = pd.concat([self.train_df, new_data])
    
    def save(self, file=None):
        if self.state == RegressorState.DEFAULT:
            raise Exception("You don't need to save the default regressor.")
        
        if not file:
            file = self.save_file

        dump(self.regressor, file) 
    
    def load_saved_regressor(self):
        if not self.save_file:
            raise Exception("Save file must not be None.")

        if os.path.exists(save_file):
            return load(save_file)

        reg_file = pkg_resources.resource_filename(
            'nn_regressors', save_file
        )
        return load(reg_file)
        


class CNN():

    @staticmethod
    def CPURegressor(save_file='cnn_cpu.joblib'):
        return Regressor(
            save_file, Architecture.CNN, RegressorType.CPU
        )
    
    @staticmethod
    def MemoryRegressor(save_file='cnn_mem.joblib'):
        return Regressor(
            save_file, Architecture.CNN, RegressorType.MEM
        )

class RNN():

    @staticmethod
    def CPURegressor(save_file='rnn_cpu.joblib'):
        return Regressor(
            save_file, Architecture.RNN, RegressorType.CPU
        )
    
    @staticmethod
    def MemoryRegressor(save_file='rnn_mem.joblib'):
        return Regressor(
            save_file, Architecture.RNN, RegressorType.MEM
        )
