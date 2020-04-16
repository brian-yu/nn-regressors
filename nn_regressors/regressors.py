from enum import Enum
import pkg_resources

import pandas as pd
from joblib import load

def get_layer_features(model):
    layers = pd.DataFrame()
    layers['name'] = pd.Series([layer.name for layer in model.layers])
    layers['input_shape'] = pd.Series([layer.input_shape for layer in model.layers])
    layers['output_shape'] = pd.Series([layer.output_shape for layer in model.layers])

    features = ['units','filters','activation','strides','kernel_size']
    for feature in features:
        layers[feature] = pd.Series(
            [layer.get_config()[feature] if feature in layer.get_config() else None for layer in model.layers])
    return layers

class Architecture(Enum):
    NONE = 0
    CNN = 1
    RNN = 2

class RegressorType(Enum):
    NONE = 0
    CPU = 1
    MEM = 2

class Regressor():
    def __init__(self, regressor, arch, reg_type):
        self.regressor = regressor
        self.arch = arch
        self.type = reg_type

    def predict(self, model):
        layer_features = get_layer_features(model)
        cleaned_features = clean(layer_features, inference=True)
        
        X = cleaned_features.drop(['name', 'input_shape', 'output_shape', 'strides'], axis=1)  # Features
        
        predicted = self.regressor.predict(X)
        
        return pd.DataFrame({'name': cleaned_features['name'], f'pred_{self.type}': predicted})

class CNNRegressor():

    @staticmethod
    def CPU():
        reg_file = pkg_resources.resource_filename(
            'nn_regressors', 'cnn_cpu.joblib'
        )
        reg = load(reg_file)
        return Regressor(reg, Architecture.CNN, RegressorType.CPU)
    
    @staticmethod
    def Memory():
        reg_file = pkg_resources.resource_filename(
            'nn_regressors', 'cnn_mem.joblib'
        )
        reg = load(reg_file)
        return Regressor(reg, Architecture.CNN, RegressorType.MEM)

class RNNRegressor():

    @staticmethod
    def CPU():
        reg_file = pkg_resources.resource_filename(
            'nn_regressors', 'rnn_cpu.joblib'
        )
        reg = load(reg_file)
        return Regressor(reg, Architecture.RNN, RegressorType.CPU)
    
    @staticmethod
    def Memory():
        reg_file = pkg_resources.resource_filename(
            'nn_regressors', 'rnn_mem.joblib'
        )
        reg = load(reg_file)
        return Regressor(reg, Architecture.RNN, RegressorType.MEM)
