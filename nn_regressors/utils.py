import pandas as pd

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

def flatten_shape(shape):
    if not shape:
        return None
    
    def reduce(tup):
        acc = 1
        for val in tup:
            if val:
                acc *= val
        return acc
    
    if isinstance(shape, list):
        return sum(reduce(tup) for tup in shape)
    
    return reduce(shape)

def clean(data, inference=False):
    if inference:
        cleaned = pd.get_dummies(data, columns=['activation'], dummy_na=True)
    else:
        cleaned = pd.get_dummies(
            data.dropna(subset=['[avg ms]', '[mem KB]']), columns=['activation'], dummy_na=True)
    
    for activation in ['selu', 'elu', 'softmax', 'softplus', 'softsign', 'relu', 'tanh', 'sigmoid', 'hard_sigmoid', 'exponential', 'linear']:
        col = f"activation_{activation}"
        if col not in cleaned.columns:
            cleaned[col] = pd.Series(0)
    

    cleaned['input_size'] = cleaned['input_shape'].apply(flatten_shape)
    cleaned['output_size'] = cleaned['output_shape'].apply(flatten_shape)
    cleaned['stride_size'] = cleaned['strides'].apply(flatten_shape)
    cleaned['kernel_size'] = cleaned['kernel_size'].apply(flatten_shape)

    return cleaned.fillna(-1)