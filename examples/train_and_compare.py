import os

import tensorflow.compat.v1 as tf
from nn_regressors import CNN, RNN, benchmark_model
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Lasso, ElasticNet, Ridge
from sklearn.svm import SVR, NuSVR
from sklearn.ensemble import RandomForestRegressor
import lightgbm as lgb
import numpy as np


def clear_saved_regressors():
    to_delete = [
        "cnn_cpu.joblib",
        "cnn_mem.joblib",
        "rnn_cpu.joblib",
        "rnn_mem.joblib",
    ]
    for file in to_delete:
        if os.path.exists(file):
            os.remove(file)


clear_saved_regressors()

# Create regressors
cnn_cpu_reg = CNN.CPURegressor()
cnn_mem_reg = CNN.MemoryRegressor()

print("===== Adding new data =====")

# Instantiate example models.
mobilenet = tf.keras.applications.mobilenet_v2.MobileNetV2(
    include_top=True, weights="imagenet",
)
resnet = tf.keras.applications.resnet50.ResNet50(include_top=True, weights="imagenet",)
densenet = tf.keras.applications.densenet.DenseNet121(
    include_top=True, weights="imagenet",
)
xception = tf.keras.applications.xception.Xception(
    include_top=True, weights="imagenet",
)
vgg16 = tf.keras.applications.vgg16.VGG16(include_top=True, weights="imagenet",)

inception = tf.keras.applications.inception_v3.InceptionV3(
    include_top=True, weights="imagenet",
)

models = [mobilenet, resnet, densenet, xception, vgg16, inception]
for model in models:
    cnn_cpu_reg.add_model_data(model)
    cnn_mem_reg.add_model_data(model)

print("===== Fitting =====")

# Fit regressors to new model data.
cnn_cpu_reg.fit(model=SVR())  # Fit an SvR model
cnn_mem_reg.fit()  # default to Random Forest

# Compare different regressor types

# Defines custom loss function for LGBM that penalizes underestimates.
# See https://towardsdatascience.com/custom-loss-functions-for-gradient-boosting-f79c1b40466d
def custom_asymmetric_train(y_true, y_pred):
    residual = (y_true - y_pred).astype("float")
    grad = np.where(residual > 0, -2 * 10.0 * residual, -2 * residual)
    hess = np.where(residual > 0, 2 * 10.0, 2.0)
    return grad, hess


gbm = lgb.LGBMRegressor()
gbm.set_params(
    **{"objective": custom_asymmetric_train, "num_leaves": 70, "max_bin": 400}, metrics=["mse", "mae"]
)


def get_regressors():
    return [
        Ridge(),
        RandomForestRegressor(n_estimators=750, random_state=42, criterion="mae"),
        Lasso(),
        ElasticNet(),
        SVR(),
        NuSVR(),
        gbm,
    ]


print("===== Evaluating on fitted models =====")
mobilenet_cpu_eval = cnn_cpu_reg.compare(get_regressors(), mobilenet)
resnet_cpu_eval = cnn_cpu_reg.compare(get_regressors(), resnet)
densenet_cpu_eval = cnn_cpu_reg.compare(get_regressors(), densenet)
xception_cpu_eval = cnn_cpu_reg.compare(get_regressors(), xception)
vgg16_cpu_eval = cnn_cpu_reg.compare(get_regressors(), vgg16)
inception_cpu_eval = cnn_cpu_reg.compare(get_regressors(), inception)

mobilenet_mem_eval = cnn_mem_reg.compare(get_regressors(), mobilenet)
resnet_mem_eval = cnn_mem_reg.compare(get_regressors(), resnet)
densenet_mem_eval = cnn_mem_reg.compare(get_regressors(), densenet)
xception_mem_eval = cnn_mem_reg.compare(get_regressors(), xception)
vgg16_mem_eval = cnn_mem_reg.compare(get_regressors(), vgg16)
inception_mem_eval = cnn_mem_reg.compare(get_regressors(), inception)


# Test on new models
print("===== Evaluating on new models =====")

inception_resnet = tf.keras.applications.inception_resnet_v2.InceptionResNetV2(
    include_top=True, weights="imagenet",
)
nasnet = tf.keras.applications.nasnet.NASNetLarge(include_top=True, weights="imagenet",)

inception_resnet_cpu_eval = cnn_cpu_reg.compare(get_regressors(), inception_resnet)
nasnet_cpu_eval = cnn_cpu_reg.compare(get_regressors(), nasnet)


inception_resnet_mem_eval = cnn_mem_reg.compare(get_regressors(), inception_resnet)
nasnet_mem_eval = cnn_mem_reg.compare(get_regressors(), nasnet)
