import os

import tensorflow.compat.v1 as tf
from nn_regressors import CNN, RNN, benchmark_model
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Lasso, ElasticNet, Ridge
from sklearn.svm import SVR, NuSVR
from sklearn.ensemble import RandomForestRegressor


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
def get_regressors():
    return [
        Ridge(),
        RandomForestRegressor(n_estimators=1000, random_state=42),
        Lasso(),
        ElasticNet(),
        SVR(),
        NuSVR(),
    ]


print("===== Evaluating on fitted models =====")
cnn_cpu_reg.compare(get_regressors(), mobilenet)
cnn_cpu_reg.compare(get_regressors(), resnet)
cnn_cpu_reg.compare(get_regressors(), densenet)
cnn_cpu_reg.compare(get_regressors(), xception)

cnn_mem_reg.compare(get_regressors(), mobilenet)
cnn_mem_reg.compare(get_regressors(), resnet)
cnn_mem_reg.compare(get_regressors(), densenet)
cnn_mem_reg.compare(get_regressors(), xception)


# Test on new models
print("===== Evaluating on new models =====")

inception_resnet = tf.keras.applications.inception_resnet_v2.InceptionResNetV2(
    include_top=True, weights="imagenet",
)
nasnet = tf.keras.applications.nasnet.NASNetLarge(include_top=True, weights="imagenet",)

cnn_cpu_reg.compare(get_regressors(), inception_resnet)
cnn_cpu_reg.compare(get_regressors(), nasnet)

cnn_mem_reg.compare(get_regressors(), inception_resnet)
cnn_mem_reg.compare(get_regressors(), nasnet)
