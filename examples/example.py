import os

import tensorflow.compat.v1 as tf
from nn_regressors import CNN, RNN, benchmark_model
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Lasso, ElasticNet
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

cnn_cpu_reg = CNN.CPURegressor(pretrained=True)
cnn_mem_reg = CNN.MemoryRegressor(pretrained=True)

# Instantiate example model.
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

# Both method calls return a Pandas dataframe containing predictions for each layer.
layer_cpu_time = cnn_cpu_reg.predict(mobilenet)
layer_mem_usage = cnn_mem_reg.predict(mobilenet)

print(layer_cpu_time.head())

# Print MSEs
print("===== Default evaluation =====")
print("MobileNet cpu MSE:", cnn_cpu_reg.evaluate(mobilenet))
print("MobileNet mem MSE:", cnn_mem_reg.evaluate(mobilenet))

print("ResNet cpu MSE:", cnn_cpu_reg.evaluate(resnet))
print("ResNet mem MSE:", cnn_mem_reg.evaluate(resnet))

print("MobileNet cpu MASE:", cnn_cpu_reg.evaluate_mase(mobilenet))
print("MobileNet mem MASE:", cnn_mem_reg.evaluate_mase(mobilenet))

print("ResNet cpu MASE:", cnn_cpu_reg.evaluate_mase(resnet))
print("ResNet mem MASE:", cnn_mem_reg.evaluate_mase(resnet))

# Compare different regressor types
cnn_cpu_reg.compare(
    [RandomForestRegressor(n_estimators=1000, random_state=42), Lasso(), ElasticNet()],
    mobilenet,
)
cnn_mem_reg.compare(
    [RandomForestRegressor(n_estimators=1000, random_state=42), Lasso(), ElasticNet()],
    mobilenet,
)

cnn_cpu_reg.compare(
    [RandomForestRegressor(n_estimators=1000, random_state=42), Lasso(), ElasticNet()],
    resnet,
)
cnn_mem_reg.compare(
    [RandomForestRegressor(n_estimators=1000, random_state=42), Lasso(), ElasticNet()],
    resnet,
)
cnn_cpu_reg.compare(
    [RandomForestRegressor(n_estimators=1000, random_state=42), Lasso(), ElasticNet()],
    densenet,
)
cnn_mem_reg.compare(
    [RandomForestRegressor(n_estimators=1000, random_state=42), Lasso(), ElasticNet()],
    densenet,
)

cnn_cpu_reg.compare(
    [RandomForestRegressor(n_estimators=1000, random_state=42), Lasso(), ElasticNet()],
    xception,
)
cnn_mem_reg.compare(
    [RandomForestRegressor(n_estimators=1000, random_state=42), Lasso(), ElasticNet()],
    xception,
)

# Add new model data
print("===== Adding new data =====")
cnn_cpu_reg.add_model_data(mobilenet)
cnn_mem_reg.add_model_data(mobilenet)

cnn_cpu_reg.add_model_data(resnet)
cnn_mem_reg.add_model_data(resnet)

vgg16 = tf.keras.applications.vgg16.VGG16(include_top=True, weights="imagenet",)

inception = tf.keras.applications.inception_v3.InceptionV3(
    include_top=True, weights="imagenet",
)

cnn_cpu_reg.add_model_data(vgg16)
cnn_mem_reg.add_model_data(vgg16)

cnn_cpu_reg.add_model_data(inception)
cnn_mem_reg.add_model_data(inception)

print(cnn_cpu_reg.state)

# Fit regressors to new model data.
cnn_cpu_reg.fit()
cnn_mem_reg.fit()

print("===== New evaluation =====")
# print("MobileNet cpu MSE:", cnn_cpu_reg.evaluate(mobilenet))
# print("MobileNet mem MSE:", cnn_mem_reg.evaluate(mobilenet))

# print("ResNet cpu MSE:", cnn_cpu_reg.evaluate(resnet))
# print("ResNet mem MSE:", cnn_mem_reg.evaluate(resnet))

# print("MobileNet cpu MASE:", cnn_cpu_reg.evaluate_mase(mobilenet))
# print("MobileNet mem MASE:", cnn_mem_reg.evaluate_mase(mobilenet))

# print("ResNet cpu MASE:", cnn_cpu_reg.evaluate_mase(resnet))
# print("ResNet mem MASE:", cnn_mem_reg.evaluate_mase(resnet))

# cnn_cpu_reg.compare(
#     [RandomForestRegressor(n_estimators=1000, random_state=42), Lasso(), ElasticNet()],
#     mobilenet,
# )
# cnn_mem_reg.compare(
#     [RandomForestRegressor(n_estimators=1000, random_state=42), Lasso(), ElasticNet()],
#     mobilenet,
# )

# cnn_cpu_reg.compare(
#     [RandomForestRegressor(n_estimators=1000, random_state=42), Lasso(), ElasticNet()],
#     resnet,
# )
# cnn_mem_reg.compare(
#     [RandomForestRegressor(n_estimators=1000, random_state=42), Lasso(), ElasticNet()],
#     resnet,
# )


cnn_cpu_reg.compare(
    [RandomForestRegressor(n_estimators=1000, random_state=42), Lasso(), ElasticNet()],
    densenet,
)
cnn_mem_reg.compare(
    [RandomForestRegressor(n_estimators=1000, random_state=42), Lasso(), ElasticNet()],
    densenet,
)

cnn_cpu_reg.compare(
    [RandomForestRegressor(n_estimators=1000, random_state=42), Lasso(), ElasticNet()],
    xception,
)
cnn_mem_reg.compare(
    [RandomForestRegressor(n_estimators=1000, random_state=42), Lasso(), ElasticNet()],
    xception,
)


print("===== Saving =====")
cnn_cpu_reg.save()
cnn_mem_reg.save()

new_cnn_cpu_reg = CNN.CPURegressor()
new_cnn_mem_reg = CNN.MemoryRegressor()


# print("===== Loaded model evaluation =====")
# print("MobileNet cpu MSE:", cnn_cpu_reg.evaluate(mobilenet))
# print("MobileNet mem MSE:", cnn_mem_reg.evaluate(mobilenet))

# print("ResNet cpu MSE:", cnn_cpu_reg.evaluate(resnet))
# print("ResNet mem MSE:", cnn_mem_reg.evaluate(resnet))

clear_saved_regressors()
