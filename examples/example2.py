import tensorflow.compat.v1 as tf
from nn_regressors import CNN

# Create Regressor instances
cnn_cpu_reg = CNN.CPURegressor()
cnn_mem_reg = CNN.MemoryRegressor()

# Load VGG model.
tf.keras.backend.clear_session() # IMPORTANT for layer names to match up.
vgg16_model = tf.keras.applications.vgg16.VGG16(
    include_top=True,
    weights='imagenet')

# Add vgg model data
cnn_cpu_reg.add_model_data(vgg16_model)
cnn_mem_reg.add_model_data(vgg16_model)

# Load inception model.
tf.keras.backend.clear_session() # IMPORTANT for layer names to match up.
inception_model = tf. keras.applications.inception_v3.InceptionV3(
    include_top=True, weights='imagenet')

# Add inception model data
cnn_cpu_reg.add_model_data(inception_model)
cnn_mem_reg.add_model_data(inception_model)

# Fit regressors to new model data.
cnn_cpu_reg.fit()
cnn_mem_reg.fit()

## Evaluate on new model.
mobilenet = tf.keras.applications.mobilenet_v2.MobileNetV2(
    include_top=True,
    weights='imagenet')

# Print MSEs of regression models.
print(cnn_cpu_reg.evaluate(mobilenet))
print(cnn_mem_reg.evaluate(mobilenet))

# You can also save the model state.
# After calling save() with no arguments, future instances
# will automatically load the latest saved model.
cnn_cpu_reg.save()
cnn_mem_reg.save()

load_prev_save_cpu_reg = CNN.CPURegressor()

# If you call save(filename) with an arg, it will save to that specific
# file, which you can load later.
cnn_cpu_reg.save('reg1.joblib')
new_cnn_cpu_reg = CNN.CPURegressor(save_file='reg1.joblib')