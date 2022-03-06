import tensorflow as tf
import numpy as np
from tensorflow.keras.models import load_model
print("hi")
# SignModel = load_model('Models/action.h5')
# run_model = tf.function(lambda x: SignModel(x))
# # This is important, let's fix the input size.
# BATCH_SIZE = 32
# STEPS = 30
# INPUT_SIZE = 1662
# concrete_func = run_model.get_concrete_function(
#     tf.TensorSpec([BATCH_SIZE, STEPS, INPUT_SIZE], SignModel.inputs[0].dtype))

# # model directory.
# MODEL_DIR = 'Models/action.h5'
# SignModel.save(MODEL_DIR, save_format="tf", signatures=concrete_func)

# converter = tf.lite.TFLiteConverter.from_saved_model(MODEL_DIR)
# tflite_model = converter.convert()