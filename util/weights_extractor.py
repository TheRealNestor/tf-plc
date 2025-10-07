# This utility file should read in a .tflite file and extract the weights and output them in a numpy array?

import tensorflow as tf
import numpy as np


# https://stackoverflow.com/questions/59559289/is-there-any-way-to-convert-a-tensorflow-lite-tflite-file-back-to-a-keras-fil/59566157#59566157



def print_weights(file_path: str):
  interpreter = tf.lite.Interpreter(model_path=file_path)
  all_tensor_details = interpreter.get_tensor_details()
  interpreter.allocate_tensors()

  for tensor_item in all_tensor_details:
    print("Weight %s:" % tensor_item["name"])
    try:
      print(interpreter.tensor(tensor_item["index"])())
    except ValueError as e:
      print(f"  Could not extract tensor data: {tensor_item['name']}")
      print(f"  Exception: {e}")



def weights_as_np(file_path: str):
  interpreter = tf.lite.Interpreter(model_path=file_path)
  all_tensor_details = interpreter.get_tensor_details()
  interpreter.allocate_tensors()

  weights_dict = {}
  for tensor_item in all_tensor_details:
    try:
      weights_dict[tensor_item["name"]] = interpreter.tensor(tensor_item["index"])()
    except ValueError as e:
      print(f"  Could not extract tensor data: {tensor_item['name']}")
      print(f"  Exception: {e}")

  return weights_dict  





if __name__ == "__main__":
  # print_weights("models/temp_sensor_float16.tflite")

  weights = weights_as_np("models/temp_sensor_float16.tflite")
  np.savez("weights.npz", **weights)


  loaded = np.load("weights.npz")
  print(loaded.files)  # See all weight names
  print(loaded['arith.constant7'])  








