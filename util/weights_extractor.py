# This utility file should read in a .tflite file and extract the weights and output them in a numpy array?

import tensorflow as tf
import numpy as np
from pathlib import Path
import os

# https://stackoverflow.com/questions/59559289/is-there-any-way-to-convert-a-tensorflow-lite-tflite-file-back-to-a-keras-fil/59566157#59566157



def print_weights(file_path: str | os.PathLike):
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



def weights_as_np(file_path: str | os.PathLike) -> dict[str, np.ndarray]:
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
  model_dir = Path("models")
  model_name = Path("temp_sensor_float16.tflite")
  model_path = model_dir / model_name

  # print_weights("models/temp_sensor_float16.tflite")

  weights = weights_as_np(model_path)
  np.savez(model_path.with_suffix(".npz"), **weights)

  loaded = np.load(model_path.with_suffix(".npz"))
  print(loaded.files)  # See all weight names
  print(loaded['arith.constant7']) 




