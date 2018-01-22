"""Basic conv2d example."""

import argparse
import argparse
import sys
import tempfile

import google3
import numpy as np
import tensorflow as tf

from google3.third_party.tensorflow.python.framework import dtypes
from google3.third_party.tensorflow.python.ops import array_ops


def main(_):

  input_sizes = [1, 2, 3, 3]
  filter_sizes = [1, 1, 3, 3]
  strides = [1, 1, 1, 1]
  padding = "VALID"
  input_total_size = np.prod(input_sizes)
  filter_total_size = np.prod(filter_sizes)
  print("input_total_size:")
  print(input_total_size)
  print("filter_total_size:")
  print(filter_total_size)

  input = np.arange(
      1, input_total_size + 1, dtype=np.float32).reshape(input_sizes)
  filter = np.arange(
      1, filter_total_size + 1, dtype=np.float32).reshape(filter_sizes)

  # input = [[[[99.0], [99.0], [99.0]]]]
  # filter = [[[[ 1.0, 1.0, 1.0 ]]]]

  print("input:")
  print(input)
  print("filter:")
  print(filter)
  print("strides:")
  print(strides)

  session = tf.Session()
  out = tf.nn.conv2d(
      input, filter, strides=strides, padding=padding, data_format="NHWC")
  value = session.run(out)
  print("output:")
  print(value)


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
