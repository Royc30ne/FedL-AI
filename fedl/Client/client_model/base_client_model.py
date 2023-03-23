import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

from fedl.base_model import BaseModel
import numpy as np


IMAGE_SIZE = 28

def get_convolution_extractor_shape(filter_list):
    with tf.Graph().as_default():
        """Model function for CNN."""
        features = tf.placeholder(
            tf.float32, shape=[None, IMAGE_SIZE * IMAGE_SIZE], name='features')
        labels = tf.placeholder(tf.int64, shape=[None], name='labels')
        input_layer = tf.reshape(features, [-1, IMAGE_SIZE, IMAGE_SIZE, 1])
        conv1 = tf.layers.conv2d(
          inputs=input_layer,
          filters=filter_list[0],
          kernel_size=[5, 5],
          padding="same",
          activation=tf.nn.relu)
        pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)
        conv2 = tf.layers.conv2d(
            inputs=pool1,
            filters=filter_list[1],
            kernel_size=[5, 5],
            padding="same",
            activation=tf.nn.relu)
        pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)
    return pool2.get_shape().as_list()

# output is a new graph, we create it so it have 
# some layers frozen
# def get_retrain_graph():

class BaseClientModel(BaseModel):
    def __init__(self, seed, lr, num_classes, optimizer=None):
        self.num_classes = num_classes
        super(BaseClientModel, self).__init__(seed, lr)

    def create_model(self):
        return

    def process_x(self, raw_x_batch):
        return np.array(raw_x_batch)

    def process_y(self, raw_y_batch):
        return np.array(raw_y_batch)
        
