import tensorflow as tf
import argparse
import copy
import torchfile
import numpy as np
import os

from model import Model

CURRENT_DIR = os.path.dirname(__file__)

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str, default=os.path.join(CURRENT_DIR, '../dat'))
parser.add_argument('--checkpoint_dir', type=str, default=os.path.join(CURRENT_DIR, '../checkpoints'))
parser.add_argument('--output_dir', type=str, default=os.path.join(CURRENT_DIR, '../out'))
parser.add_argument('--log_dir', type=str, default=os.path.join(CURRENT_DIR, '../log'))

parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate step-size')
parser.add_argument('--image_dim', type=int, default=28*28)
parser.add_argument('--num_classes_per_ep', type=int, default=1, help='Number of classes per episode')
parser.add_argument('--num_support_points_per_class', type=int, default=1, help='Number of support points per class')
parser.add_argument('--num_query_points_per_class', type=int, default=10, help='Number of query points per class')


class PrototypicalNetwork(Model):

    def __init__(self, config):
        self.data_dir = config.data_dir
        self.checkpoint_dir = config.checkpoint_dir
        self.output_dir = config.output_dir
        self.log_dir = config.log_dir

        self.lr = config.lr
        self.image_dim = config.image_dim
        self.num_classes_per_ep = config.num_classes_per_ep
        self.num_support_points_per_class = config.num_support_points_per_class
        self.num_query_points_per_class = config.num_query_points_per_class
        self.config = copy.deepcopy(config)

        print('loaded data')
        self.load_data()
        self.add_placeholders()
        self.add_vars()
        self.pred = self.add_model(self.support_points_placeholder, self.query_points_placeholder)
        print('built graph')
        self.loss = self.add_loss_op(self.pred)
        self.train_op = self.add_training_op

    def load_data(self):
        """ We follow the procedure of Vinyals et al. [29] by resizing the grayscale images to 28 × 28 and
        augmenting the character classes with rotations in multiples of 90 degrees. We use 1200 characters
        plus rotations for training (4,800 classes in total) and the remaining classes, including rotations, for
        test.
        """
        # select 1200 characters (i.e. classes) for training set
        # select remaining characters for test set
        # TODO: change images to grayscale
        # TODO: 28x28
        # TODO: rotate and save

    def add_place_holders(self):
        support_points_per_ep = self.num_classes_per_ep * self.num_support_points_per_class
        query_points_per_ep   = self.num_classes_per_ep * self.num_query_points_per_class
        self.support_points_placeholder = tf.placeholder(tf.float32, shape=(support_points_per_ep, self.image_dim, name='support_points')
        self.support_labels_placeholder = tf.placeholder(tf.int32, shape=(support_points_per_ep), name='support_labels')
        self.query_points_placeholder = tf.placeholder(tf.float32, shape=(query_points_per_ep, self.image_dim), name='query_points')
        self.query_labels_placeholder = tf.placeholder(tf.int32, shape=(query_points_per_ep), name='query_labels')

    def create_feed_dict(self, support_points, support_labels, query_points, query_labels):
        feed_dict = {
            self.support_points_placeholder : support_points,
            self.support_labels_placeholder : support_labels,
            self.query_points_placeholder   : query_points,
            self.query_labels_placeholder   : query_labels
        }
        return feed_dict

    def add_embedding(self, images):
        with tf.variable_scope('conv1') as scope:
            conv1 = conv_bn_relu_pool(images, )
        with tf.variable_scope('conv2') as scope:
            conv2 = conv_bn_relu_pool(conv1, )
        with tf.variable_scope('conv3') as scope:
            conv3 = conv_bn_relu_pool(conv2, )
        with tf.variable_scope('conv4') as scope:
            conv4 = conv_bn_relu_pool(conv3, )
        return conv4

    def conv_bn_relu_pool(input, kernel_shape, bias_shape):
        weights = tf.get_variable("weights", kernel_shape, initializer=tf.random_normal_initializer())
        biases = tf.get_variable("biases", bias_shape, initializer=tf.constant_initializer(0.0))
        conv = tf.nn.conv2d(input, weights, strides=[1, 1, 1, 1], padding='SAME')
        relu = tf.relu(conv + biases)
        pool = tf.nn.max_pool(relu, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        return pool

    def add_model(self, support_points, support_labels, query_points):
        # compute embeddings of all examples
        with tf.variable_scope('embedding') as scope:
            support_embedding = self.add_embedding(support_points)
            scope.reuse_variables()
            query_embedding = self.add_embedding(query_points)
        # create prototypes for each class
        class_prototypes = 
        with tf.variable_scope('prototype') as scope:
            for i in range(self.num_classes_per_ep):
                class_embeddings = tf.slice(support_embedding, )
                prototype = tf.reduce_mean(support_embedding, axis=0)
        with tf.variable_scope('distance') as scope:
            # for each query point compute distance to each prototype
            # this will be 3d: N_q x N_k x N_f
            # ^ first duplicate each q point N_k times: N_q x N_f => N_qx N_k x N_f
            # use broadcasting to subtract N_k i.e. class prototypes => N_q x N_k x N_f
            # then take the norm over the 3rd dimension => N_q x N_k
            # ^ turn that to * -1 before softmax
            # then take the softmax over the 2nd dimension
            # n.m you can just use product of matrices as eulidean dist. TODO: probably want to test it
        return support_embedding, query_embedding

    def add_loss_op(self, support_embedding, query_embedding):
        self.
        loss = 
        return loss


if __name__ == "__main__":
    config = parser.parse_args()
    print('creating model')
    model = PrototypicalNetwork(config)




