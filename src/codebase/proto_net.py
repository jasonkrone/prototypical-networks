import tensorflow as tf
import argparse
import copy
import numpy as np
import os
from datetime import datetime

from model import Model
from OmniglotGenerator import OmniglotGenerator

CURRENT_DIR = os.path.dirname(__file__)

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str, default=os.path.join('/home/jason/datasets/omniglot_images'))
parser.add_argument('--checkpoint_dir', type=str, default=os.path.join(CURRENT_DIR, '../checkpoints'))
parser.add_argument('--output_dir', type=str, default=os.path.join(CURRENT_DIR, '../out'))
parser.add_argument('--log_dir', type=str, default=os.path.join(CURRENT_DIR, '../log'))

parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate step-size')
# TODO: cut learning rate in half every 2000 episodes
parser.add_argument('--num_max_episodes', type=int, default=2000*6)
parser.add_argument('--image_dim', type=int, default=(28, 28))
parser.add_argument('--num_classes_per_ep', type=int, default=60, help='Number of classes per episode')
parser.add_argument('--num_support_points_per_class', type=int, default=5, help='Number of support points per class')
parser.add_argument('--num_query_points_per_class', type=int, default=5, help='Number of query points per class')


class PrototypicalNetwork(Model):

    def __init__(self, config):
        date_time = datetime.today().strftime('%Y%m%d_%H%M%S')
        hyper_params = 'lr_'+str(config.lr)+'max_ep_'+str(config.num_max_episodes)+\
                       'num_sprt_'+str(config.num_support_points_per_class)+\
                       'num_qry_'+str(config.num_query_points_per_class)
        suffix = '_' + date_time + '_' + hyper_params + '/'

        self.log_dir = config.log_dir + suffix
        self.checkpoint_dir = config.checkpoint_dir + suffix
        self.output_dir = config.output_dir + suffix
        self.data_dir = config.data_dir

        self.lr = config.lr
        self.num_max_episodes = config.num_max_episodes
        self.image_dim = config.image_dim
        self.num_classes_per_ep = config.num_classes_per_ep
        self.num_support_points_per_class = config.num_support_points_per_class
        self.num_query_points_per_class = config.num_query_points_per_class
        self.config = copy.deepcopy(config)

        self.load_data()
        self.add_placeholders()
        data = (self.support_points_placeholder, self.query_points_placeholder, self.support_labels_placeholder, \
                self.is_training_placeholder, self.query_labels_placeholder)
        self.predictions, self.accuracy_op = self.predict(*data)
        self.loss, self.loss_summary = self.add_loss_op(distances, query_labels)
        self.train_op = self.add_training_op(loss)

    def load_data(self):
        self.data_generator = OmniglotGenerator(self.data_dir, self.num_max_episodes, self.num_classes_per_ep, \
                                                self.num_support_points_per_class, self.num_query_points_per_class)

    def add_placeholders(self):
        support_points_per_ep = self.num_classes_per_ep * self.num_support_points_per_class
        query_points_per_ep   = self.num_classes_per_ep * self.num_query_points_per_class
        height, width = self.image_dim
        self.support_points_placeholder = tf.placeholder(tf.float32, shape=(support_points_per_ep, height, width, 1), name='support_points')
        self.support_labels_placeholder = tf.placeholder(tf.int32, shape=(support_points_per_ep), name='support_labels')
        self.query_points_placeholder = tf.placeholder(tf.float32, shape=(query_points_per_ep, height, width, 1), name='query_points')
        self.query_labels_placeholder = tf.placeholder(tf.int32, shape=(query_points_per_ep), name='query_labels')
        self.is_training_placeholder = tf.placeholder(tf.bool, name='is_training')

    def create_feed_dict(self, support_points, support_labels, query_points, query_labels, is_training):
        feed_dict = {
            self.support_points_placeholder : support_points,
            self.support_labels_placeholder : support_labels,
            self.query_points_placeholder   : query_points,
            self.query_labels_placeholder   : query_labels,
            self.is_training_placeholder    : is_training
        }
        return feed_dict

    def add_embedding(self, images, is_training):
        # in dim: Nx28x28x1 => outdim: Nx14x14x64
        with tf.variable_scope('conv1') as scope:
            conv1 = self.conv_bn_relu_pool(images, is_training, [3, 3, 1, 64], [64])
        # in dim: Nx14x14x64 => outdim: Nx7x7x64
        with tf.variable_scope('conv2') as scope:
            conv2 = self.conv_bn_relu_pool(conv1, is_training, [3, 3, 64, 64], [64])
        # in dim: Nx7x7x64 => outdim: Nx3x3x64
        with tf.variable_scope('conv3') as scope:
            conv3 = self.conv_bn_relu_pool(conv2, is_training, [3, 3, 64, 64], [64])
        # in dim: Nx3x3x64 => outdim: Nx1x1x64
        with tf.variable_scope('conv4') as scope:
            conv4 = self.conv_bn_relu_pool(conv3, is_training, [3, 3, 64, 64], [64])
        return conv4

    def conv_bn_relu_pool(self, input, is_training, kernel_shape, bias_shape):
        weights = tf.get_variable("weights", kernel_shape, initializer=tf.random_normal_initializer())
        biases = tf.get_variable("biases", bias_shape, initializer=tf.constant_initializer(0.0))
        conv = tf.nn.conv2d(input, weights, strides=[1, 1, 1, 1], padding='SAME')
        # TODO: double check that batchnorm is being reused for a given layer
        batch_norm = tf.layers.batch_normalization(conv + biases, taining=is_training, name='batch_norm')
        relu = tf.relu(batch_norm)
        pool = tf.nn.max_pool(relu, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
        return pool

    def add_model(self, support_points, support_labels, query_points, is_training):
        # compute embeddings of all examples
        with tf.variable_scope('embedding') as scope:
            support_embedding = self.add_embedding(support_points, is_training)
            scope.reuse_variables()
            query_embedding = self.add_embedding(query_points, is_training)

        # create prototypes for each class
        ones = tf.ones_like(support_embedding)
        per_class_embedding_sum = tf.unsorted_segment_sum(support_embedding, support_labels, self.num_classes_per_ep, name='embedding_sum')
        class_counts = tf.unsorted_segment_sum(ones, support_labels, self.num_classes_per_ep)
        class_prototypes = per_class_embedding_sum / class_counts

        #dist = np.sqrt(np.sum(X**2, axis=1).reshape(-1, 1) - 2*X.dot(self.X_train.T) + np.sum(self.X_train**2, axis=1))
        # dists[i, j] is the Euclidean distance between the ith test point and the jth trainin
        # dists[i, j] distance between ith query_point and jth prototype
        query_square_sum = tf.reshape(tf.reduce_sum(tf.square(query_embedding), 1), shape=[-1, 1])
        proto_square_sum = tf.reduce_sum(tf.square(class_prototypes), 1)
        distances = tf.add(query_square_sum, proto_square_sum) - 2*tf.matmul(query_embedding, class_prototypes, transpose_b=True)
        return distances

    def add_training_op(self, loss):
        # TODO: check this is the right optimizer
        optimizer = tf.train.AdamOptimizer(self.lr)
        global_step = tf.get_variable("global_step", [1], dtype=tf.int32, trainable=False, initializer=tf.zeros_initializer)
        train_op = optimizer.minimize(loss, global_step=global_step)
        return train_op

    def add_loss_op(self, distances, query_labels):
        # this won't work right now because of the labels values i.e. we don't produce logits
        # for each label they're only for the num_classes_per_ep
        # TODO: make sure the sign on the distances is correct
        entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=query_labels, logits=distances*-1.0)
        loss = tf.reduce_mean(entropy, name='loss')
        # not sure exactly what this does
        with tf.name_scope('summaries'):
            tf.summary.scalar('loss', loss)
            tf.summary.histogram('histogram loss', loss)
            summary_op = tf.summary.merge_all()
        return loss, summary_op

    def fit(self, sess):
        """Fit model on provided data.

        Args:
          sess: tf.Session()
          input_data: np.ndarray of shape (n_samples, n_features)
          input_labels: np.ndarray of shape (n_samples, n_classes)
        Returns:
          losses: list of loss per epoch
        """
        losses = []
        self.summary_writer = tf.summary.FileWriter(self.log_dir, graph=tf.get_default_graph())
        for i, ((sprt_label_batch, sprt_batch), (qry_label_batch, qry_batch)) in enumerate(self.data_generator):
            feed_dict = self.create_feed_dict(sprt_batch, sprt_label_batch, qry_batch, qry_label_batch, True)
            _, loss, summary = sess.run([self.train_op, self.loss, self.loss_summary], feed_dict=feed_dict)
            self.summary_writer.add_summary(summary, i)
            losses.append(loss)
        return losses

    def predict(self, support_points, query_points, support_labels, is_training, query_labels=None):
        # TODO: set is_training to false. Maybe return accuracy
        """Make predictions from the provided model.
        Args:
          sess: tf.Session()
          input_data: np.ndarray of shape (n_samples, n_features)
          input_labels: np.ndarray of shape (n_samples, n_classes)
        Returns:
          average_loss: Average loss of model.
          predictions: Predictions of model on input_data
        """
        distances = self.add_model(support_points, query_points, support_labels, is_training)
        predictions = tf.argmax(distances, axis=1)
        if query_labels != None:
            accuracy_op = tf.metrics.accuracy(query_labels, predictions)
        return predictions, accuracy_op

if __name__ == "__main__":
    config = parser.parse_args()
    net = PrototypicalNetwork(config)
    sess = tf.InteractiveSession()
    net.fit(sess)

