import tensorflow as tf
from tensorflow.python import debug as tf_debug
import argparse
import copy
import numpy as np
import os
from datetime import datetime

from model import Model
from OmniglotGenerator import OmniglotGenerator
from MiniImagenetGenerator import MiniImagenetGenerator

CURRENT_DIR = os.path.dirname(__file__)

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str, default=os.path.join('/dvmm-filer2/datasets/ImageNet/train/'))
parser.add_argument('--checkpoint_dir', type=str, default=os.path.join(CURRENT_DIR, '../checkpoints'))
parser.add_argument('--output_dir', type=str, default=os.path.join(CURRENT_DIR, '../out'))
parser.add_argument('--log_dir', type=str, default=os.path.join(CURRENT_DIR, '../log'))
parser.add_argument('--checkpoint', type=str, default=None)
parser.add_argument('--unpause', action='store_true', default=False)

parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate step-size')
parser.add_argument('--num_max_episodes', type=int, default=2000*200)
parser.add_argument('--image_dim', type=int, default=(84, 84, 3))
parser.add_argument('--num_steps_per_checkpoint', type=int, default=100, help='Number of steps between checkpoints')
parser.add_argument('--num_train_classes', type=int, default=20, help='Number of classes per training episode')
parser.add_argument('--num_test_classes', type=int, default=5, help='Number of classes per test episode')
parser.add_argument('--num_support_points_per_class', type=int, default=5, help='Number of support points per class')
parser.add_argument('--num_query_points_per_class', type=int, default=15, help='Number of query points per class')


class PrototypicalNetwork(Model):

    def __init__(self, config):
        date_time = datetime.today().strftime('%Y%m%d_%H%M%S')
        hyper_params = 'lr_'+str(config.lr)+'_max_ep_'+str(config.num_max_episodes)+\
                       '_num_support_'+str(config.num_support_points_per_class)+\
                       '_num_query_'+str(config.num_query_points_per_class)+\
                       '_train_classes_'+str(config.num_train_classes)+\
                       '_test_classes_'+str(config.num_test_classes)+\
                       '_data_dir_'+str(config.data_dir)
        subdir = date_time + '_' + hyper_params

        self.log_dir = config.log_dir + '/' + subdir
        self.checkpoint_dir = config.checkpoint_dir + '/' + subdir
        self.output_dir = config.output_dir + '/' + subdir
        self.data_dir = config.data_dir
        self.checkpoint = config.checkpoint

        self.lr = config.lr
        self.num_max_episodes = config.num_max_episodes
        self.image_dim = config.image_dim

        self.num_train_classes = config.num_train_classes
        self.num_test_classes  = config.num_test_classes

        self.num_support_points_per_class = config.num_support_points_per_class
        self.num_query_points_per_class = config.num_query_points_per_class
        self.num_steps_per_checkpoint = config.num_steps_per_checkpoint
        self.config = copy.deepcopy(config)

        # set up model
        self.load_data()
        self.add_placeholders()
        data = (self.support_points_placeholder, self.support_labels_placeholder, \
                self.query_points_placeholder, self.is_training_placeholder)
        self.distances = self.add_model(*data)
        self.preds, self.accuracy_op = self.predict(self.distances, self.query_labels_placeholder)
        self.loss, self.loss_summary = self.add_loss_op(self.distances, self.query_labels_placeholder)
        self.train_op = self.add_training_op(self.loss)

    def load_data(self):
        #self.data_generator = OmniglotGenerator(self.data_dir, self.num_max_episodes, self.num_train_classes, self.num_test_classes, \
        #                                        self.num_support_points_per_class, self.num_query_points_per_class)
        self.data_generator = MiniImagenetGenerator(self.data_dir, self.num_max_episodes, self.num_support_points_per_class,\
                                                    self.num_query_points_per_class, self.num_train_classes, self.num_test_classes, \
                                                    num_val_classes=None, mode='train')

    def add_placeholders(self):
        height, width, channels = self.image_dim
        with tf.name_scope('data'):
            self.support_points_placeholder = tf.placeholder(tf.float32, shape=(None, height, width, channels), name='support_points')
            self.support_labels_placeholder = tf.placeholder(tf.int64, shape=None, name='support_labels')
            self.query_points_placeholder = tf.placeholder(tf.float32, shape=(None, height, width, channels), name='query_points')
            self.query_labels_placeholder = tf.placeholder(tf.int64, shape=(None), name='query_labels')
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
        height, width, channels = self.image_dim
        with tf.variable_scope('block1'):
            conv1 = self.conv_bn_relu_pool(images, is_training, [3, 3, channels, 64], [64])
        with tf.variable_scope('block2'):
            conv2 = self.conv_bn_relu_pool(conv1, is_training, [3, 3, 64, 64], [64])
        with tf.variable_scope('block3'):
            conv3 = self.conv_bn_relu_pool(conv2, is_training, [3, 3, 64, 64], [64])
        with tf.variable_scope('block4'):
            conv4 = self.conv_bn_relu_pool(conv3, is_training, [3, 3, 64, 64], [64])
        return conv4

    def conv_bn_relu_pool(self, input, is_training, kernel_shape, bias_shape):
        weights = tf.get_variable("weights", kernel_shape, initializer=tf.random_normal_initializer())
        biases = tf.get_variable("biases", bias_shape, initializer=tf.constant_initializer(0.0))
        conv = tf.nn.conv2d(input, weights, strides=[1, 1, 1, 1], padding='SAME', name='conv_3x3')
        # TODO: double check that batchnorm is being reused for a given layer
        batch_norm = tf.layers.batch_normalization(conv + biases, training=is_training, name='batch_norm')
        relu = tf.nn.relu(batch_norm, name='relu')
        pool = tf.nn.max_pool(relu, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID', name='max_pool_2x2')
        return pool

    def add_model(self, support_points, support_labels, query_points, is_training):
        # compute embeddings of all examples
        # TODO: double check that weights are shared
        with tf.variable_scope('embedding') as scope:
            support_embedding = tf.contrib.layers.flatten(self.add_embedding(support_points, is_training))
            scope.reuse_variables()
            query_embedding = tf.contrib.layers.flatten(self.add_embedding(query_points, is_training))

        # create prototypes for each class
        with tf.name_scope('prototype'):
            ones = tf.ones_like(support_embedding)
            num_classes = tf.to_int32(tf.reduce_max(support_labels) + 1)
            per_class_embedding_sum = tf.unsorted_segment_sum(support_embedding, support_labels, num_classes, name='embedding_sum')
            class_counts = tf.unsorted_segment_sum(ones, support_labels, num_classes, name='class_counts')
            class_prototypes = per_class_embedding_sum / class_counts

        #dist = np.sqrt(np.sum(X**2, axis=1).reshape(-1, 1) - 2*X.dot(self.X_train.T) + np.sum(self.X_train**2, axis=1))
        # dists[i, j] is the Euclidean distance between the ith test point and the jth trainin
        # dists[i, j] distance between ith query_point and jth prototype
        # TODO: these are 60 for some reason WTF
        with tf.name_scope('distance'):
            query_square_sum = tf.reshape(tf.reduce_sum(tf.square(query_embedding), 1), shape=[-1, 1], name='query_square_sum')
            proto_square_sum = tf.reduce_sum(tf.square(class_prototypes), 1, name='proto_square_sum')
            distances = tf.add(query_square_sum, proto_square_sum, name='square_sum') - 2*tf.matmul(query_embedding, class_prototypes, transpose_b=True, name='cross_term')
        return distances

    def add_training_op(self, loss):
        # TODO: check this is the right optimizer
        with tf.name_scope('optimizer'):
            optimizer = tf.train.AdamOptimizer(self.lr)
            self.global_step = tf.train.get_or_create_global_step()
            train_op = optimizer.minimize(loss, self.global_step)
        return train_op

    def add_loss_op(self, distances, query_labels):
        with tf.name_scope('loss'):
            entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=query_labels, logits=distances*-1.0)
            loss = tf.reduce_mean(entropy, name='loss')
        with tf.name_scope('summaries'):
            tf.summary.scalar('loss', loss)
            tf.summary.histogram('histogram loss', loss)
            summary_op = tf.summary.merge_all()
        return loss, summary_op

    def predict(self, distances, query_labels=None):
        """Make predictions from the provided model.
        Args:
          sess: tf.Session()
          input_data: np.ndarray of shape (n_samples, n_features)
          input_labels: np.ndarray of shape (n_samples, n_classes)
        Returns:
          average_loss: Average loss of model.
          predictions: Predictions of model on input_data
        """
        with tf.name_scope('predictions'):
            logits = -1.0 * distances
            predictions = tf.argmax(tf.nn.softmax(logits), 1)
        if query_labels != None:
            with tf.name_scope('accuracy'):
                correct_preds = tf.equal(predictions, query_labels)
                accuracy_op = tf.reduce_mean(tf.cast(correct_preds, tf.float32))
        return predictions, accuracy_op

    def evaluate(self, sess, mode='test'):
        self.data_generator.mode = mode
        ave_accuracy = 0.0
        for i, ((sprt_batch, sprt_label_batch), (qry_batch, qry_label_batch)) in enumerate(self.data_generator):
            if i == 20:
                break
            feed_dict = self.create_feed_dict(sprt_batch, sprt_label_batch, qry_batch, qry_label_batch, False)
            accuracy, preds, dist = sess.run([self.accuracy_op, self.preds, self.distances], feed_dict=feed_dict)
            ave_accuracy += accuracy
        ave_accuracy = ave_accuracy / 20.0
        self.data_generator.mode = 'train'
        return ave_accuracy

    def fit(self, sess, saver):
        """Fit model on provided data.

        Args:
          sess: tf.Session()
          input_data: np.ndarray of shape (n_samples, n_features)
          input_labels: np.ndarray of shape (n_samples, n_classes)
        Returns:
          losses: list of loss per epoch
        """
        losses = []
        logdir = self.log_dir
        self.summary_writer = tf.summary.FileWriter(logdir, graph=tf.get_default_graph())
        for i, ((sprt_batch, sprt_label_batch), (qry_batch, qry_label_batch)) in enumerate(self.data_generator):
            # take gradient step
            feed_dict = self.create_feed_dict(sprt_batch, sprt_label_batch, qry_batch, qry_label_batch, True)
            _, loss, summary, step = sess.run([self.train_op, self.loss, self.loss_summary, self.global_step], feed_dict=feed_dict)
            self.summary_writer.add_summary(summary, step)
            losses.append(loss)
            # evaluate on test set and train set
            if (i + 1) % self.num_steps_per_checkpoint == 0 or i == 0:
                ave_test_accuracy = self.evaluate(sess, mode='test')
                ave_train_accuracy = self.evaluate(sess, mode='train')
                summary = tf.Summary()
                summary.value.add(tag='test accuracy', simple_value=ave_test_accuracy)
                summary.value.add(tag='train accuracy', simple_value=ave_train_accuracy)
                self.summary_writer.add_summary(summary, step)
            # save model
            if (i + 1) % self.num_steps_per_checkpoint == 0:
                saver.save(sess, self.checkpoint_dir, step)
            # every 2000 episodes cut the learning rate in half
            if (i + 1) % 2000 == 0:
                self.lr = self.lr * 0.5
        return losses

if __name__ == "__main__":
    args = parser.parse_args()
    net = PrototypicalNetwork(args)
    init = tf.global_variables_initializer()
    saver = tf.train.Saver()
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

    with tf.control_dependencies(update_ops):
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config)
        # if debug mode on
        #sess = tf_debug.LocalCLIDebugWrapperSession(sess)
        #sess.add_tensor_filter("has_inf_or_nan", tf_debug.has_inf_or_nan)

        checkpoint = tf.train.get_checkpoint_state(os.path.dirname(net.checkpoint_dir))
        if net.checkpoint != None:
            print 'restoring from checkpoint:', net.checkpoint
            saver.restore(sess, net.checkpoint)
        elif checkpoint and checkpoint.model_checkpoint_path and args.unpause:
            print 'restoring from checkpoint:', checkpoint.model_checkpoint_path
            net.checkpoint = checkpoint.model_checkpoint_path
            saver.restore(sess, checkpoint.model_checkpoint_path)
        else:
            print 'training from scratch'
            sess.run(init)
        losses = net.fit(sess, saver)
