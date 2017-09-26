import tensorflow as tf
import argparse
import copy

from model import Model


parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str, default='../dat')
parser.add_argument('--checkpoint_dir', type=str, default='../checkpoints')
parser.add_argument('--output_dir', type=str, default='../out')
parser.add_argument('--log_dir', type=str, default='../log')
# cu birds episodes: 50 classes, 10 query images / class
#parser.add_argument('--batch_size', type=int, default=10, help='Minibatch size during training')
parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate step-size')
parser.add_argument('--num_classes_per_ep', type=int, default=1, help='Number of classes per episode')
parser.add_argument('--num_support_points', type=int, default=1, help='Number of support points per class')
parser.add_argument('--num_query_points', type=int, default=10, help='Number of query points during training')


class PrototypicalNetwork(Model):

    def __init__(self, config):
        self.lr = config.lr
        self.num_classes_per_ep = config.num_classes_per_ep
        self.num_support_points = config.num_support_points
        self.num_query_points   = config.num_query_points
        self.config = copy.deepcopy(config)
        self.graph = self.build_graph(tf.Graph())

    def add_place_holders(self):
        image_dim = 1024
        meta_data_dim = 312
        num_query_points = self.num_query_points
        episode_size = self.num_classes_per_ep * self.num_support_points

        self.support_points_placeholder = tf.placeholder(tf.float32, shape=(episode_size, meta_data_dim), name='support_points')
        self.support_labels_placeholder = tf.placeholder(tf.int32, shape=(episode_size), name='support_labels')
        self.query_points_placeholder = tf.placeholder(tf.float32, shape=(num_query_points, image_dim), name='query_points')
        self.query_labels_placeholder = tf.placeholder(tf.int32, shape=(num_query_points), name='query_labels')

    def create_feed_dict(self, support_points, support_labels, query_points, query_labels):
        feed_dict = {
            self.support_points_placeholder : support_points,
            self.support_labels_placeholder : support_labels,
            self.query_points_placeholder   : query_points,
            self.query_labels_placeholder   : query_labels
        }
        return feed_dict

    def add_model(self, support_points, query_points):
        # TODO: fix meta-data embedding to have unit length
        with tf.variable_scope('support_embedding') as scope:
            support_emedding = tf.get_variable('support_embedding', shape=[1024, 1024], initializer=tf.contrib.layers.xavier_initializer())
        with tf.variable_scope('query_embedding') as scope:
            query_emedding = tf.get_variable('query_embedding', shape=[312, 1024], initializer=tf.contrib.layers.xavier_initializer())
        return support_embedding, query_embedding

    def add_loss_op(self, pred):
        pass

if __name__ == "__main__":
    config = parser.parse_args()
    model = PrototypicalNetwork(config)


