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
parser.add_argument('--classes_per_ep', type=int, default=1, help='Number of classes per episode')
parser.add_argument('--support_points', type=int, default=1, help='Number of support points per class')
parser.add_argument('--query_points', type=int, default=10, help='Number of query points during training')


class PrototypicalNetwork(Model):

    def __init__(self, config):
        self.lr = config.lr
        self.classes_per_ep = config.classes_per_ep
        self.support_points = config.support_points
        self.query_points   = config.query_points
        self.config = copy.deepcopy(config)
        self.graph = self.build_graph(tf.Graph())

    def add_place_holders(self):
        image_dim = 1024
        meta_data_dim = 312
        query_points = self.query_points
        episode_size = self.classes_per_ep * self.support_points

        self.support_points_placeholder = tf.placeholder(tf.float32, shape=(episode_size, meta_data_dim))
        self.support_labels_placeholder = tf.placeholder(tf.int32, shape=(episode_size))
        self.query_points_placeholder = tf.placeholder(tf.float32, shape=(query_points, image_dim))
        self.query_labels_placeholder = tf.placeholder(tf.int32, shape=(query_points))

    
if __name__ == "__main__":
    config = parser.parse_args()
    model = PrototypicalNetwork(config)


