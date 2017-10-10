import numpy as np
import os


class OmniglotGenerator(object):

    def __init__(self, data_dir, num_classes_per_ep, num_support_points_per_class, num_query_points_per_class):
        self.num_classes_per_ep = config.num_classes_per_ep
        self.num_support_points_per_class = config.num_support_points_per_class
        self.num_query_points_per_class = config.num_query_points_per_class
        
        class_dirs = np.array([data_dir + '/' + cls for cls in os.listdir(self.data_dir)])
        self.train_class_dirs, self.test_class_dirs = split_train_test(class_dirs)

    def split_train_test(self, class_dirs):
        # select 1200 characters (i.e. classes) for training set
        num_classes = len(class_dirs)
        train_idxs = np.random.choice(num_classes, size=1200, replace=False)
        train_class_dirs = class_dirs[train_idxs]
        # select remaining characters for test set
        mask = np.ones(num_classes, np.bool)
        mask[train_idxs] = 0
        test_class_dirs = class_dirs[mask]
        return train_class_dirs, test_class_dirs

    def 



    def data_for_classes(self, classes):
        filenames = [self.data_dir+'/'+str(c)+'/'+f for c in classes for f in os.listdir(self.data_dir+'/'+str(c))]
        filenames = tf.convert_to_tensor(filenames, dtype=tf.string)
        labels = tf.convert_to_tensor([int(c) for c classes], dtype=tf.int32)
        filename_queue = tf.train.string_input_producer(filenames)
        reader = tf.WholeFileReader()
        key, value = reader.read(filename_queue)
        images = tf.image.decode_png(value)
        # may have to do this images.set_shape([])
        labels, images

