import numpy as np
import os

from scipy.ndimage import rotate,shift
from scipy.misc import imread,imresize


class OmniglotGenerator(object):

    def __init__(self, data_dir, max_iter, num_train_classes, num_test_classes, num_support_points_per_class, num_query_points_per_class, mode='train'):
        self.num_train_classes = num_train_classes
        self.num_test_classes = num_test_classes
        self.num_support_points_per_class = num_support_points_per_class
        self.num_query_points_per_class = num_query_points_per_class
        self.mode = mode
        self.data_dir = data_dir
        self.max_iter = max_iter
        self.cur_iter = 0
        self.rotations = [0.0, 90.0, 180.0, 270.0]
        class_dirs = np.array([data_dir + '/' + cls for cls in os.listdir(data_dir)])
        train_class_dirs, test_class_dirs = self.split_train_test(class_dirs)
        print('train class dirs:', train_class_dirs.shape)
        print('test class dirs:', test_class_dirs.shape)
        self.train_classes = np.array([(r, dir) for dir in train_class_dirs for r in self.rotations])
        self.test_classes  = np.array([(r, dir) for dir in test_class_dirs for r in self.rotations])
        print('train_classes:', self.train_classes.shape)
        print('test classes:', self.test_classes.shape)
        self.class_instances = ['01.png', '02.png', '03.png', '04.png', '05.png',
                                '06.png', '07.png', '08.png', '09.png', '10.png',
                                '11.png', '12.png', '13.png', '14.png', '15.png',
                                '16.png', '17.png', '18.png', '19.png', '20.png']

    def split_train_test(self, class_dirs):
        # select 1200 characters (i.e. classes) for training set
        num_classes = len(class_dirs)
        train_idxs = np.random.choice(num_classes, size=1200, replace=False)
        train_classes = class_dirs[train_idxs]
        # select remaining characters for test set
        mask = np.ones(num_classes, np.bool)
        mask[train_idxs] = 0
        test_classes = class_dirs[mask]
        return train_classes, test_classes

    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    def next(self):
        if (self.max_iter is None) or (self.cur_iter < self.max_iter):
            if self.mode == 'train':
                self.cur_iter += 1
                return self.sample_episode(self.train_classes, self.num_train_classes)
            elif self.mode == 'test':
                return self.sample_episode(self.test_classes, self.num_test_classes)
        else:
            raise StopIteration

    def sample_episode(self, classes, num_classes):
        # randomly sample num_classes from the train or test class list
        ep_idxs = np.random.choice(len(classes), size=num_classes, replace=False)
        ep_classes  = classes[ep_idxs]
        # shuffle classes, this may be unecessary
        perm = np.random.permutation(num_classes)
        ep_classes = ep_classes[perm]
        num_support_points = num_classes * self.num_support_points_per_class
        num_query_points = num_classes * self.num_query_points_per_class
        support_points = np.zeros((num_support_points, 28, 28, 1))
        support_labels = np.zeros((num_support_points))
        # keep track of mapping between str labels and int labels
        support_label_names = []
        query_points = np.zeros((num_query_points, 28, 28, 1))
        query_labels = np.zeros((num_query_points))
        query_label_names = []

        for k, (r, dir) in enumerate(ep_classes):
            num_points = self.num_support_points_per_class+self.num_query_points_per_class
            # randomly sample support points and query points for each class
            points = np.random.choice(self.class_instances, size=num_points, replace=False)
            support_points_k = points[:self.num_support_points_per_class]
            query_points_k = points[self.num_support_points_per_class:num_points]
            # get paths to selected examples for class k
            support_files_k = [dir + '/' + sp for sp in support_points_k]
            query_files_k = [dir + '/' + qp for qp in query_points_k]
            num_sp, num_qp = self.num_support_points_per_class, self.num_query_points_per_class
            # load image data from disk and set labels
            support_points[k*num_sp:(k+1)*num_sp, :, :, :] = self.image_data_for_files(support_files_k, float(r))
            support_labels[k*num_sp:(k+1)*num_sp] = k
            support_label_names += [(r, sf) for sf in support_files_k]
            query_points[k*num_qp:(k+1)*num_qp, :, :, :] = self.image_data_for_files(query_files_k, float(r))
            query_labels[k*num_qp:(k+1)*num_qp] = k
            query_label_names += [(r, qf) for qf in query_files_k]

        support_label_names = np.array(support_label_names)
        query_label_names = np.array(query_label_names)
        # shuffle support and query set so classes don't appear in order
        support_perm = np.random.permutation(num_support_points)
        support_points = support_points[support_perm]
        support_labels = support_labels[support_perm]
        support_label_names = support_label_names[support_perm]
        query_perm = np.random.permutation(num_query_points)
        query_points = query_points[query_perm]
        query_labels = query_labels[query_perm]
        query_label_names = query_label_names[query_perm]
        return (support_labels, support_points, support_label_names), (query_labels, query_points, query_label_names)

    def image_data_for_files(self, file_paths, degree_rotation):
        num_files = len(file_paths)
        images = np.zeros((num_files, 28, 28, 1))
        for i, path in enumerate(file_paths):
            original = imread(path)
            resized = imresize(original, (28, 28))
            rotated = rotate(resized, angle=degree_rotation)
            images[i, :, :, 0] = rotated / np.max(rotated) # trying normalization
        return images
