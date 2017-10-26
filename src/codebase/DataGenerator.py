import numpy as np
import os

from scipy.ndimage import rotate,shift
from scipy.misc import imread,imresize

class DataGenerator(object):
    
    def __init__(self, data_dir, max_iter, image_dim, num_support_points_per_class, num_query_points_per_class, \
                 num_train_classes, num_test_classes, num_val_classes=None, mode='train'):
        self.num_train_classes = num_train_classes
        self.num_test_classes  = num_test_classes
        self.num_val_classes   = num_val_classes
        self.num_support_points_per_class = num_support_points_per_class
        self.num_query_points_per_class = num_query_points_per_class
        self.mode = mode
        self.image_dim = image_dim
        self.data_dir = data_dir
        self.max_iter = max_iter
        self.cur_iter = 0

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
            elif self.mode == 'val':
                return self.sample_episode(self.val_classes, self.num_val_classes)
        else:
            raise StopIteration

    def split_classes(self, class_dirs, split_sizes):
        splits = [] 
        total  = sum(split_sizes)
        # generates total random classes from class_dirs
        idxs   = np.random.choice(len(class_dirs), size=total, replace=False)
        s_prev = 0
        for s in split_sizes:
            new_split = idxs[s_prev:s_prev+s]
            splits.append(new_split)
            s_prev += s
        # check there are no duplicates
        splits_flat = [x for s in splits for x in s] 
        assert sum([len(s) for s in splits]) == len(set(splits_flat))
        return splits
 
    def sample_episode(self, classes, num_classes):
        # randomly sample num_classes from the train or test class list
        ep_idxs = np.random.choice(len(classes), size=num_classes, replace=False)
        ep_classes  = classes[ep_idxs]
        # shuffle classes, this may be unecessary
        perm = np.random.permutation(num_classes)
        ep_classes = ep_classes[perm]
        num_support_points = num_classes * self.num_support_points_per_class
        num_query_points = num_classes * self.num_query_points_per_class

        h, w, c = self.image_dim
        support_points = np.zeros((num_support_points, h, w, c))
        support_labels = np.zeros((num_support_points))
        query_points = np.zeros((num_query_points, h, w, c))
        query_labels = np.zeros((num_query_points))

        num_sp, num_qp = self.num_support_points_per_class, self.num_query_points_per_class
        for k, c in enumerate(ep_classes):
            support_points_k, query_points_k = data_for_class(c)
            support_points[k*num_sp:(k+1)*num_sp, :, :, :] = support_points_k
            query_points[k*num_qp:(k+1)*num_qp, :, :, :] = query_points_k 
            # assign integer labels
            support_labels[k*num_sp:(k+1)*num_sp] = k 
            query_labels[k*num_qp:(k+1)*num_qp] = k 

        # shuffle support and query set so classes don't appear in order
        support_perm = np.random.permutation(num_support_points)
        support_points = support_points[support_perm]
        support_labels = support_labels[support_perm]
        query_perm = np.random.permutation(num_query_points)
        query_points = query_points[query_perm]
        query_labels = query_labels[query_perm]
        return (support_points, support_labels), (query_points, query_labels)

    def image_data_for_files(self, file_paths, degree_rotation=0.0, new_size=(28, 28)):
        num_files = len(file_paths)
        images = np.zeros((num_files, 28, 28, 1))
        for i, path in enumerate(file_paths):
            original = imread(path)
            resized  = imresize(original, new_size)
            rotated  = rotate(resized, angle=degree_rotation)
            # TODO: : might break this
            images[i, :, :, :] = rotated #/ np.max(rotated) # trying normalization
        return images

    def data_for_class(cls):
        raise NotImplementedError("Each generator must re-implement this method.")

