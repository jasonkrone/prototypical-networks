import numpy as np
import os

from DataGenerator import DataGenerator


class MiniImagenetGenerator(DataGenerator):

    def __init__(self, data_dir, max_iter, num_support_points_per_class, num_query_points_per_class, \
                 num_train_classes, num_test_classes, num_val_classes=None, mode='train'):
        # call parent constructor
        super(MiniImagenetGenerator, self).__init__(data_dir, max_iter, (84, 84, 3), num_support_points_per_class, num_query_points_per_class, \
                                                    num_train_classes, num_test_classes, num_val_classes, mode)

        self.classes_2_files = self.classes_2_files_dict()
        #print('classes to files:', self.classes_2_files, 'keys:', len(self.classes_2_files.keys()), 'values:', [len(a) for a in self.classes_2_files.values()])
        classes = np.array(self.classes_2_files.keys())
        self.train_classes, self.val_classes, self.test_classes  = self.split_classes(classes, [64, 16, 20])
        print self.train_classes

    def data_for_class(self, cls):
        num_points = self.num_support_points_per_class + self.num_query_points_per_class
        class_files = np.array(self.classes_2_files[cls])
        idxs = np.random.choice(len(class_files), size=num_points, replace=False)

        files = class_files[idxs]
        support_files = files[:self.num_support_points_per_class]
        query_files   = files[self.num_support_points_per_class:num_points]
        # check examples don't overlap
        assert len(set(support_files).intersection(set(query_files))) == 0
        support_points = self.image_data_for_files(support_files, 0.0, (84, 84, 3))
        query_points   = self.image_data_for_files(query_files, 0.0, (84, 84, 3))
        return support_points, query_points

    def classes_2_files_dict(self, file_path='/home/jason/datasets/mini_imagenet.txt'):
        lines = [l for l in open(file_path).readlines() if l != '\n']
        classes = np.array([l.split('/')[-2] for l in lines if l.startswith('data')])
        cls = None
        classes_2_files = {c : [] for c in classes}
        for l in lines:
            l = l.strip()
            if l.startswith('data'):
                cls = l.split('/')[-2]
            else:
                path = self.data_dir + cls + '/' + l
                assert os.path.exists(path), path
                classes_2_files[cls].append(path)
        return classes_2_files

if __name__ == '__main__':
    generator = MiniImagenetGenerator('/dvmm-filer2/datasets/ImageNet/train/', 10, 60, 5, 5, 5)
    (support_points, support_labels), (query_points, query_labels) = generator.next()
    print(support_points.shape, support_labels.shape, query_points.shape, query_labels.shape)

