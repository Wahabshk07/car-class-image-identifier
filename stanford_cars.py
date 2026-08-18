from scipy.io import loadmat
import numpy as np
from torch.utils.data import Dataset
import torchvision
from enum import Enum
import os
from PIL import Image
import math
import logging
logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)


class CarDatasetAttributes(Enum):
    LABEL = 'class'
    REL_IMAGE_PATH = 'relative_im_path'
    FILE_NAME = 'fname'


def annotation_image_name(data_point):
    """Return the image filename from either official (fname) or combined (relative_im_path) annotations."""
    field_names = data_point.dtype.names
    if CarDatasetAttributes.FILE_NAME.value in field_names:
        return data_point[CarDatasetAttributes.FILE_NAME.value][0][0]
    return data_point[CarDatasetAttributes.REL_IMAGE_PATH.value][0][0]


def annotation_class_id(data_point):
    """Return 0-indexed class id from a 1-indexed MATLAB annotation row."""
    return data_point[CarDatasetAttributes.LABEL.value][0][0][0] - 1


def is_official_stanford_cars_root(path_images):
    return os.path.isdir(os.path.join(path_images, 'cars_train')) and os.path.isdir(
        os.path.join(path_images, 'cars_test'))


def resolve_official_dataset_paths(path_images, path_data, path_labels):
    train_images = os.path.join(path_images, 'cars_train')
    test_images = os.path.join(path_images, 'cars_test')
    train_annos = path_data or os.path.join(path_images, 'devkit', 'cars_train_annos.mat')
    test_annos = os.path.join(path_images, 'cars_test_annos_withlabels.mat')
    if not os.path.isfile(test_annos):
        test_annos = os.path.join(path_images, 'devkit', 'cars_test_annos_withlabels.mat')
    labels = path_labels or os.path.join(path_images, 'devkit', 'cars_meta.mat')
    return train_images, test_images, train_annos, test_annos, labels


def load_annotations(path_to_matdata, data_subset):
    """Load a single split's annotations and optionally keep a per-class subset."""
    logging.info("loading annotations from %s", path_to_matdata)
    annotations = loadmat(path_to_matdata)['annotations']
    unique_labels = np.unique(annotations['class'])

    if data_subset >= 1.0:
        selected = np.reshape(annotations, (annotations.size, 1))
        np.random.shuffle(selected)
        return selected, unique_labels

    selected = np.array([])
    for label in unique_labels:
        class_label = label[0][0]
        class_struct = annotations[annotations['class'] == class_label]
        class_struct = np.reshape(class_struct, (class_struct.shape[0], 1))
        np.random.shuffle(class_struct)
        class_struct = class_struct[:math.ceil(class_struct.shape[0] * data_subset)]
        if selected.shape[0] == 0:
            selected = class_struct
        else:
            selected = np.append(selected, class_struct)

    selected = np.reshape(selected, (selected.shape[0], 1))
    np.random.shuffle(selected)
    return selected, unique_labels


class StanfordCars(Dataset):

    TRANSFORMED_IMAGE = 'train'
    LABEL = 'label'

    def __init__(self, data_matrix, path_images, transforms, path_human_readable_labels):
        self.transforms = transforms
        self.data_matrix = data_matrix
        self.path_images = path_images
        self.path_human_readable_labels = path_human_readable_labels
        self.human_readable_labels = None
        self.load_human_readable_labels()

    def load_human_readable_labels(self):
        self.human_readable_labels = loadmat(self.path_human_readable_labels)['class_names']

    def get_label_unique_count(self):
        return np.unique(self.data_matrix['class'], return_counts=True)

    def get_class_distribution(self):
        return self.get_label_unique_count()[1]/len(self.data_matrix)

    def __len__(self):
        return self.data_matrix.size

    def __getitem__(self, item):
        data_point = self.data_matrix[item]
        image_path = os.path.join(self.path_images, annotation_image_name(data_point))
        # shifting labels from 1-index to 0-index
        label = annotation_class_id(data_point)
        logging.debug("image: %s is a %s with label %s", image_path, self.human_readable_labels[0, label], label)
        image = Image.open(image_path)
        if len(np.array(image).shape) < 3:
            image = image.convert("RGB")
        composed_transforms = torchvision.transforms.Compose(self.transforms)
        return {self.TRANSFORMED_IMAGE: composed_transforms(image), self.LABEL: label}


class StanfordCarsTestData(Dataset):

    TRANSFORMED_IMAGE = 'train'
    LABEL = 'label'
    IMAGE_PATH = 'image_path'

    def __init__(self, data_matrix, path_images, transforms, path_human_readable_labels):
        self.transforms = transforms
        self.data_matrix = data_matrix
        self.path_images = path_images
        self.path_human_readable_labels = path_human_readable_labels
        self.human_readable_labels = None
        self.load_human_readable_labels()

    def load_human_readable_labels(self):
        self.human_readable_labels = loadmat(self.path_human_readable_labels)['class_names']

    def get_label_unique_count(self):
        return np.unique(self.data_matrix['class'], return_counts=True)

    def get_class_distribution(self):
        return self.get_label_unique_count()[1]/len(self.data_matrix)

    def __len__(self):
        return self.data_matrix.size

    def __getitem__(self, item):
        data_point = self.data_matrix[item]
        image_path = os.path.join(self.path_images, annotation_image_name(data_point))
        # shifting labels from 1-index to 0-index
        label = annotation_class_id(data_point)
        logging.debug("image: %s is a %s with label %s", image_path, self.human_readable_labels[0, label], label)
        image = Image.open(image_path)
        image.show()
        if len(np.array(image).shape) < 3:
            image = image.convert("RGB")
        composed_transforms = torchvision.transforms.Compose(self.transforms)
        return {self.TRANSFORMED_IMAGE: composed_transforms(image), self.LABEL: label, self.IMAGE_PATH: image_path}


def preprocess_data(path_to_matdata, validation_percentage, data_subset):

    logging.info("preprocessing data")
    data_struct = loadmat(path_to_matdata)
    annotations = data_struct['annotations']
    annotations_labels = annotations['class']

    validation_struct = np.array([])
    training_struct = np.array([])

    unique_labels = np.unique(annotations_labels)

    for label in unique_labels:
        class_label = label[0][0]
        class_struct = annotations[annotations['class'] == class_label]
        class_struct = np.reshape(class_struct, (class_struct.shape[0], 1))
        np.random.shuffle(class_struct)

        #subset data
        class_struct = class_struct[:math.ceil(class_struct.shape[0] * data_subset)]

        # split data into training and validation
        class_struct_shape = class_struct.shape
        validation_split = math.floor(class_struct_shape[0] * validation_percentage)
        validation_data_points = class_struct[:validation_split]
        training_data_points = class_struct[validation_split:]

        if validation_struct.shape[0] == 0:
            validation_struct = validation_data_points
        else:
            validation_struct = np.append(validation_struct, validation_data_points)

        if training_struct.shape[0] == 0:
            training_struct = training_data_points
        else:
            training_struct = np.append(training_struct, training_data_points)

    # shuffle training and validation data
    validation_struct = np.reshape(validation_struct, (validation_struct.shape[0],1))
    np.random.shuffle(validation_struct)

    training_struct = np.reshape(training_struct, (training_struct.shape[0],1))
    np.random.shuffle(training_struct)

    return training_struct, validation_struct, unique_labels
