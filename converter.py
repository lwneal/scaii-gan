"""
A Converter converts between:
    examples (each one a dict with keys like "filename" and "label")
    arrays (numpy arrays input to or output from a network)

Dataset augmentation can be accomplished with a Converter that returns a
different array each time to_array is called with the same example
"""
import os
import numpy as np
import random
from gnomehat import imutil

# TODO: Configure this
DATA_DIR = '/mnt/nfs/data'

# Converters can be used like a function, on a single example or a batch
class Converter(object):
    def __call__(self, inputs):
        if isinstance(inputs, np.ndarray):
            return [self.from_array(e) for e in inputs]
        elif isinstance(inputs, list):
            return np.array([self.to_array(e) for e in inputs])
        else:
            return self.to_array(inputs)


# Crops, resizes, normalizes, performs any desired augmentations
# Outputs images as eg. 32x32x3 np.array or eg. 3x32x32 torch.FloatTensor
class ImageConverter(Converter):
    def __init__(self,
            dataset,
            image_size=32,
            crop_to_bounding_box=True,
            random_horizontal_flip=False,
            delete_background=False,
            torch=True,
            normalize=True,
            **kwargs):
        width, height = image_size, image_size
        self.img_shape = (width, height)
        self.bounding_box = crop_to_bounding_box
        self.data_dir = dataset.data_dir
        self.random_horizontal_flip = random_horizontal_flip
        self.torch = torch
        self.normalize = normalize
        self.delete_background = delete_background

    def to_array(self, example):
        filename = os.path.expanduser(example['filename'])
        if not filename.startswith('/'):
            filename = os.path.join(DATA_DIR, filename)
        box = example.get('box') if self.bounding_box else None
        img = imutil.decode_jpg(filename,
                resize_to=self.img_shape,
                crop_to_box=box)
        if self.delete_background:
            seg_filename = os.path.expanduser(example['segmentation'])
            segmentation = imutil.decode_jpg(seg_filename,
                    resize_to=self.img_shape,
                    crop_to_box=box)
            foreground_mask = np.mean(segmentation, axis=-1) / 255.
            img = img * np.expand_dims(foreground_mask, axis=-1)
        if self.random_horizontal_flip and random.getrandbits(1):
            img = np.flip(img, axis=1)
        if self.torch:
            img = img.transpose((2,0,1))
        if self.normalize:
            img *= 1.0 / 255
        return img

    def from_array(self, array):
        return array


class SkyRTSConverter(Converter):
    def __init__(self,
            dataset,
            **kwargs):
        self.data_dir = dataset.data_dir

    def to_array(self, example):
        filename = os.path.expanduser(example['filename'])
        if not filename.startswith('/'):
            filename = os.path.join(DATA_DIR, filename)
        # Input is a PNG composed of 6 40x40 monochrome images
        # It encodes frames of a game, similar to the SC2 API
        # From top-left to bottom-right, maps represent:
        # Health, Agent, Small Towers, Big Towers, Friends, Enemies
        img = imutil.decode_jpg(filename, resize_to=None)
        assert img.shape == (40*3, 40*2, 3)
        # Pytorch convnets require BCHW inputs
        channels = np.zeros((6, 40, 40))
        channels[0] = img[0:40, 0:40, 0]
        channels[1] = img[0:40, 40:80, 0]
        channels[2] = img[40:80, 0:40, 0]
        channels[3] = img[40:80, 40:80, 0]
        channels[4] = img[80:120, 0:40, 0]
        channels[5] = img[80:120, 40:80, 0]
        # Normalize to [0, 1]
        return channels / 255.0

    def from_array(self, array):
        return array


# LabelConverter extracts the class labels from DatasetFile examples
# Each example can have only one class
class LabelConverter(Converter):
    def __init__(self, dataset, label_key="label", **kwargs):
        self.label_key = label_key
        self.labels = get_labels(dataset, label_key)
        self.num_classes = len(self.labels)
        self.idx = {self.labels[i]: i for i in range(self.num_classes)}
        print("LabelConverter: labels are {}".format(self.labels))

    def to_array(self, example):
        return self.idx[example[self.label_key]]

    def from_array(self, array):
        return self.labels[np.argmax(array)]


# FlexibleLabelConverter extracts class labels including partial and negative labels
# Each example now has a label for each class:
#    1 (X belongs to class Y)
#   -1 (X does not belong to class Y)
#   0  (X might or might not belong to Y)
class FlexibleLabelConverter(Converter):
    def __init__(self, dataset, label_key="label", negative_key="label_n", **kwargs):
        self.label_key = label_key
        self.negative_key = negative_key
        self.labels = sorted(list(set(get_labels(dataset, label_key) + get_labels(dataset, negative_key))))
        self.num_classes = len(self.labels)
        self.idx = {self.labels[i]: i for i in range(self.num_classes)}
        print("FlexibleLabelConverter: labels are {}".format(self.labels))

    def to_array(self, example):
        array = np.zeros(self.num_classes)
        if self.label_key in example:
            array[:] = -1  # Negative labels
            idx = self.idx[example[self.label_key]]
            array[idx] = 1  # Positive label
        if self.negative_key in example:
            idx = self.idx[example[self.negative_key]]
            array[idx] = -1
        return array

    def from_array(self, array):
        return self.labels[np.argmax(array)]


# QValueConverter extracts action-value pairs from Dataset files
# A network performs regression to a Q-value for each possible action
# The label consists of a ground truth value for one action (set to 1 in the mask)
# Other actions (set to 0 in the mask) should be ignored in the loss
class QValueConverter(Converter):
    def __init__(self, dataset, action_key='action', value_key='value', **kwargs):
        self.action_key = action_key
        self.value_key = value_key
        self.actions = sorted(list(set(get_labels(dataset, action_key))))
        self.num_classes = len(self.actions)
        print("QValueConverter: actions are {}".format(self.actions))
        values = set(get_labels(dataset, value_key))
        self.min_val = min(values)
        self.max_val = max(values)
        print('Q value range: from {} to {}'.format(self.min_val, self.max_val))

    def to_array(self, example):
        qvals = np.zeros(self.num_classes)
        mask = np.zeros(self.num_classes)
        qvals[example[self.action_key] - 1] = example[self.value_key]
        mask[example[self.action_key] - 1] = 1
        return qvals, mask


def get_labels(dataset, label_key):
    unique_labels = set()
    for example in dataset.examples:
        if label_key in example:
            unique_labels.add(example[label_key])
    return sorted(list(unique_labels))


# AttributeConverter extracts boolean attributes from DatasetFile examples
# An example might have many attributes. Each attribute is True or False.
class AttributeConverter(Converter):
    def __init__(self, dataset, **kwargs):
        unique_attributes = set()
        for example in dataset.examples:
            for key in example:
                if key.startswith('is_') or key.startswith('has_'):
                    unique_attributes.add(key)
        self.attributes = sorted(list(unique_attributes))
        self.num_attributes = len(self.attributes)
        self.idx = {self.attributes[i]: i for i in range(self.num_attributes)}

    def to_array(self, example):
        attrs = np.zeros(self.num_attributes)
        for i, attr in enumerate(self.attributes):
            # Attributes not present on an example are set to False
            attrs[i] = float(example.get(attr, False))
        return attrs

    def from_array(self, array):
        return ",".join(self.attributes[i] for i in range(self.attributes) if array[i > .5])

