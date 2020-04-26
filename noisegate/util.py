import itertools
import numpy as np
from astropy.io import fits
from typing import Optional, Callable, Generic
import collections


class HanningWindow:
    """
    A 3 dimensional apodization function based on the normal Hann window, the raised cosine
    """

    @classmethod
    def construct(cls, widths: list):
        """
        Helper function to actually create the window contents
        """
        # determine the window coordinates
        coords = [np.arange(dim) for dim in widths]
        coords = np.array(list(itertools.product(*coords)))
        x, y, t = coords[:, 0], coords[:, 1], coords[:, 2]

        # do the 3 dimensional calculation
        window = (np.power(np.sin((x + 0.5) * np.pi / widths[0]), 2.0) *
                  np.power(np.sin((y + 0.5) * np.pi / widths[1]), 2.0) *
                  np.power(np.sin((t + 0.5) * np.pi / widths[2]), 2.0))

        # create the window
        window = window.reshape(widths)
        return window


class LRUCache:
    """
    A cache that removes the least recently used elements when the capacity is exceeded.
    """
    def __init__(self, capacity: int):
        """
        Initialize the LRUCache
        :param capacity: maximum elements to keep
        """
        self.capacity = capacity
        self.cache = collections.OrderedDict()

    def __getitem__(self, key):
        """
        Retrieve an item with its key
        :param key: key to search by
        :return: the resulting object found
        """
        try:
            value = self.cache.pop(key)
            self.cache[key] = value
            return value
        except KeyError:
            return None

    def __setitem__(self, key, value):
        """
        Update/create an element in the LRU Cache with specified key and value
        :param key: index element
        :param value: data element
        """
        try:
            self.cache.pop(key)
        except KeyError:
            if len(self.cache) >= self.capacity:
                self.cache.popitem(last=False)
        self.cache[key] = value

    def __str__(self) -> str:
        """
        :return: simple string representation
        """
        return str(self.cache)

    def __len__(self) -> int:
        """
        :return: number of elements stored in cache currently
        """
        return len(self.cache)

    def __contains__(self, item) -> bool:
        """
        Check whether the LRUCache contains the item
        :param item: something to check for
        :return: whether item in cache
        """
        return item in self.cache


class ImageLoader:
    """
    A tool that manages opening image patches from a list of paths.
    It can be used to sample randomly from the patches.
    You pass a function that handles how to open your specific images into the initializer and then it handles all the
    sampling from there.
    """
    def __init__(self, paths: list, width: int, stride: int, capacity: Optional[int] = None,
                 load_img: Optional[Callable] = None) -> None:
        """
        
        :param paths: a list of file paths for all the images to use, assume that the order means the temporal order
        :param width: the side of the windows in each dimension
        :param stride: how many pixels between each patch
        :param capacity: the maximum number of images to keep open
        :param load_img: a function that takes a path and returns the image data
        """
        # keep the passed variables
        self.paths = paths
        self.width = width
        self.stride = stride
        self.capacity = capacity if capacity is not None else np.inf
        self.load_img = load_img
        if self.load_img is None:
            self.load_img = ImageLoader.default_loader

        # cache files once they're opened
        self.cache = LRUCache(self.capacity)

        # open an image to get the image size and properties
        self.img_shape = self.load_img(self.paths[0]).shape
        self.img_count = len(self.paths)
        self.patches_per_row = self.img_shape[0] // self.stride
        self.patches_per_image = (self.img_shape[1] // self.stride) * self.patches_per_row
        self.patches_count = (self.img_count // self.stride) * self.patches_per_image

    @classmethod
    def default_loader(cls, path: str) -> np.ndarray:
        """
        A default loader that opens using astropy
        :param path: the path to the file
        :return: the data from an image
        """
        with fits.open(path) as hdus:
            data = hdus[len(hdus) - 1].data
        return data.copy()

    def fetch(self, loader_index: np.ndarray) -> np.ndarray:
        """
        Fetch the patches related to a set of loader indices
        :param loader_index: the index in loader coordinates for the specific patch
        :return: the actual data of the patches:
            zeroth dimension: iterates over patches
            first dimension: iterates over first spatial (in original image)
            second dimension: iterates over second spatial (in original image)
            third dimension: the time dimension
        """
        # determine what images are needed
        img_indices, ii, jj = self.loader_index_to_patch_index(loader_index)
        img_indices = np.concatenate([img_indices + i for i in range(self.width)])
        img_indices_set = set(img_indices)

        # if too many images will be loaded warn the user
        if len(img_indices_set) > self.capacity:
            msg = "You have requested a fetch requiring {} images, more than the maximum limit of {}.".format(
                len(img_indices_set), self.capacity)
            raise RuntimeWarning(msg)

        # load the necessary images and cache them
        imgs = {index: self.load_img(self.paths[index]) for index in img_indices_set if index < len(self.paths)}
        for k, v in imgs.items():
            self.cache[k] = v
        # duplicate final image for last stack
        max_index = max(list(imgs.keys()))
        for i in range(self.width):
            imgs[max_index + i] = imgs[max_index]

        # construct all the patches
        patches = []
        for img_index, i, j in zip(img_indices, ii, jj):
            patch = np.stack([imgs[img_index+step][i:i+self.width, j:j+self.width]
                              for step in range(self.width)], axis=2)

            # if the patch goes off the image, pad it by wrapping
            if not np.all(patch.shape == np.repeat(self.width, 3)):
                pad_shape = list((int(0), int(num)) for num in np.repeat(self.width, 3) - patch.shape)
                patch = np.pad(patch, pad_shape, 'wrap')
            patches.append(patch)
        return np.stack(patches)

    def sample(self, n: int = 1) -> np.ndarray:
        """
        Sample from the patches with a uniform distribution
        :param n: the number of samples to draw
        :return: patches sampled randomly without replacement
        """
        try:
            indices = np.random.randint(0, len(self), size=n)
        except ValueError:
            raise ValueError("n must be between 1 and {}, but n={}".format(len(self), n))
        else:
            return self.fetch(indices)

    def loader_index_to_patch_index(self, indices: np.ndarray) -> [np.ndarray, np.ndarray, np.ndarray]:
        """
        Convert an ImageLoader index to the patch coordinates
            image number, and the (i,j) of the upper left corner for that patch

        The ImageLoader manages opening patches. In it, two types of indices are used:
        - patch indices: this is a set of integers (image_index, i, j)
            where image_index is the index of the image path in paths and (i,j) are the pixel coordinates of the upper
            left of the patch in that image (with the image being the specified widths)
        - loader indices: this is a single integer: it is computed as follows:
            loader_index = image_index * patches_per_image + i * patches_per_row + j
        :param indices:
        :return: the patch indices (img_number, ii, jj)
        """
        img_number = (indices // self.patches_per_image) * self.stride
        ii = ((indices % self.patches_per_image) % self.patches_per_row) * self.stride
        jj = ((indices % self.patches_per_image) // self.patches_per_row) * self.stride
        return img_number, ii, jj

    def __len__(self) -> int:
        return self.patches_count
