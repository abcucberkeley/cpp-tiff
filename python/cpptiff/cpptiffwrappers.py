import numpy as np
import os
from .cpptiff import pybind11_read_tiff, pybind11_write_tiff, pybind11_get_image_shape


def read_tiff(file_name, z_range=None):
    if not os.path.isfile(file_name):
        raise Exception(f'{file_name} does not exist')
    if z_range is None:
        z_range = []
    else:
        if type(z_range) is not list:
            if np.isscalar(z_range):
                z_range = [z_range]
            else:
                raise Exception(f'z_range must be a number or list of two numbers')

    im = pybind11_read_tiff(file_name, z_range)
    im = np.transpose(im, (2, 1, 0))
    if im.shape[0] == 1:
        im = np.squeeze(im, axis=0)
    return im


def write_tiff(file_name, data, compression='lzw'):
    data = np.transpose(data, (2, 1, 0))
    pybind11_write_tiff(file_name, data, compression)
    return


def get_image_shape(file_name):
    return pybind11_get_image_shape(file_name)
