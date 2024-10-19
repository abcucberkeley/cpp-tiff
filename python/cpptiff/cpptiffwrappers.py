import numpy as np
from .cpptiff import pybind11_read_tiff, pybind11_write_tiff

def read_tiff(file_name):
    im = pybind11_read_tiff(file_name)
    im = np.transpose(im, (2, 0, 1))
    return im
def write_tiff(file_name, data, flip_xy=False):
    data = np.transpose(data, (1, 2, 0))
    pybind11_write_tiff(file_name, data, flip_xy)
    return