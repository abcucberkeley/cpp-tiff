import numpy as np
import os
from .cpptiff import pybind11_read_tiff, pybind11_write_tiff, pybind11_get_image_shape

def read_tiff(file_name):
	if not os.path.isfile(file_name):
		raise Exception(f'{file_name} does not exist')
	im = pybind11_read_tiff(file_name)
	im = np.transpose(im, (2, 1, 0))
	return im
def write_tiff(file_name, data, compression='lzw'):
	data = np.transpose(data, (2, 1, 0))
	pybind11_write_tiff(file_name, data, compression)
	return
def get_image_shape(file_name):
	return pybind11_get_image_shape(file_name)
