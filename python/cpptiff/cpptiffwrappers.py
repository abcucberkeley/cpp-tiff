import numpy as np
import os
from .cpptiff import pybind11_read_tiff, pybind11_write_tiff, pybind11_get_image_shape


def read_tiff(file_name, z_range=None):
    if not os.path.isfile(file_name):
        raise Exception(f'{file_name} does not exist!')
    if z_range is None:
        z_range = []
    else:
        if type(z_range) is not list:
            if np.isscalar(z_range):
                z_range = [z_range]
            else:
                raise Exception(f'z_range must be a number or list of two numbers!')
        elif len(z_range) == 2:
            if z_range[0] > z_range[1]:
                raise Exception(f'z_range is invalid! {z_range[0]} > {z_range[1]}')
            elif z_range[0] == z_range[1]:
                raise Exception(f'z_range is invalid! {z_range[0]} == {z_range[1]}')

    im = pybind11_read_tiff(file_name, z_range)
    if im.ndim == 4:
        im = np.transpose(im, (2, 1, 0, 3))  # (x,y,z,c) -> (z,y,x,c) for RGB
    else:
        im = np.transpose(im, (2, 1, 0))     # (x,y,z) -> (z,y,x)
    if im.shape[0] == 1:
        im = np.squeeze(im, axis=0)
    return im


def write_tiff(file_name, data, compression='lzw', tile=0, rgb=None):
    """Write data as a TIFF file, one page per slice.

    Grayscale data is (z, y, x) or (y, x). RGB and RGBA data has its 3 or 4
    color samples last: (z, y, x, c), or (y, x, c) for one image. rgb=None
    treats 4D data as color and anything else as grayscale, so a single
    (y, x, c) color image needs rgb=True.
    """
    ndims = data.ndim
    if rgb is None:
        rgb = ndims == 4
    if rgb:
        if ndims not in (3, 4):
            raise Exception('RGB data must be (y, x, c) or (z, y, x, c)!')
        if data.shape[-1] not in (3, 4):
            raise Exception(f'RGB data must have 3 (RGB) or 4 (RGBA) samples on its last axis, not {data.shape[-1]}!')
    elif ndims not in (2, 3):
        raise Exception('data must be 3D or 2D, or 4D (z, y, x, c) color data!')
    transpose = False
    if data.flags['F_CONTIGUOUS'] or not data.flags['C_CONTIGUOUS']:
        data = np.ascontiguousarray(data)
    if rgb:
        if ndims == 3:
            data = data[np.newaxis]
        data = np.transpose(data, (2, 1, 0, 3))
    elif ndims == 3:
        data = np.transpose(data, (2, 1, 0))
    else:
        data = np.transpose(data, (1, 0))
    pybind11_write_tiff(file_name, data, transpose, compression, tile)
    return


def get_image_shape(file_name):
    return pybind11_get_image_shape(file_name)
