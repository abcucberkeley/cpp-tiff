import itertools
import numpy as np
import operator
import os
import threading
from concurrent.futures import ThreadPoolExecutor, wait
from .cpptiff import pybind11_read_tiff, pybind11_write_tiff, pybind11_get_image_shape, pybind11_default_threads


def _threads(num_threads):
    # A call's number of threads: None (0 to the module) for the default,
    # OpenMP's (OMP_NUM_THREADS, or every CPU)
    if num_threads is None:
        return 0
    if isinstance(num_threads, bool) or int(num_threads) != num_threads or num_threads < 1:
        raise ValueError(f'num_threads must be a whole number of at least 1, not {num_threads!r}')
    return int(num_threads)


def _file_name(file_name):
    # A path as str or bytes: os.PathLike names, such as pathlib.Path, too
    try:
        return os.fspath(file_name)
    except TypeError:
        raise TypeError(f'file_name must be a str, bytes or os.PathLike path, not {type(file_name).__name__}') from None


def _order(order):
    # 'C' (or None) for (z, y, x[, c]) arrays, 'F' for MATLAB's (y, x[, c], z)
    if order is None:
        return 'C'
    if order not in ('C', 'F'):
        raise ValueError(f"order must be 'C' or 'F', not {order!r}")
    return order


def _slice_number(value):
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f'a z_range value must be a whole number, not {value!r}')
    n = operator.index(value)
    if n < 0:
        raise ValueError(f'z_range values must be at least 0, not {n}')
    return n


def _z_range(z_range):
    # [] for every slice (None, or an empty sequence), [s] for slice s, or
    # [start, end) from any sequence of two whole numbers (a list, tuple, range or
    # NumPy array)
    if z_range is None:
        return []
    wrong = f'z_range must be a slice number or a sequence of two, [start, end), not {z_range!r}'
    if isinstance(z_range, (str, bytes)):
        raise TypeError(wrong)
    # range(1, 3) is slices 1 and 2: as a list, [1, 2], it would be slice 1 alone
    if isinstance(z_range, range):
        if z_range.step != 1 or not len(z_range):
            raise ValueError(f'z_range {z_range!r} must be a non-empty range with a step of 1')
        z_range = [z_range.start, z_range.stop]
    try:
        return [_slice_number(z_range)]
    except TypeError:
        if isinstance(z_range, (bool, np.bool_)):
            raise
    try:
        values = list(z_range)
    except TypeError:
        raise TypeError(wrong) from None
    if not values:
        return []  # (as before: every slice)
    if len(values) > 2:
        raise ValueError(f'z_range must be a slice number or a sequence of two, [start, end), not {len(values)} values')
    try:
        values = [_slice_number(v) for v in values]
    except TypeError:
        raise TypeError(f'z_range values must be whole numbers, not {z_range!r}') from None
    if len(values) == 2 and values[0] >= values[1]:
        raise ValueError(f'z_range [{values[0]}, {values[1]}] is empty: its start must be less than its end')
    return values


_COMPRESSIONS = {'lzw': 'lzw', 'zstd': 'zstd', 'none': 'none', '': 'none'}


def _compression(compression):
    # A compression name in any case, or None for 'none'. Other names were
    # written as LZW, and 'ZSTD' and 'NONE' too.
    if compression is None:
        return 'none'
    name = _COMPRESSIONS.get(compression.lower()) if isinstance(compression, str) else None
    if name is None:
        raise ValueError(f"compression must be 'lzw', 'zstd' or 'none', not {compression!r}")
    return name


def read_tiff(file_name, z_range=None, num_threads=None, out=None, order='C'):
    """Read a TIFF file into a NumPy array.

    order='C' (the default) returns (z, y, x), or (z, y, x, c) for RGB and RGBA
    files, C-contiguous. order='F' returns the array MATLAB's parallelReadTiff
    returns: (y, x, z), or (y, x, c, z), F-contiguous, filled in that layout as
    the file is decoded (no copy). Either way a single slice comes back without
    its z axis: (y, x[, c]).

    z_range: one slice, or [start, end) as a sequence of two (a list, tuple or
    NumPy array); 0-based, end excluded. By default every slice.
    num_threads: the threads this call uses (by default OMP_NUM_THREADS, or
    every CPU). The GIL is released while reading, so reads from several Python
    threads run at once: give each a share of the CPUs.
    out: a NumPy array to read into instead of a new one, of the shape read_tiff
    returns (or with a z axis of 1 for a single slice) and the file's data type,
    writeable and C-contiguous (F-contiguous for order='F'). It is returned.
    """
    file_name = _file_name(file_name)
    order = _order(order)
    z_range = _z_range(z_range)
    if not os.path.isfile(file_name):
        raise FileNotFoundError(f'{os.fsdecode(file_name)} does not exist!')
    if out is not None:
        return pybind11_read_tiff(file_name, z_range, _threads(num_threads), out, order)
    im = pybind11_read_tiff(file_name, z_range, _threads(num_threads), None, order)
    if order == 'F':
        # (y, x[, c], z)
        return np.squeeze(im, axis=-1) if im.shape[-1] == 1 else im
    if im.ndim == 4:
        im = np.transpose(im, (2, 1, 0, 3))  # (x,y,z,c) -> (z,y,x,c) for RGB
    else:
        im = np.transpose(im, (2, 1, 0))     # (x,y,z) -> (z,y,x)
    if im.shape[0] == 1:
        im = np.squeeze(im, axis=0)
    return im


def write_tiff(file_name, data, compression='lzw', tile=0, rgb=None, num_threads=None, order='C'):
    """Write data as a TIFF file, one page per slice.

    order='C' (the default): grayscale data is (z, y, x) or (y, x); RGB and RGBA
    data has its 3 or 4 color samples last, (z, y, x, c), or (y, x, c) for one
    image. order='F': the layout MATLAB's parallelWriteTiff takes, (y, x, z) or
    (y, x), and (y, x, c, z) or (y, x, c) for color; F-contiguous data is written
    as it is, without a copy. The same values give the same file either way.
    rgb=None treats 4D data as color and anything else as grayscale, so a single
    color image needs rgb=True.

    compression: 'lzw' (the default), 'zstd' or 'none', in any case (None is
    'none'). Data in the other byte order is converted first.
    num_threads: the threads this call uses, as for read_tiff (the GIL is
    released while writing).
    """
    file_name = _file_name(file_name)
    order = _order(order)
    compression = _compression(compression)
    data = np.asarray(data)
    ndims = data.ndim
    if rgb is None:
        rgb = ndims == 4
    if order == 'F':
        if rgb:
            if ndims not in (3, 4):
                raise ValueError("RGB data must be (y, x, c) or (y, x, c, z) with order='F'!")
            if data.shape[2] not in (3, 4):
                raise ValueError(f"RGB data must have 3 (RGB) or 4 (RGBA) samples on its third axis with order='F', not {data.shape[2]}!")
        elif ndims not in (2, 3):
            raise ValueError("data must be (y, x, z) or (y, x), or 4D (y, x, c, z) color data, with order='F'!")
    elif rgb:
        if ndims not in (3, 4):
            raise ValueError('RGB data must be (y, x, c) or (z, y, x, c)!')
        if data.shape[-1] not in (3, 4):
            raise ValueError(f'RGB data must have 3 (RGB) or 4 (RGBA) samples on its last axis, not {data.shape[-1]}!')
    elif ndims not in (2, 3):
        raise ValueError('data must be 3D or 2D, or 4D (z, y, x, c) color data!')
    # An empty array was written as one slice read from past its end
    if data.size == 0:
        raise ValueError(f'cannot write an empty array of shape {data.shape}: a TIFF image has at least one pixel')
    if data.dtype.kind not in 'uif' or data.dtype.itemsize not in (1, 2, 4, 8) or data.dtype.kind + str(data.dtype.itemsize) == 'f1':
        raise TypeError(f'data type {data.dtype} cannot be written: use integers of 8, 16, 32 or 64 bits, '
                        'or floats of 16, 32 or 64 bits')
    # The file's bytes are the data's: big-endian data was written with every
    # value wrong
    if not data.dtype.isnative:
        data = data.astype(data.dtype.newbyteorder('='))
    if order == 'F':
        # The writer transposes MATLAB's column-major data as it compresses it,
        # as the mex does: no copy for F-contiguous data
        if rgb and ndims == 3:
            data = data[:, :, :, np.newaxis]
        data = np.asfortranarray(data)
        if rgb:
            view = np.transpose(data, (1, 0, 3, 2))  # (y,x,c,z) -> (x,y,z,c), the module's view
        elif ndims == 3:
            view = np.transpose(data, (1, 0, 2))     # (y,x,z) -> (x,y,z)
        else:
            view = data.T                            # (y,x) -> (x,y)
        pybind11_write_tiff(file_name, view, True, compression, tile, _threads(num_threads))
        return
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
    pybind11_write_tiff(file_name, data, transpose, compression, tile, _threads(num_threads))
    return


# The threads that run batches of reads, one pool per number of threads each read
# uses: a thread keeps its OpenMP team at one size (OpenMP runtimes end and
# recreate threads when a thread's team changes size, which made small reads
# several times slower)
_pools = {}
_pools_lock = threading.Lock()
_pools_pid = os.getpid()


def _pool(threads_per_read):
    global _pools_pid
    with _pools_lock:
        # (a forked child has the pools but none of their threads: it makes its own)
        if os.getpid() != _pools_pid:
            _pools.clear()
            _pools_pid = os.getpid()
        pool = _pools.get(threads_per_read)
        if pool is None:
            cpus = len(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else (os.cpu_count() or 1)
            pool = _pools[threads_per_read] = ThreadPoolExecutor(max_workers=max(cpus, 1),
                                                                 thread_name_prefix=f'cpptiff{threads_per_read}')
        return pool


# Bytes of file a read's thread works on: files up to this size are read with one
# thread each, larger ones with a thread per this many. (A file's size is known
# without opening it, unlike its dimensions.)
_BYTES_PER_THREAD = 16 << 20


def read_tiff_batch(requests, num_threads=None, order='C'):
    # Read many files (or slice ranges of them) at once, as reading them one after
    # another with read_tiff does, but with the reads overlapping (the GIL is
    # released while reading): the way to read a batch of files when each is small
    # or the storage is slow. requests: a list of file names, or of
    # (file_name, z_range) or (file_name, z_range, out) tuples, where z_range and
    # out may be None. Returns the arrays, in the order of the requests, in the
    # layout order gives (as for read_tiff).
    # num_threads: the threads the batch uses (by default, as for read_tiff,
    # OMP_NUM_THREADS, or every CPU). Each read gets a thread per 16 MB of file (at
    # least one, at most all of them), and as many reads as the threads allow run
    # at once.
    order = _order(order)
    reqs = []
    for r in requests:
        if isinstance(r, (str, bytes, os.PathLike)):
            r = (r,)
        r = tuple(r)
        if not 1 <= len(r) <= 3:
            raise Exception('Each request is a file name, (file_name, z_range) or (file_name, z_range, out)')
        reqs.append(r + (None,) * (3 - len(r)))
    if not reqs:
        return []
    total = _threads(num_threads) or pybind11_default_threads()
    sizes = sorted(os.path.getsize(r[0]) if os.path.isfile(r[0]) else 0 for r in reqs)
    per_read = min(total, max(1, round(sizes[len(sizes) // 2] / _BYTES_PER_THREAD)))
    at_once = max(1, min(len(reqs), round(total / per_read)))
    # (the threads split as evenly as they go: some reads may get one more)
    threads = [total // at_once + (i < total % at_once) for i in range(at_once)]
    results = [None] * len(reqs)
    following = itertools.count()
    lock = threading.Lock()
    failed = []

    def reader(each):
        # (each task takes the next request until none are left, or a read has failed)
        while not failed:
            with lock:
                i = next(following)
            if i >= len(reqs):
                return
            file_name, z_range, out = reqs[i]
            try:
                results[i] = read_tiff(file_name, z_range, num_threads=each, out=out, order=order)
            except BaseException:
                failed.append(True)
                raise

    futures = [_pool(each).submit(reader, each) for each in threads]
    # (a failed read's error is raised once the reads under way have ended)
    wait(futures)
    for f in futures:
        f.result()
    return results


def get_image_shape(file_name, order='C'):
    """The shape read_tiff returns for the whole file, without reading it:
    (z, y, x), or (z, y, x, c) for RGB and RGBA files, and with order='F'
    MATLAB's (y, x, z) or (y, x, c, z). z is 1 for a single image, whose z axis
    read_tiff leaves out."""
    shape = pybind11_get_image_shape(_file_name(file_name))
    if _order(order) == 'F':
        shape = shape[1:3] + shape[3:] + shape[:1]  # (z,y,x[,c]) -> (y,x[,c],z)
    return shape
