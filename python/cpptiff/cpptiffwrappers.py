import itertools
import numpy as np
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
        raise Exception(f'num_threads must be a whole number of at least 1, not {num_threads!r}')
    return int(num_threads)


def read_tiff(file_name, z_range=None, num_threads=None, out=None):
    # num_threads: the threads this call uses (by default OMP_NUM_THREADS, or
    # every CPU). The GIL is released while reading, so reads from several Python
    # threads run at once: give each a share of the CPUs.
    # out: a NumPy array to read into instead of a new one, of the shape read_tiff
    # returns and the file's data type, writeable and C-contiguous; it is returned.
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

    if out is not None:
        return pybind11_read_tiff(file_name, z_range, _threads(num_threads), out)
    im = pybind11_read_tiff(file_name, z_range, _threads(num_threads))
    if im.ndim == 4:
        im = np.transpose(im, (2, 1, 0, 3))  # (x,y,z,c) -> (z,y,x,c) for RGB
    else:
        im = np.transpose(im, (2, 1, 0))     # (x,y,z) -> (z,y,x)
    if im.shape[0] == 1:
        im = np.squeeze(im, axis=0)
    return im


def write_tiff(file_name, data, compression='lzw', tile=0, rgb=None, num_threads=None):
    """Write data as a TIFF file, one page per slice.

    Grayscale data is (z, y, x) or (y, x). RGB and RGBA data has its 3 or 4
    color samples last: (z, y, x, c), or (y, x, c) for one image. rgb=None
    treats 4D data as color and anything else as grayscale, so a single
    (y, x, c) color image needs rgb=True. num_threads: the threads this call
    uses, as for read_tiff (the GIL is released while writing).
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


def read_tiff_batch(requests, num_threads=None):
    # Read many files (or slice ranges of them) at once, as reading them one after
    # another with read_tiff does, but with the reads overlapping (the GIL is
    # released while reading): the way to read a batch of files when each is small
    # or the storage is slow. requests: a list of file names, or of
    # (file_name, z_range) or (file_name, z_range, out) tuples, where z_range and
    # out may be None. Returns the arrays, in the order of the requests.
    # num_threads: the threads the batch uses (by default, as for read_tiff,
    # OMP_NUM_THREADS, or every CPU). Each read gets a thread per 16 MB of file (at
    # least one, at most all of them), and as many reads as the threads allow run
    # at once.
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
                results[i] = read_tiff(file_name, z_range, num_threads=each, out=out)
            except BaseException:
                failed.append(True)
                raise

    futures = [_pool(each).submit(reader, each) for each in threads]
    # (a failed read's error is raised once the reads under way have ended)
    wait(futures)
    for f in futures:
        f.result()
    return results


def get_image_shape(file_name):
    return pybind11_get_image_shape(file_name)
