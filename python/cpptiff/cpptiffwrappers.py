import itertools
import numpy as np
import operator
import os
import threading
from concurrent.futures import ThreadPoolExecutor, wait
from .cpptiff import pybind11_read_tiff, pybind11_write_tiff, pybind11_get_image_shape, pybind11_default_threads
from .cpptiff import pybind11_read_shape, pybind11_read_tiff_gpu, pybind11_gpu_available, pybind11_gpu_set_nvcomp


def _threads(num_threads):
    # A call's number of threads: None (0 to the module) for the default,
    # OpenMP's (OMP_NUM_THREADS, or every CPU)
    if num_threads is None:
        return 0
    if isinstance(num_threads, bool) or int(num_threads) != num_threads or num_threads < 1:
        raise ValueError(f'num_threads must be a whole number of at least 1, not {num_threads!r}')
    return int(num_threads)


def _file_name(file_name):
    # A path as str or bytes (os.PathLike names, such as pathlib.Path, too), with a leading ~ or ~name expanded to
    # that home folder on Linux and macOS, as the MEX files and cpp-zarr expand it (os.path.expanduser): nothing else
    # in the path is expanded, and on Windows it is used as named
    try:
        file_name = os.fspath(file_name)
    except TypeError:
        raise TypeError(f'file_name must be a str, bytes or os.PathLike path, not {type(file_name).__name__}') from None
    return file_name if os.name == 'nt' else os.path.expanduser(file_name)


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


def read_tiff(file_name, z_range=None, num_threads=None, out=None, order='C', gpu=False, stream=None):
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
    out can also be GPU memory: a CuPy array, a PyTorch tensor on a CUDA device,
    or anything with __cuda_array_interface__. The strips' stored bytes are then
    read, copied to the GPU and decompressed there (zstd and deflate with nvCOMP;
    LZW and other codecs on the CPU), with the GPU work on stream (a PyTorch or
    CuPy stream, or a CUDA stream handle; by default the default stream), done
    when this returns. gpu=True returns a new CuPy array read that way. Needs an
    NVIDIA GPU and nvCOMP (pip install "cpp-tiff[gpu]"); see gpu_available().
    """
    file_name = _file_name(file_name)
    order = _order(order)
    z_range = _z_range(z_range)
    if not os.path.isfile(file_name):
        raise FileNotFoundError(f'{os.fsdecode(file_name)} does not exist!')
    if gpu or (out is not None and _cuda_array(out) is not None):
        return _read_gpu([(file_name, z_range, out)], order, num_threads, stream)[0]
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


def read_tiff_batch(requests, num_threads=None, order='C', gpu=False, stream=None):
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
    # With gpu=True, or GPU memory as outs (see read_tiff), the batch goes to the GPU
    # together, which decompresses all of its strips at once, on stream.
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
    if gpu or any(r[2] is not None and _cuda_array(r[2]) is not None for r in reqs):
        return _read_gpu([(_file_name(r[0]), _z_range(r[1]), r[2]) for r in reqs], order, num_threads, stream)
    total = _threads(num_threads) or pybind11_default_threads()
    names = [_file_name(r[0]) for r in reqs]
    sizes = sorted(os.path.getsize(f) if os.path.isfile(f) else 0 for f in names)
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


# ---- GPU reads (adapted from cpp-zarr's Python package)

_TORCH_TYPES = {'uint8': '|u1', 'int8': '|i1', 'uint16': '<u2', 'int16': '<i2', 'uint32': '<u4', 'int32': '<i4',
                'uint64': '<u8', 'int64': '<i8', 'float16': '<f2', 'float32': '<f4', 'float64': '<f8'}


def _contiguous(shape, strides):
    # (C-contiguous, F-contiguous) for element strides
    def ok(axes):
        expect = 1
        for d in axes:
            if shape[d] != 1 and strides[d] != expect:
                return False
            expect *= shape[d]
        return True
    n = len(shape)
    return ok(range(n - 1, -1, -1)), ok(range(n))


def _cuda_array(a):
    # (device pointer, NumPy type string, shape, C-contiguous, F-contiguous) of GPU memory a (a PyTorch tensor
    # on a CUDA device, or anything with __cuda_array_interface__, such as a CuPy array), or None when a is not
    # GPU memory
    if type(a).__module__.split('.')[0] == 'torch' and getattr(a, 'is_cuda', False):
        name = str(a.dtype).replace('torch.', '')
        if name not in _TORCH_TYPES:
            raise TypeError(f'out has the data type {a.dtype}, which cpp-tiff does not read')
        shape = tuple(int(v) for v in a.shape)
        c, f = _contiguous(shape, tuple(a.stride()))
        return int(a.data_ptr()), _TORCH_TYPES[name], shape, c, f
    cai = getattr(a, '__cuda_array_interface__', None)
    if cai is None:
        return None
    shape = tuple(int(v) for v in cai['shape'])
    dt = np.dtype(cai['typestr'])
    strides = cai.get('strides')
    if strides is None:
        c, f = True, len([v for v in shape if v != 1]) <= 1
    else:
        if any(st % dt.itemsize for st in strides):
            raise ValueError('out must be C- or F-contiguous')
        c, f = _contiguous(shape, tuple(st // dt.itemsize for st in strides))
    if cai.get('readonly', False) or (len(cai['data']) > 1 and cai['data'][1]):
        raise ValueError('out must be writeable')
    return int(cai['data'][0]), dt.str if dt.itemsize > 1 else '|' + dt.str[1:], shape, c, f


def _stream_handle(stream):
    # A CUDA stream's handle: of a PyTorch stream (cuda_stream), a CuPy stream (ptr), or an int
    if stream is None:
        return 0
    for attr in ('cuda_stream', 'ptr'):
        if hasattr(stream, attr):
            return int(getattr(stream, attr))
    if isinstance(stream, int):
        return stream
    raise TypeError(f'stream must be a PyTorch or CuPy stream, or a CUDA stream handle, not {stream!r}')


_nvcomp_looked = False


def _find_nvcomp():
    # nvCOMP from its pip package, for the library to load, found from pip's record of the files it installed
    # (importing the package fails where another package named nvidia hides it, as in NVIDIA's PyTorch
    # containers): the CUDA 12 build first, which the gpu extra installs and which runs on every GPU from Volta
    # on (CUDA 13 builds need a Turing or newer one), then the CUDA 13 one and the older nvidia-nvcomp layout.
    # CPPTIFF_NVCOMP_LIBRARY, when set, comes first.
    global _nvcomp_looked
    if _nvcomp_looked:
        return
    _nvcomp_looked = True
    if os.environ.get('CPPTIFF_NVCOMP_LIBRARY'):
        return
    from importlib import metadata
    for name in ('nvidia-libnvcomp-cu12', 'nvidia-libnvcomp-cu13', 'nvidia-nvcomp-cu12', 'nvidia-nvcomp-cu13'):
        dist = None
        for spelling in (name, name.replace('-', '_')):
            try:
                dist = metadata.distribution(spelling)
                break
            except metadata.PackageNotFoundError:
                pass
        if dist is None:
            continue
        libs = sorted(str(dist.locate_file(f)) for f in (dist.files or [])
                      if (f.name.startswith('libnvcomp.so') or (f.name.startswith('nvcomp') and f.name.endswith('.dll')))
                      and 'cpu' not in f.name)
        libs = [p for p in libs if os.path.exists(p)]
        if libs:
            pybind11_gpu_set_nvcomp(libs[-1])
            return


def gpu_available():
    """(True, '') when reads into GPU memory can run here; (False, why) when not (no NVIDIA driver or GPU, no
    nvCOMP: pip install "cpp-tiff[gpu]")"""
    _find_nvcomp()
    return pybind11_gpu_available()


def _read_gpu(requests, order, num_threads, stream):
    # requests: (file_name, z_range, out) with out GPU memory, or None for a new CuPy array; read in one GPU
    # batch. Returns the outs.
    _find_nvcomp()
    outs = []
    for file_name, z_range, out in requests:
        if out is None:
            try:
                import cupy
            except ImportError:
                raise ImportError('gpu=True makes CuPy arrays, and CuPy is not installed (pip install cupy-cuda12x); or '
                                  'pass GPU memory to read into as out= (a PyTorch tensor on a CUDA device, for one)') from None
            shape, dtype = pybind11_read_shape(file_name, z_range or [], order)
            out = cupy.empty(tuple(shape), dtype=dtype, order=order)
        elif _cuda_array(out) is None:
            raise TypeError('out must be GPU memory for a read into GPU memory (a CuPy array, a PyTorch CUDA tensor, ...)')
        outs.append(out)
    reqs = []
    for (file_name, z_range, _), out in zip(requests, outs):
        ptr, typestr, shape, c, f = _cuda_array(out)
        if order == 'C' and not c:
            raise ValueError("out must be C-contiguous (or pass order='F' for a (y, x, z) F-contiguous out)")
        if order == 'F' and not f:
            raise ValueError("out must be F-contiguous for order='F'")
        reqs.append((file_name, list(z_range or []), ptr, typestr, list(shape)))
    pybind11_read_tiff_gpu(reqs, order, _threads(num_threads), _stream_handle(stream))
    return outs
