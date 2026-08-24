"""Smoke test for the cpp-tiff Python wheel.

Writes small volumes, reads them back, and checks the data survives the round
trip -- across every supported dtype, each compression, 2D and 3D, striped and
tiled, plus z-range reads. Run automatically by cibuildwheel against the freshly
built+installed wheel, so it also verifies the wheel imports and bundles its
native library. Exits non-zero on any failure.

Only numpy is available here (see test-requires in pyproject.toml), so file
layout is checked with the small TIFF tag reader below rather than tifffile.
"""
import os
import struct
import sys
import tempfile

import numpy as np
import cpptiff


DTYPES = [np.uint8, np.int8, np.uint16, np.int16, np.uint32, np.int32,
          np.float32, np.uint64, np.int64, np.float64]
COMPRESSIONS = ["none", "lzw", "zstd"]

_failures = []


def check(cond, msg):
    if not cond:
        _failures.append(msg)
    return cond


def tile_width(path):
    """TileWidth (tag 322) of the first IFD, or None if the file is striped.

    Handles the classic and BigTIFF headers libtiff writes. Only little-endian
    files are decoded (every platform cpp-tiff targets is little-endian); a
    big-endian file returns None so the caller simply skips the layout check.
    """
    with open(path, "rb") as fh:
        head = fh.read(16)
        if head[:2] != b"II":
            return None
        magic = struct.unpack("<H", head[2:4])[0]
        # The IFD usually sits after the image data, so seek to it rather than
        # assuming it is near the start of the file.
        if magic == 42:      # classic TIFF: 12-byte entries, 4-byte offsets
            fh.seek(struct.unpack("<I", head[4:8])[0])
            count, entry = struct.unpack("<H", fh.read(2))[0], 12
        elif magic == 43:    # BigTIFF: 20-byte entries, 8-byte offsets
            fh.seek(struct.unpack("<Q", head[8:16])[0])
            count, entry = struct.unpack("<Q", fh.read(8))[0], 20
        else:
            return None
        if count > 4096:     # not a directory we understand
            return None
        table = fh.read(count * entry)
    for i in range(count):
        e = table[i * entry:(i + 1) * entry]
        if len(e) < entry:
            break
        tag, typ = struct.unpack("<HH", e[:4])
        if tag != 322:                       # TIFFTAG_TILEWIDTH
            continue
        # Value field: BigTIFF is tag(2)+type(2)+count(8) then 8 bytes of value,
        # classic is tag(2)+type(2)+count(4) then 4. Little-endian, so left-aligned.
        value = e[12:20] if entry == 20 else e[8:12]
        width = {3: 2, 4: 4, 16: 8}.get(typ)
        if width is None:
            return None
        return int.from_bytes(value[:width], "little")
    return None


def make(shape, dt, rng):
    n = int(np.prod(shape))
    if np.issubdtype(dt, np.integer):
        info = np.iinfo(dt)
        arr = (np.arange(n) % min(int(info.max), 4000)).astype(dt).reshape(shape)
        arr.flat[0] = info.min          # extremes catch sign/width mistakes
        arr.flat[1] = info.max
        return arr
    return (rng.standard_normal(shape) * 1000).astype(dt)


def main():
    rng = np.random.default_rng(1234567)
    tmp = tempfile.mkdtemp()

    # 1. Every dtype x compression, 3D and 2D, striped and tiled.
    for dt in DTYPES:
        name = np.dtype(dt).name
        for comp in COMPRESSIONS:
            for tile in (0, 16):
                for shape in ((3, 24, 40), (24, 40)):
                    data = make(shape, dt, rng)
                    tag = f"{name}/{comp}/tile{tile}/{len(shape)}D"
                    path = os.path.join(tmp, f"rt_{name}_{comp}_{tile}_{len(shape)}.tif")
                    cpptiff.write_tiff(path, data, comp, tile)

                    back = cpptiff.read_tiff(path)
                    check(back.dtype == data.dtype, f"{tag}: dtype {back.dtype}")
                    check(back.shape == data.shape, f"{tag}: shape {back.shape}")
                    check(np.array_equal(back, data), f"{tag}: data mismatch")

                    # A garbage z (the 2D shape[2] bug) blew the file up ~250 MB.
                    raw = int(np.prod(shape)) * data.dtype.itemsize
                    check(os.path.getsize(path) < raw * 10 + 100_000,
                          f"{tag}: file size {os.path.getsize(path)} vs raw {raw}")

                    tw = tile_width(path)
                    if tw is not None or tile:
                        check(tw == (tile or None), f"{tag}: TileWidth={tw}")
                    os.remove(path)

    # 2. z-range reads, striped and tiled. read_tiff squeezes a single slice.
    vol = (np.arange(6 * 40 * 50, dtype=np.uint16) % 4000).reshape(6, 40, 50)
    for tile in (0, 16):
        path = os.path.join(tmp, f"zr_{tile}.tif")
        cpptiff.write_tiff(path, vol, "lzw", tile)
        check(np.array_equal(cpptiff.read_tiff(path), vol), f"zrange tile{tile}: full")
        for a, b in ((0, 2), (2, 5), (4, 6), (3, 4)):
            want = vol[a:b]
            want = want[0] if want.shape[0] == 1 else want
            check(np.array_equal(cpptiff.read_tiff(path, [a, b]), want),
                  f"zrange tile{tile}: [{a},{b})")
        check(cpptiff.get_image_shape(path) == (6, 40, 50), f"zrange tile{tile}: shape")
        os.remove(path)

    # 3. A ragged tiled volume, so the padded edge tiles are exercised.
    ragged = (np.arange(2 * 70 * 90, dtype=np.uint16) % 4000).reshape(2, 70, 90)
    for tile in (16, 64):
        path = os.path.join(tmp, f"ragged_{tile}.tif")
        cpptiff.write_tiff(path, ragged, "zstd", tile)
        check(tile_width(path) == tile, f"ragged tile{tile}: not tiled")
        check(np.array_equal(cpptiff.read_tiff(path), ragged), f"ragged tile{tile}: data")
        os.remove(path)

    if _failures:
        for f in _failures[:20]:
            print("FAIL:", f)
        sys.exit(f"cpptiff Python tests FAILED ({len(_failures)} checks)")
    print("cpptiff Python tests PASSED")


if __name__ == "__main__":
    main()
