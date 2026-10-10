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


def strip_location(path, page, strip):
    """(offset, byte count) of one strip, for damaging it on purpose.

    Little-endian classic and BigTIFF, as written by this library.
    """
    with open(path, "rb") as fh:
        head = fh.read(16)
        big = struct.unpack("<H", head[2:4])[0] == 43
        cnt_fmt, ent, off_fmt = ("<Q", 20, "<Q") if big else ("<H", 12, "<I")
        cnt_sz, off_sz = struct.calcsize(cnt_fmt), struct.calcsize(off_fmt)
        ifd = struct.unpack(off_fmt, head[8:16] if big else head[4:8])[0]
        for _ in range(page):
            fh.seek(ifd)
            n = struct.unpack(cnt_fmt, fh.read(cnt_sz))[0]
            fh.seek(ifd + cnt_sz + n * ent)
            ifd = struct.unpack(off_fmt, fh.read(off_sz))[0]
        fh.seek(ifd)
        n = struct.unpack(cnt_fmt, fh.read(cnt_sz))[0]
        table = fh.read(n * ent)
        found = {}
        for i in range(n):
            e = table[i * ent:(i + 1) * ent]
            tag, typ = struct.unpack("<HH", e[:4])
            if tag not in (273, 279):
                continue
            count = struct.unpack("<Q" if big else "<I", e[4:12] if big else e[4:8])[0]
            size = {3: 2, 4: 4, 16: 8}[typ]
            field = e[12:20] if big else e[8:12]
            if count * size <= off_sz:
                raw = field[strip * size:(strip + 1) * size]
            else:
                fh.seek(struct.unpack(off_fmt, field[:off_sz])[0] + strip * size)
                raw = fh.read(size)
            found[tag] = int.from_bytes(raw, "little")
    return found[273], found[279]


def make(shape, dt, rng):
    n = int(np.prod(shape))
    if np.issubdtype(dt, np.integer):
        info = np.iinfo(dt)
        arr = (np.arange(n) % min(int(info.max), 4000)).astype(dt).reshape(shape)
        arr.flat[0] = info.min          # extremes catch sign/width mistakes
        arr.flat[1] = info.max
        return arr
    return (rng.standard_normal(shape) * 1000).astype(dt)


def write_imagej_tiff(path, arr, byteorder="<"):
    """Minimal single-IFD ImageJ-format TIFF: one IFD describing slice 0, every
    slice's raw data contiguous after it, and an ImageJ description carrying
    images=Z. This is the layout ImageJ's fast save (and tifffile's
    imagej=True, truncate=True) produce, which cpp-tiff reads through its
    ImageJ fast path -- the library's own writer cannot produce it, hence this
    hand-rolled fixture writer. arr is (z, y, x) or (z, y, x, 3)."""
    e = byteorder
    z, y, x = arr.shape[:3]
    spp = arr.shape[3] if arr.ndim == 4 else 1
    bits = arr.dtype.itemsize * 8
    desc = f"ImageJ=1.11a\nimages={z}\nslices={z}\n".encode() + b"\x00"
    if len(desc) % 2:
        desc += b"\x00"

    def entry(tag, typ, count, value_field):
        return struct.pack(e + "HHI", tag, typ, count) + value_field

    def short_v(v):
        return struct.pack(e + "H", v) + b"\x00\x00"

    def long_v(v):
        return struct.pack(e + "I", v)

    nentries = 11
    aux_off = 8 + 2 + nentries * 12 + 4
    bps_blob = struct.pack(e + f"{spp}H", *([bits] * spp)) if spp > 1 else b""
    desc_off = aux_off + len(bps_blob)
    pixel_off = desc_off + len(desc)
    pixel_off += pixel_off % 2

    ents = [
        entry(256, 3, 1, short_v(x)),                        # ImageWidth
        entry(257, 3, 1, short_v(y)),                        # ImageLength
        entry(258, 3, spp, short_v(bits) if spp == 1 else long_v(aux_off)),
        entry(259, 3, 1, short_v(1)),                        # Compression: none
        entry(262, 3, 1, short_v(2 if spp == 3 else 1)),     # Photometric
        entry(270, 2, len(desc), long_v(desc_off)),          # ImageDescription
        entry(273, 4, 1, long_v(pixel_off)),                 # StripOffsets
        entry(277, 3, 1, short_v(spp)),                      # SamplesPerPixel
        entry(278, 3, 1, short_v(y)),                        # RowsPerStrip
        entry(279, 4, 1, long_v(x * y * spp * (bits // 8))), # StripByteCounts
        entry(284, 3, 1, short_v(1)),                        # PlanarConfig: chunky
    ]
    with open(path, "wb") as f:
        f.write((b"II" if e == "<" else b"MM") + struct.pack(e + "HI", 42, 8))
        f.write(struct.pack(e + "H", nentries) + b"".join(ents) + struct.pack(e + "I", 0))
        f.write(bps_blob + desc)
        if f.tell() < pixel_off:
            f.write(b"\x00" * (pixel_off - f.tell()))
        f.write(np.ascontiguousarray(arr).astype(arr.dtype.newbyteorder(e)).tobytes())


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

    # 4. ImageJ-format files (single IFD + contiguous slices). Covers the ImageJ
    #    fast path: RGB scaling by samples, z-range start offset, and byte-order
    #    dependent swapping (real ImageJ writes big-endian, tifffile little-endian).
    yy, xx = np.meshgrid(np.arange(24), np.arange(40), indexing="ij")
    ij_rgb = np.zeros((5, 24, 40, 3), np.uint8)
    for k in range(5):
        ij_rgb[k, ..., 0] = (xx * 3 + k * 7) % 250
        ij_rgb[k, ..., 1] = (yy * 5 + k * 11) % 250
        ij_rgb[k, ..., 2] = (xx + yy * 2 + k * 13) % 250
    ij_g16 = (np.arange(4 * 24 * 40, dtype=np.uint16) % 6000).reshape(4, 24, 40)
    cases = [("ij_rgb", ij_rgb, "<"), ("ij_g16_le", ij_g16, "<"), ("ij_g16_be", ij_g16, ">")]
    for name, arr, bo in cases:
        path = os.path.join(tmp, name + ".tif")
        write_imagej_tiff(path, arr, bo)
        back = cpptiff.read_tiff(path)
        check(back.dtype == arr.dtype and back.shape == arr.shape
              and np.array_equal(back, arr), f"imagej {name}: full read")
        sub = cpptiff.read_tiff(path, [1, 3])
        check(np.array_equal(sub, arr[1:3]), f"imagej {name}: z-range [1,3)")
        os.remove(path)

    # 5. Unreadable and damaged files raise instead of crashing the interpreter.
    missing = os.path.join(tmp, "missing.tif")
    not_tiff = os.path.join(tmp, "not_a_tiff.tif")
    with open(not_tiff, "wb") as fh:
        fh.write(b"this is not a tiff file\n" * 100)
    for label, path in (("missing file", missing), ("non-TIFF file", not_tiff)):
        for fn_name, fn in (("read_tiff", lambda p=path: cpptiff.read_tiff(p)),
                            ("get_image_shape", lambda p=path: cpptiff.get_image_shape(p))):
            try:
                fn()
                check(False, f"{label}: {fn_name} did not raise")
            except Exception:
                pass
    os.remove(not_tiff)

    # Damage one LZW strip of page 2. Reading it must raise, through the 3D and
    # single-slice paths, while the undamaged pages still read correctly.
    vol = rng.integers(0, 65536, (4, 48, 64), dtype=np.uint16)
    path = os.path.join(tmp, "corrupt.tif")
    cpptiff.write_tiff(path, vol, "lzw")
    off, cnt = strip_location(path, page=2, strip=0)
    with open(path, "r+b") as fh:
        fh.seek(off + cnt // 4)
        fh.write(b"\xff" * min(256, cnt // 2))
    for label, zr in (("full read", None), ("damaged slice", [2, 3])):
        try:
            cpptiff.read_tiff(path, zr)
            check(False, f"corrupt strip: {label} did not raise")
        except RuntimeError:
            pass
    check(np.array_equal(cpptiff.read_tiff(path, [0, 2]), vol[0:2]), "corrupt strip: undamaged pages")
    os.remove(path)

    # An ImageJ stack cut short, as an interrupted copy leaves it. Reading past
    # the end must raise, while the complete slices still read.
    path = os.path.join(tmp, "ij_truncated.tif")
    write_imagej_tiff(path, ij_g16)
    with open(path, "r+b") as fh:
        fh.truncate(os.path.getsize(path) - ij_g16[0].nbytes // 2)
    for label, zr in (("full read", None), ("last slice", [3, 4])):
        try:
            cpptiff.read_tiff(path, zr)
            check(False, f"truncated imagej: {label} did not raise")
        except RuntimeError:
            pass
    check(np.array_equal(cpptiff.read_tiff(path, [0, 3]), ij_g16[0:3]), "truncated imagej: complete slices")
    os.remove(path)

    # Writes that cannot succeed raise instead of returning normally (the path
    # runs through a regular file, so its folder can never be created).
    blocker = os.path.join(tmp, "not_a_folder.txt")
    with open(blocker, "w") as fh:
        fh.write("x")
    try:
        cpptiff.write_tiff(os.path.join(blocker, "sub", "out.tif"), vol)
        check(False, "write below a regular file did not raise")
    except RuntimeError:
        pass
    os.remove(blocker)

    if _failures:
        for f in _failures[:20]:
            print("FAIL:", f)
        sys.exit(f"cpptiff Python tests FAILED ({len(_failures)} checks)")
    print("cpptiff Python tests PASSED")


if __name__ == "__main__":
    main()
