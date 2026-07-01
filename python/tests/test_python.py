"""Smoke test for the cpp-tiff Python wheel.

Writes small random volumes, reads them back, and checks the data survives the
round trip (plus a z-range read). Run automatically by cibuildwheel against the
freshly built+installed wheel, so it also verifies the wheel imports and bundles
its native library. Exits non-zero on any failure.
"""
import os
import sys
import tempfile

import numpy as np
import cpptiff


def main():
    rng = np.random.default_rng(1234567)
    shape = (3, 24, 40)  # z, y, x
    dtypes = [np.uint8, np.uint16, np.int16, np.int32, np.float32, np.float64]

    tmp = tempfile.mkdtemp()
    ok = True
    for dt in dtypes:
        name = np.dtype(dt).name
        if np.issubdtype(dt, np.integer):
            info = np.iinfo(dt)
            lo, hi = max(info.min, -1000), min(info.max, 1000)
            data = rng.integers(lo, hi, size=shape, endpoint=True).astype(dt)
        else:
            data = (rng.standard_normal(shape) * 1000).astype(dt)

        path = os.path.join(tmp, f"rt_{name}.tif")
        cpptiff.write_tiff(path, data)

        back = cpptiff.read_tiff(path)
        sub = cpptiff.read_tiff(path, [0, 2])  # z-range: first two slices (0-indexed, half-open)

        good = (back.dtype == data.dtype and back.shape == data.shape
                and np.array_equal(back, data) and np.array_equal(sub, data[0:2]))
        print(f"{name:8s} {'OK' if good else 'FAIL'}")
        ok = ok and good

    if not ok:
        sys.exit("cpptiff Python round-trip test FAILED")
    print("cpptiff Python round-trip tests PASSED")


if __name__ == "__main__":
    main()
