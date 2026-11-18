#!/usr/bin/env python3
# No wheel names a folder of the machine that built it: every member of every wheel (the modules and the libraries
# bundled with them, RECORD, METADATA, WHEEL, delvewheel's DELVEWHEEL record, .py files and the rest) is searched as
# bytes for a user's home folder (C:\Users\<user>\, /home/<user>/, /Users/<user>/), a temporary build environment
# (/tmp/build-env-..., ...\Temp\build-env-..., /tmp/pip-...) and cibuildwheel's own folders (cibw-run-...,
# pypa\cibuildwheel\Cache), and for any pattern given with --forbid (a build machine's or account's name). The
# released 1.7.3 Windows wheel's module held 14 such strings, of the form
# C:\Users\<user>\AppData\Local\Temp\build-env-xxxx\Lib\site-packages\pybind11\include/... and
# C:\Users\<user>\AppData\Local\pypa\cibuildwheel\Cache\..., from __FILE__ in pybind11's and Python's headers, and the
# 1.7.3 Linux modules held /tmp/build-env-xxxxxxxx/... in their debug information. The neutral folders the builds now
# use on purpose (C:\cibw-tmp, C:\mex-tmp, relative names such as tmp/..., python-include, pybind11-include) and a
# bare /tmp, which system libraries hold, are no failure. Compressed ELF sections (SHF_COMPRESSED, .zdebug_*) are
# searched after decompression, so compressed debug information can hide nothing.
# usage: check_wheel_paths.py [--forbid REGEX ...] <wheel or folder> ...  (a folder means every *.whl in it)
# exit: 0 if all is clean, 1 on any finding, 2 on bad arguments or a missing or unreadable wheel
import argparse
import os
import re
import struct
import sys
import zipfile
import zlib

# A name in a path: no separator, no NUL or control byte, none of Windows' reserved characters (so a placeholder
# such as <user> is no name). Windows names may hold spaces, POSIX ones here not (names in strings end at a space).
_WIN_NAME = rb'[^\\/\x00-\x1f"<>|:*?]+'
_POSIX_NAME = rb'[A-Za-z0-9_.][A-Za-z0-9_.-]*'
PATTERNS = [
    ("Windows user folder", rb'(?i)[A-Z]:[\\/]+Users[\\/]+' + _WIN_NAME + rb'[\\/]'),
    ("POSIX home folder", rb'(?<![A-Za-z0-9_.-])/home/' + _POSIX_NAME + rb'/'),
    # Not after a drive letter's colon, which the Windows pattern reports
    ("macOS user folder", rb'(?<![A-Za-z0-9_.:-])/Users/' + _POSIX_NAME + rb'/'),
    ("temporary build environment", rb'/tmp/+(?:build-env-|pip-)'),
    ("temporary build environment", rb'(?i)[\\/]Temp[\\/]+(?:build-env-|pip-)'),
    ("cibuildwheel temp folder", rb'cibw-run-'),
    ("cibuildwheel cache", rb'(?i)pypa[\\/]+cibuildwheel[\\/]+Cache'),
]
MAX_PER_MEMBER = 5
SHOW = 120


def _printable(c):
    return 0x20 <= c < 0x7f


def string_start(data, pos):
    """Where the printable run that holds pos starts (looking back at most 4096 bytes)."""
    lo = pos
    while lo > 0 and pos - lo < 4096 and _printable(data[lo - 1]):
        lo -= 1
    return lo


def printable_context(data, lo, start, end):
    """The printable run from lo (at most 40 bytes before the match), cut to SHOW characters."""
    lo = max(lo, start - 40)
    hi = end
    while hi < len(data) and hi - lo < SHOW and _printable(data[hi]):
        hi += 1
    more = hi < len(data) and _printable(data[hi])
    text = "".join(chr(c) if _printable(c) else "\\x%02x" % c for c in data[lo:hi])
    return text + ("..." if more else "")


def elf_compressed_sections(data):
    """Yield (name, bytes) for each compressed section of an ELF file, decompressed; nothing for other files."""
    if data[:4] != b"\x7fELF" or len(data) < 64:
        return
    is64 = data[4] == 2
    end = "<" if data[5] == 1 else ">"
    try:
        if is64:
            shoff, = struct.unpack_from(end + "Q", data, 0x28)
            shentsize, shnum, shstrndx = struct.unpack_from(end + "HHH", data, 0x3a)
            fmt, chfmt = end + "IIQQQQIIQQ", end + "IIQQ"
        else:
            shoff, = struct.unpack_from(end + "I", data, 0x20)
            shentsize, shnum, shstrndx = struct.unpack_from(end + "HHH", data, 0x2e)
            fmt, chfmt = end + "IIIIIIIIII", end + "III"
        sections = [struct.unpack_from(fmt, data, shoff + i * shentsize) for i in range(shnum)]
        stroff = sections[shstrndx][4]
    except (struct.error, IndexError):
        return
    for sec in sections:
        name_off, flags, offset, size = sec[0], sec[2], sec[4], sec[5]
        nul = data.find(b"\0", stroff + name_off)
        name = data[stroff + name_off:nul].decode("latin-1", "replace")
        raw = data[offset:offset + size]
        try:
            if flags & 0x800:  # SHF_COMPRESSED: an Elf_Chdr, then the data
                ch_type = struct.unpack_from(chfmt, raw, 0)[0]
                body = raw[struct.calcsize(chfmt):]
                if ch_type == 1:  # ELFCOMPRESS_ZLIB
                    yield name, zlib.decompress(body)
                else:
                    yield name, None  # ELFCOMPRESS_ZSTD, which the standard library cannot read before 3.14
            elif name.startswith(".zdebug") and raw[:4] == b"ZLIB":
                yield name, zlib.decompress(raw[12:])
        except (struct.error, zlib.error):
            yield name, None


def scan_bytes(data, patterns):
    """Return the contexts of the first MAX_PER_MEMBER strings with matches and the number of such strings (a
    string that several patterns match, as C:\\Users\\<user>\\AppData\\Local\\Temp\\build-env-... does, is one
    finding)."""
    found = {}
    for label, rx in patterns:
        for m in rx.finditer(data):
            key = string_start(data, m.start())
            if key not in found:
                found[key] = (m.start(), m.end(), [label])
            elif label not in found[key][2]:
                found[key][2].append(label)
    shown = []
    for key in sorted(found)[:MAX_PER_MEMBER]:
        start, end, labels = found[key]
        shown.append("%s [%s]" % (printable_context(data, key, start, end), ", ".join(labels)))
    return shown, len(found)


def check_wheel(path, patterns):
    """Print the findings of one wheel and its summary line; return its number of findings."""
    findings, members, notes = 0, 0, []
    with zipfile.ZipFile(path) as zf:
        for info in zf.infolist():
            if info.is_dir():
                continue
            members += 1
            data = zf.read(info)
            parts = [(info.filename, data)]
            for sec, body in elf_compressed_sections(data):
                if body is None:
                    notes.append("%s: section %s could not be decompressed and was not searched" % (info.filename, sec))
                else:
                    parts.append(("%s[%s]" % (info.filename, sec), body))
            for name, body in parts:
                shown, total = scan_bytes(body, patterns)
                findings += total
                for s in shown:
                    print("%s: %s: %s" % (path, name, s))
                if total > len(shown):
                    print("%s: %s: ... and %d more" % (path, name, total - len(shown)))
    for n in notes:
        print("%s: note: %s" % (path, n))
    print("%s: %s, %d findings in %d members" % (path, "FAIL" if findings else "ok", findings, members))
    return findings


def main():
    ap = argparse.ArgumentParser(description="Fail if any wheel names a build machine's folders.")
    ap.add_argument("--forbid", action="append", default=[], metavar="REGEX",
                    help="another pattern to fail on, such as a build machine's or account's name (repeatable)")
    ap.add_argument("paths", nargs="+", metavar="wheel or folder")
    args = ap.parse_args()  # exits 2 on bad arguments

    patterns = [(label, re.compile(rx)) for label, rx in PATTERNS]
    for i, rx in enumerate(args.forbid):
        try:
            patterns.append(("--forbid %d" % (i + 1), re.compile(rx.encode("utf-8"))))
        except re.error as e:
            print("bad --forbid pattern %r: %s" % (rx, e), file=sys.stderr)
            return 2

    wheels = []
    for p in args.paths:
        if os.path.isdir(p):
            found = sorted(os.path.join(p, f) for f in os.listdir(p) if f.endswith(".whl"))
            if not found:
                print("no wheels in %s" % p, file=sys.stderr)
                return 2
            wheels += found
        elif os.path.isfile(p):
            wheels.append(p)
        else:
            print("no wheel or folder at %s" % p, file=sys.stderr)
            return 2

    bad = 0
    for w in wheels:
        try:
            bad += check_wheel(w, patterns) > 0
        except (OSError, zipfile.BadZipFile) as e:
            print("%s: cannot read: %s" % (w, e), file=sys.stderr)
            return 2
    print("%d wheels, %d with findings" % (len(wheels), bad))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
