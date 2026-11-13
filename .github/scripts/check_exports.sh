#!/bin/bash
# The installed library exports none of its static C++ runtime (CMakeLists.txt's CPPTIFF_PRIVATE_RUNTIME), counted
# by demangled name, and still exports the libtiff and zstd APIs, whose headers are installed, and cpp-tiff's own. A
# Python module given second exports no C++ symbol at all (macOS, where setup.py links it with its entry point alone
# exported). Linux and macOS.
# usage: check_exports.sh <installed library> [<Python module>]
set -u
lib=$1; mod=${2:-}
# The names as Linux gives them (a Mach-O name has one more _ in front, taken off here), demangled by c++filt -n,
# which takes them as given: Apple's c++filt by default takes a leading _ off itself, and would leave every C++ name
# here mangled, so that no runtime name could be found
case "$(uname -s)" in
    Darwin) exports(){ nm -gU "$1" | awk '{print $NF}' | sed 's/^_//'; } ;;
    *)      exports(){ nm -D --defined-only "$1" | awk '{print $NF}' | sed 's/@.*//'; } ;;
esac
demangle(){ c++filt -n; }
builtin='(unsigned |signed )?(char|short|int|long|long long|bool|float|double|long double|void|wchar_t|char8_t|char16_t|char32_t|decltype\(nullptr\)|__int128|__float128)( const)?\*?'
runtime_re="^(std::|__gnu_cxx::|__cxxabiv1::|operator new|operator delete|(typeinfo|typeinfo name|vtable|VTT|construction vtable|guard variable|TLS init function|TLS wrapper function) for (std::|__gnu_cxx::|__cxxabiv1::|$builtin\$)|(non-)?virtual thunk to std::|covariant return thunk to std::|__cxa_|__gxx_personality|__dynamic_cast\$)"
# The check fails, rather than passing, if c++filt does not demangle here or the pattern misses the runtime's names
[ "$(echo _ZNSt9exceptionD1Ev | demangle)" = 'std::exception::~exception()' ] ||
    { echo "c++filt -n does not demangle _ZNSt9exceptionD1Ev here: $(echo _ZNSt9exceptionD1Ev | demangle)"; exit 1; }
for probe in 'std::exception::~exception()' 'virtual thunk to std::basic_ostream<char, std::char_traits<char> >::~basic_ostream()' \
             'typeinfo for char const*' '__cxa_throw'; do
    printf '%s\n' "$probe" | grep -qE "$runtime_re" || { echo "the runtime pattern misses $probe"; exit 1; }
done
[ -f "$lib" ] || { echo "no library at $lib"; exit 1; }
bad=0
all=$(exports "$lib")
names=$(printf '%s\n' "$all" | demangle)
runtime=$(printf '%s\n' "$names" | grep -E "$runtime_re")
n=$(printf '%s' "$runtime" | grep -c . || true)
echo "$lib: $(printf '%s\n' "$all" | grep -c .) exports, $n of the C++ runtime"
if [ "$n" -gt 0 ]; then printf '%s\n' "$runtime" | head -10; bad=1; fi
for api in TIFFOpen TIFFReadDirectory ZSTD_compress ZSTD_decompress; do
    printf '%s\n' "$names" | grep -qx "$api" || { echo "$lib does not export $api"; bad=1; }
done
# (cpp-tiff's own functions, found by their demangled names)
for api in 'getTiffInfo(' 'readTiffParallel(' 'writeTiffParallelWrapper('; do
    printf '%s\n' "$names" | grep -q "^$api" || { echo "$lib does not export $api...)"; bad=1; }
done
if [ -n "$mod" ]; then
    [ -f "$mod" ] || { echo "no module at $mod"; exit 1; }
    cxx=$(exports "$mod" | grep -c '^_Z' || true)
    echo "$mod: $(exports "$mod" | grep -c .) exports, $cxx of C++"
    if [ "$cxx" -gt 0 ]; then exports "$mod" | grep '^_Z' | demangle | head -10; bad=1; fi
fi
exit $bad
