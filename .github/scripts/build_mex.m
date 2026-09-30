function build_mex(installDir, outDir)
%BUILD_MEX Compile the cpp-tiff mex files for the GitHub Actions tests.
%   BUILD_MEX(installDir, outDir) builds parallelReadTiff, parallelWriteTiff and
%   getImageSizeMex into outDir, linked against the cpp-tiff library that CMake
%   installed into installDir. The compile_*.m scripts in mexSrc build the
%   release binaries from each Jenkins agent's own paths, so CI uses this.
    inc = fullfile(installDir, 'include');
    lib = fullfile(installDir, 'lib');
    src = fullfile(fileparts(fileparts(fileparts(mfilename('fullpath')))), 'mexSrc');
    if ~exist(outDir, 'dir')
        mkdir(outDir);
    end

    mexes = {'parallelreadtiffmex.cpp',  'parallelReadTiff'
             'parallelwritetiffmex.cpp', 'parallelWriteTiff'
             'getimagesizemex.cpp',      'getImageSizeMex'};
    args = {'-outdir', outDir, ['-I' inc], ['-L' lib], 'CXXOPTIMFLAGS=-O3 -DNDEBUG'};

    if ispc
        % Use the MinGW that built the library. Select it explicitly, since
        % MATLAB would otherwise prefer Visual Studio when both are installed.
        mingw = getenv('MW_MINGW64_LOC');
        gxx = fullfile(mingw, 'bin', 'g++.exe');
        xml = dir(fullfile(matlabroot, 'bin', 'win64', 'mexopts', 'mingw64_g++.xml'));
        assert(~isempty(xml), 'build_mex:noMinGW', 'MATLAB''s MinGW C++ config was not found');
        mex(['-setup:' fullfile(xml(1).folder, xml(1).name)], 'C++');
        [status, gccEh] = system(['"' gxx '" -print-file-name=libgcc_eh.a']);
        assert(status == 0, 'build_mex:noGcc', 'Could not run %s', gxx);
        % -lcppTiff.dll makes the linker pick up the import library libcppTiff.dll.a.
        args = [args, {['CXX=' gxx], strtrim(gccEh), '-lcppTiff.dll'}];
        % The mex files load these at run time. MATLAB looks next to the mex first.
        copyfile(fullfile(installDir, 'bin', 'libcppTiff.dll'), outDir);
        runtime = {'libgomp-1.dll', 'libwinpthread-1.dll', 'libstdc++-6.dll', 'libgcc_s_seh-1.dll'};
        for k = 1:numel(runtime)
            copyfile(fullfile(mingw, 'bin', runtime{k}), outDir);
        end
    elseif ismac
        % The library is built with Homebrew GCC, and the writer passes a
        % std::string across the library boundary, so the mex files need GCC's
        % libstdc++ as well rather than Apple clang's libc++. MATLAB's default
        % flags are clang-only (-stdlib=libc++, -weak-lmx), so replace them.
        % mex adds -l libraries to LINKLIBS, so cpp-tiff goes in the override.
        gxx = getenv('CXX');
        if isempty(gxx)
            gxx = '/opt/homebrew/bin/g++-13';
        end
        [status, sdk] = system('xcrun --show-sdk-path');
        assert(status == 0, 'build_mex:noSdk', 'xcrun could not find the macOS SDK');
        sdk = strtrim(sdk);
        args = [args, {['CXX=' gxx], ...
                       ['CXXFLAGS=-fno-common -fexceptions -std=c++11 -isysroot ' sdk], ...
                       ['LDFLAGS=-bundle -Wl,-twolevel_namespace -isysroot ' sdk ' -Wl,-rpath,' lib], ...
                       ['LINKLIBS=-L"' fullfile(matlabroot, 'bin', 'maca64') '" -lmx -lmex -lmat -L"' lib '" -lcppTiff']}];
    else
        % Static libstdc++ keeps the mex files independent of the older
        % libstdc++ that MATLAB loads itself.
        args = [args, {['LDFLAGS=$LDFLAGS -static-libstdc++ -Wl,-rpath,' lib], '-lcppTiff'}];
    end

    for i = 1:size(mexes, 1)
        mex(args{:}, '-output', mexes{i, 2}, fullfile(src, mexes{i, 1}));
    end
end
