% Builds against the cpp-tiff install in ../jenkinsBuild/install, where the release builds install it (cmake
% --install into jenkinsBuild/install), and is run from this folder (mexSrc): -outdir and the other paths below
% are relative to it.
% parallelReadTiffGPU needs the Parallel Computing Toolbox (its GPU MEX API: gpu/mxGPUArray.h and the mwgpu
% library) and no CUDA toolkit: cpp-tiff loads the CUDA driver and nvCOMP when it first reads on the GPU. Linux
% only for now.
gpuInclude = ['-I' fullfile(matlabroot, 'toolbox', 'parallel', 'gpu', 'extern', 'include')];
if isunix && ~ismac
    releaseFolder = '../linux';
    if ~exist(releaseFolder, 'dir')
        mkdir(releaseFolder);
    end
    mex('-outdir', releaseFolder, '-output', 'parallelReadTiffGPU.mexa64', '-v', 'CXXOPTIMFLAGS=-DNDEBUG -O3', ...
        'LDOPTIMFLAGS=-Wl,-rpath,''$ORIGIN'' -O3 -DNDEBUG', 'CXXFLAGS=$CXXFLAGS -O3', 'LDFLAGS=$LDFLAGS -O3', ...
        '-I../jenkinsBuild/install/include', gpuInclude, '-L../jenkinsBuild/install/lib64', '-lstdc++', '-lcppTiff', '-lmwgpu', ...
        'parallelreadtiffgpumex.cpp');
    % (as for parallelReadTiff: MATLAB preloads its own libstdc++, and the RPATH is $ORIGIN alone, kept an RPATH)
    system(['patchelf --replace-needed libstdc++.so.6 libstdc++.so.6.0.32 ' releaseFolder '/parallelReadTiffGPU.mexa64']);
    system(['patchelf --force-rpath --set-rpath ''$ORIGIN'' ' releaseFolder '/parallelReadTiffGPU.mexa64']);
else
    % (Windows: not built yet; macOS: MATLAB has no GPU arrays there)
    error('parallelReadTiffGPU is built for Linux only');
end
