% Builds against the cpp-tiff install in ../jenkinsBuild/install, where the release builds install it (cmake
% --install into jenkinsBuild/install), and is run from this folder (mexSrc): -outdir and the other paths below
% are relative to it
debug = false;
% ADD --disable-new-dtags
if isunix && ~ismac
    releaseFolder = '../linux';
    if ~exist(releaseFolder, 'dir')
        mkdir(releaseFolder);
    end
    if debug
        % (a debug build: the line below with -g -O0 in place of -O3)
    else
        mex -outdir ../linux -output getImageSizeMex.mexa64 -v CXXOPTIMFLAGS="-DNDEBUG -O3" LDOPTIMFLAGS="-Wl',-rpath='''$ORIGIN'''' -O3 -DNDEBUG" CXXFLAGS='$CXXFLAGS -O3' LDFLAGS='$LDFLAGS -O3 -static-libstdc++' -I'../jenkinsBuild/install/include' -L'../jenkinsBuild/install/lib64' -lcppTiff getimagesizemex.cpp
    end

    % The MEX file loads MATLAB's own libstdc++.so.6, as CI's build_mex.m's do: MATLAB's mex links -lstdc++ itself
    % (its LINKLIBS start with it), which takes the shared libstdc++, and -static-libstdc++ only changes the one g++
    % adds, so it takes no effect here. The MEX file needs nothing newer than GLIBCXX_3.4.21 and CXXABI_1.3.9, which
    % MATLAB's has (R2022a to R2024a ship libstdc++ 6.0.28, with up to GLIBCXX_3.4.28; R2024b and R2025b 6.0.30), so
    % the release folder ships libgomp alone with the library
    % The RPATH $ORIGIN alone (a Spack-built GCC adds its own lib64), kept an RPATH: a RUNPATH would let
    % MATLAB's LD_LIBRARY_PATH win over the libraries next to the MEX file
    system(['patchelf --force-rpath --set-rpath ''$ORIGIN'' ' releaseFolder '/getImageSizeMex.mexa64']);
elseif ismac
    % May need to fix /Users/username/Library/Application/Support/MathWorks/MATLAB/R2024b/mex_C++_mac*64.xml
    % in Matlab R2024b and newer for mex compilation to work with g++
    if computer == "MACI64"
        releaseFolder = '../mac';
        if ~exist(releaseFolder, 'dir')
            mkdir(releaseFolder);
        end 
        % Might have to do this part in terminal. First change the library
        % linked to libstdc++
    
        % Can make a custom libtiff dylib where the libaries are linked by
        % loader path
           
        mex -v -outdir ../mac -output getImageSizeMex.mexmaci64 CXX="/usr/local/bin/g++-13" CXXOPTIMFLAGS='-O3 -DNDEBUG' LDOPTIMFLAGS='-O3 -DNDEBUG' CXXFLAGS='-fno-common -arch x86_64 -mmacosx-version-min=10.15 -fexceptions -isysroot /Library/Developer/CommandLineTools/SDKs/MacOSX.sdk -std=c++11 -O3 -DMATLAB_DEFAULT_RELEASE=R2017b  -DUSE_MEX_CMD   -DMATLAB_MEX_FILE' LDFLAGS='$LDFLAGS -O3' '-I../jenkinsBuild/install/include' -L'../jenkinsBuild/install/lib' /usr/local/opt/gcc/lib/gcc/current/libstdc++.a -lcppTiff getimagesizemex.cpp
    
        % We need to change all the current paths to be relative to the mex file
        %system('install_name_tool -change /usr/local/opt/gcc/lib/gcc/current/libstdc++.6.dylib @loader_path/libstdc++.6.0.32.dylib ../mac/getImageSizeMex.mexmaci64');
        system('install_name_tool -change /usr/local/opt/gcc/lib/gcc/current/libgomp.1.dylib @loader_path/libgomp.1.dylib ../mac/getImageSizeMex.mexmaci64');
        system('install_name_tool -change /usr/local/opt/gcc/lib/gcc/current/libgcc_s.1.1.dylib @loader_path/libgcc_s.1.1.0.dylib ../mac/getImageSizeMex.mexmaci64');
        system('install_name_tool -change @rpath/libcppTiff.dylib @loader_path/libcppTiff.dylib ../mac/getImageSizeMex.mexmaci64');
    
        system('chmod 777 ../mac/getImageSizeMex.mexmaci64');
    else
        releaseFolder = '../macArm';
        if ~exist(releaseFolder, 'dir')
            mkdir(releaseFolder);
        end 
        mex -v -outdir ../macArm -output getImageSizeMex.mexmaca64 CXX="/opt/homebrew/bin/g++-13" CXXOPTIMFLAGS='-O3 -DNDEBUG' LDOPTIMFLAGS='-O3 -DNDEBUG' CXXFLAGS='-fno-common -arch x86_64 -mmacosx-version-min=10.15 -fexceptions -isysroot /Library/Developer/CommandLineTools/SDKs/MacOSX.sdk -std=c++11 -O3 -DMATLAB_DEFAULT_RELEASE=R2017b  -DUSE_MEX_CMD   -DMATLAB_MEX_FILE' LDFLAGS='$LDFLAGS -O3' '-I../jenkinsBuild/install/include' -L'../jenkinsBuild/install/lib' /opt/homebrew/opt/gcc@13/lib/gcc/13/libstdc++.a -lcppTiff getimagesizemex.cpp
    
        % We need to change all the current paths to be relative to the mex file
        %system('install_name_tool -change /usr/local/opt/gcc/lib/gcc/current/libstdc++.6.dylib @loader_path/libstdc++.6.0.32.dylib ../mac/getImageSizeMex.mexmaci64');
        %system('install_name_tool -change /opt/homebrew/opt/gcc@13/lib/gcc/13/libgomp.1.dylib @loader_path/libgomp.1.dylib ../macArm/getImageSizeMex.mexmaca64');
        %system('install_name_tool -change /usr/local/opt/gcc/lib/gcc/current/libgcc_s.1.1.dylib @loader_path/libgcc_s.1.1.0.dylib ../mac/getImageSizeMex.mexmaci64');
        system('install_name_tool -change @rpath/libcppTiff.dylib @loader_path/libcppTiff.dylib ../macArm/getImageSizeMex.mexmaca64');
    
        system('chmod 777 ../macArm/getImageSizeMex.mexmaca64');
    end
elseif ispc
    setenv('MW_MINGW64_LOC','C:/mingw64');
    releaseFolder = '../windows';
    if ~exist(releaseFolder, 'dir')
        mkdir(releaseFolder);
    end
    mex  -outdir ../windows -output getImageSizeMex.mexw64 -v CXX="C:/mingw64/bin/g++" CXXOPTIMFLAGS="-DNDEBUG -O3" LDOPTIMFLAGS="-Wl',-rpath='''$ORIGIN'''' -O3 -DNDEBUG" CXXFLAGS='$CXXFLAGS -O3' LDFLAGS='$LDFLAGS -O3' -I'../jenkinsBuild/install/include' -L'../jenkinsBuild/install/lib' C:\mingw64\lib\gcc\x86_64-w64-mingw32\12.2.0\libgcc_eh.a -lcppTiff.dll getimagesizemex.cpp
end
