function test_matlab(mexDir)
% Smoke test for the compiled MATLAB mex files: write small random volumes,
% read them back, and check the data survives the round trip. Also checks a
% z-range read and getImageSizeMex. Errors (non-zero exit under `matlab -batch`)
% on any failure so Jenkins fails the build.
%
% With no argument it auto-detects the platform's mex folder (matching the
% compile_*.m scripts), so Jenkins can call it quote-free right after compiling:
%   matlab -batch "compile_getImageSizeMex;compile_parallelReadTiff;compile_parallelWriteTiff;addpath ../tests;test_matlab;exit"
% Or pass an explicit folder: test_matlab('../linux')
    if nargin < 1
        if ispc
            mexDir = '../windows';
        elseif ismac
            if strcmp(computer, 'MACI64')
                mexDir = '../mac';
            else
                mexDir = '../macArm';
            end
        else
            mexDir = '../linux';
        end
    end
    addpath(mexDir);

    tmp = tempname;
    mkdir(tmp);
    cleanup = onCleanup(@() rmdir(tmp, 's')); %#ok<NASGU>

    rng(1234567);
    sz = [24 40 3];   % rows(y), cols(x), slices(z)
    types = {'uint8','uint16','int16','int32','single','double'};

    for k = 1:numel(types)
        t = types{k};
        if any(strcmp(t, {'single','double'}))
            data = cast((rand(sz) - 0.5) * 1000, t);
        elseif t(1) == 'u'
            data = cast(randi([0 200], sz), t);
        else
            data = cast(randi([-100 100], sz), t);   % negatives exercise signed types
        end

        f = fullfile(tmp, ['rt_' t '.tif']);
        parallelWriteTiff(f, data);

        back = parallelReadTiff(f);
        assert(isa(back, t),            'dtype mismatch for %s: got %s', t, class(back));
        assert(isequal(size(back), sz), 'size mismatch for %s', t);
        assert(isequal(back, data),     'round-trip data mismatch for %s', t);

        % z-range read (1-indexed, inclusive): first two slices.
        sub = parallelReadTiff(f, [1 2]);
        assert(isequal(sub, data(:,:,1:2)), 'range read mismatch for %s', t);

        dims = getImageSizeMex(f);
        assert(isequal(double(dims(:).'), sz), 'getImageSizeMex mismatch for %s', t);

        fprintf('PASS  %s\n', t);
    end

    disp('MATLAB round-trip tests PASSED');
end
