function test_matlab(mexDir)
% Smoke test for the compiled MATLAB mex files: write small volumes, read them
% back, and check the data survives the round trip -- across every supported
% class, each compression, 2D and 3D, striped and tiled, plus z-range reads and
% getImageSizeMex. Errors (non-zero exit under `matlab -batch`) on any failure so
% Jenkins fails the build.
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
    cleanup = onCleanup(@() rmdirQuiet(tmp)); %#ok<NASGU>

    rng(1234567);
    types = {'uint8','int8','uint16','int16','uint32','int32','single', ...
             'uint64','int64','double'};
    comps = {'none','lzw','zstd'};
    tiles = [0 16];   % 0 = striped. 16 tiles a 24x40 image 2x3, ragged both ways.
    nchecked = 0;

    % 1. Every class x compression x {striped, tiled} x {2D, 3D}.
    for k = 1:numel(types)
        t = types{k};
        for ci = 1:numel(comps)
            for tile = tiles
                for is3d = [true false]
                    if is3d, sz = [24 40 3]; else, sz = [24 40]; end
                    data = makeData(sz, t);
                    tag = sprintf('%s/%s/tile%d/%dD', t, comps{ci}, tile, numel(sz));

                    f = fullfile(tmp, sprintf('rt_%s_%s_%d_%d.tif', t, comps{ci}, tile, numel(sz)));
                    parallelWriteTiff(f, data, 'w', comps{ci}, tile);

                    back = parallelReadTiff(f);
                    assert(isa(back, t),            '%s: class %s', tag, class(back));
                    assert(isequal(size(back), sz), '%s: size mismatch', tag);
                    assert(isequal(back, data),     '%s: data mismatch', tag);

                    % A garbage z (reading past the dimension array for 2D input)
                    % blows the file up, so guard the size as well as the data.
                    raw = prod(sz) * numel(typecast(cast(0, t), 'uint8'));
                    d = dir(f);
                    assert(d.bytes < raw*10 + 100000, ...
                           '%s: file is %d bytes for %d bytes of data', tag, d.bytes, raw);

                    tw = tileWidthOf(f);
                    if tw >= 0
                        assert(tw == tile, '%s: TileWidth %d, expected %d', tag, tw, tile);
                    end

                    dims = getImageSizeMex(f);
                    expect = [sz 1];              % getImageSizeMex always reports z
                    assert(isequal(double(dims(1:3)), expect(1:3)), '%s: getImageSizeMex', tag);

                    delete(f);
                    nchecked = nchecked + 1;
                end
            end
        end
        fprintf('PASS  %s\n', t);
    end

    % 2. z-range reads (1-indexed, inclusive), striped and tiled.
    vol = makeData([40 50 6], 'uint16');
    for tile = tiles
        f = fullfile(tmp, sprintf('zr_%d.tif', tile));
        parallelWriteTiff(f, vol, 'w', 'lzw', tile);
        assert(isequal(parallelReadTiff(f), vol), 'zrange tile%d: full read', tile);
        ranges = {[1 2], [3 5], [5 6], [4 4]};
        for r = 1:numel(ranges)
            a = ranges{r}(1); b = ranges{r}(2);
            sub = parallelReadTiff(f, [a b]);
            assert(isequal(sub, vol(:,:,a:b)), 'zrange tile%d: [%d %d]', tile, a, b);
        end
        delete(f);
    end

    % 3. A ragged tiled volume, so the padded edge tiles are exercised.
    ragged = makeData([70 90 2], 'uint16');
    for tile = [16 64]
        f = fullfile(tmp, sprintf('ragged_%d.tif', tile));
        parallelWriteTiff(f, ragged, 'w', 'zstd', tile);
        tw = tileWidthOf(f);
        if tw >= 0
            assert(tw == tile, 'ragged tile%d: TileWidth %d', tile, tw);
        end
        assert(isequal(parallelReadTiff(f), ragged), 'ragged tile%d: data mismatch', tile);
        delete(f);
    end

    % 4. An invalid tile size must be rejected rather than silently accepted.
    threw = false;
    try
        parallelWriteTiff(fullfile(tmp, 'bad.tif'), makeData([32 32], 'uint16'), 'w', 'lzw', 100);
    catch
        threw = true;
    end
    assert(threw, 'a tile size that is not a multiple of 16 was accepted');

    fprintf('MATLAB tests PASSED (%d round trips)\n', nchecked);
end


function data = makeData(sz, t)
%MAKEDATA Deterministic test data for class t, using the type's extremes.
    n = prod(sz);
    if any(strcmp(t, {'single','double'}))
        data = cast((rand(sz) - 0.5) * 1000, t);
        return;
    end
    lim = double(intmax(t));
    data = cast(reshape(mod(0:n-1, min(lim, 4000)), sz), t);
    data(1) = intmin(t);      % extremes catch sign/width mistakes
    data(2) = intmax(t);
end


function rmdirQuiet(d)
%RMDIRQUIET Best-effort teardown; cleanup trouble must not mask the result.
    try
        rmdir(d, 's');
    catch
    end
end


function tw = tileWidthOf(f)
%TILEWIDTHOF TileWidth of the first directory, 0 if striped, -1 if unreadable.
%   Returns -1 when the file cannot be inspected (e.g. a MATLAB release whose
%   Tiff class cannot open BigTIFF) so the caller skips the check instead of
%   failing on something unrelated to the data.
    tw = -1;
    try
        tf = Tiff(f, 'r');
    catch
        return;
    end
    try
        tw = double(getTag(tf, 'TileWidth'));
    catch
        tw = 0;   % tag absent => striped
    end
    % Close explicitly rather than leaving it to onCleanup/destructor ordering:
    % Windows refuses to delete a file that still has an open handle.
    close(tf);
end
