function test_matlab(mexDir)
% Smoke test for the compiled MATLAB mex files: write small volumes, read them
% back, and check the data survives the round trip -- across every supported
% class, each compression, 2D and 3D, striped and tiled, grayscale, RGB and
% RGBA, plus z-range reads, planar color files and getImageSizeMex. Errors
% (non-zero exit under `matlab -batch`) on any failure so Jenkins fails the build.
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

    % 5. ImageJ-format files (single IFD + contiguous slices): RGB sample
    %    scaling, z-range start offset, and byte-order handling in the ImageJ
    %    fast path. Real ImageJ writes big-endian; tifffile writes little.
    ijcases = { ...
        'ij_rgb',    makeData([24 40 3 5], 'uint8'),  'l'; ...
        'ij_g16_le', makeData([24 40 4],   'uint16'), 'l'; ...
        'ij_g16_be', makeData([24 40 4],   'uint16'), 'b'};
    for ci = 1:size(ijcases, 1)
        [name, arr, bo] = ijcases{ci, :};
        f = fullfile(tmp, [name '.tif']);
        writeImageJTiff(f, arr, bo);

        im = parallelReadTiff(f);
        assert(strcmp(class(im), class(arr)), 'imagej %s: class %s', name, class(im));
        assert(isequal(size(im), size(arr)),  'imagej %s: size %s', name, mat2str(size(im)));
        assert(isequal(im, arr),              'imagej %s: data mismatch', name);

        sub = parallelReadTiff(f, [2 3]);     % 1-indexed, inclusive
        dims = getImageSizeMex(f);
        if ndims(arr) == 4
            assert(isequal(sub, arr(:,:,:,2:3)), 'imagej %s: z-range', name);
            assert(isequal(double(dims), [24 40 5 3]), 'imagej %s: getImageSizeMex %s', name, mat2str(dims));
        else
            assert(isequal(sub, arr(:,:,2:3)), 'imagej %s: z-range', name);
            assert(isequal(double(dims), [24 40 4]), 'imagej %s: getImageSizeMex %s', name, mat2str(dims));
        end
        delete(f);
        fprintf('PASS  imagej %s\n', name);
    end

    % 6. RGB and RGBA writes: 4D (y, x, c, z) data is color, and so is 3D
    %    (y, x, c) data when the sixth argument is true. They read back as
    %    written, getImageSizeMex reports [y x z c], and MATLAB's own reader
    %    sees an RGB image with straight (unassociated) alpha for RGBA.
    rgbTypes = {'uint8', 'uint16', 'single'};
    for k = 1:numel(rgbTypes)
        t = rgbTypes{k};
        for spp = [3 4]
            for ci = 1:numel(comps)
                for tile = tiles
                    for nz = [2 1]
                        data = makeData([24 40 spp nz], t);
                        tag = sprintf('rgb %s/spp%d/%s/tile%d/z%d', t, spp, comps{ci}, tile, nz);
                        f = fullfile(tmp, sprintf('rgb_%s_%d_%s_%d_%d.tif', t, spp, comps{ci}, tile, nz));
                        if nz > 1
                            parallelWriteTiff(f, data, 'w', comps{ci}, tile);
                        else
                            parallelWriteTiff(f, data, 'w', comps{ci}, tile, true);
                        end
                        back = parallelReadTiff(f);
                        assert(isa(back, t) && isequal(back, data), '%s: read back', tag);
                        dims = getImageSizeMex(f);
                        assert(isequal(double(dims), [24 40 nz spp]), '%s: getImageSizeMex %s', tag, mat2str(dims));
                        info = imfinfo(f);
                        assert(strcmp(info(1).PhotometricInterpretation, 'RGB') && info(1).SamplesPerPixel == spp ...
                               && strcmp(info(1).PlanarConfiguration, 'Chunky'), '%s: imfinfo', tag);
                        if spp == 4
                            assert(isequal(info(1).ExtraSamples, 2), '%s: ExtraSamples %s', tag, mat2str(info(1).ExtraSamples));
                        end
                        % MATLAB's reader has no zstd.
                        if strcmp(t, 'uint8') && ~strcmp(comps{ci}, 'zstd')
                            assert(isequal(imread(f, nz), data(:,:,:,nz)), '%s: imread', tag);
                        end
                        delete(f);
                        nchecked = nchecked + 1;
                    end
                end
            end
        end
        fprintf('PASS  rgb %s\n', t);
    end
    % Without the sixth argument, 3D data is still a grayscale stack.
    f = fullfile(tmp, 'gray3.tif');
    g = makeData([24 40 3], 'uint16');
    parallelWriteTiff(f, g);
    assert(isequal(parallelReadTiff(f), g) && isequal(double(getImageSizeMex(f)), [24 40 3]), ...
           'a 3-slice stack was not written as grayscale');
    delete(f);
    % Color data other than RGB or RGBA is refused.
    f = fullfile(tmp, 'bad_rgb.tif');
    assert(throws(@() parallelWriteTiff(f, makeData([8 8 2 2], 'uint8'))), '4D data with 2 samples was accepted');
    assert(throws(@() parallelWriteTiff(f, makeData([8 8 5], 'uint8'), 'w', 'lzw', 0, true)), '5 samples were written as color');
    assert(throws(@() parallelWriteTiff(f, makeData([8 8], 'uint8'), 'w', 'lzw', 0, true)), '2D data was written as color');

    % 7. Planar color (PlanarConfiguration Separate): each color stored as its
    %    own image. MATLAB's Tiff class writes the fixtures, as strips and
    %    tiles, 2D and 3D; they read like chunky color, (y, x, c, z).
    pcases = { ...
        'uint8',  3, 'LZW',          0, 3; ...
        'uint16', 4, 'None',         0, 2; ...
        'single', 3, 'AdobeDeflate', 16, 3; ...
        'uint16', 3, 'PackBits',     0, 1};
    for ci = 1:size(pcases, 1)
        [t, spp, comp, tile, nz] = pcases{ci, :};
        data = makeData([24 40 spp nz], t);
        tag = sprintf('planar %s/spp%d/%s/tile%d/z%d', t, spp, comp, tile, nz);
        f = fullfile(tmp, sprintf('planar_%d.tif', ci));
        writePlanarTiff(f, data, comp, tile);
        info = imfinfo(f);
        assert(strcmp(info(1).PlanarConfiguration, 'Planar'), '%s: fixture is not planar', tag);
        assert(isequal(parallelReadTiff(f), data), '%s: full read', tag);
        if nz > 2
            assert(isequal(parallelReadTiff(f, [2 3]), data(:,:,:,2:3)), '%s: z-range', tag);
        end
        assert(isequal(double(getImageSizeMex(f)), [24 40 nz spp]), '%s: getImageSizeMex', tag);
        delete(f);
    end
    fprintf('PASS  planar\n');

    % 8. Unreadable and damaged files raise MATLAB errors instead of crashing
    %    MATLAB.
    missing = fullfile(tmp, 'missing.tif');
    assert(throws(@() parallelReadTiff(missing)), 'missing file: parallelReadTiff did not error');
    assert(throws(@() getImageSizeMex(missing)), 'missing file: getImageSizeMex did not error');

    % Damage one LZW strip of slice 3. Reading it must error, as a full read and
    % as a single slice, while the undamaged slices still read correctly.
    rng(97531);
    vol = uint16(randi([0 65535], [48 64 4]));
    f = fullfile(tmp, 'corrupt.tif');
    parallelWriteTiff(f, vol, 'w', 'lzw');
    info = imfinfo(f);
    off = double(info(3).StripOffsets(1));
    cnt = double(info(3).StripByteCounts(1));
    fid = fopen(f, 'r+');
    fseek(fid, off + floor(cnt/4), 'bof');
    fwrite(fid, repmat(uint8(255), 1, min(256, floor(cnt/2))), 'uint8');
    fclose(fid);
    assert(throws(@() parallelReadTiff(f)), 'corrupt strip: full read did not error');
    assert(throws(@() parallelReadTiff(f, [3 3])), 'corrupt strip: damaged slice did not error');
    assert(isequal(parallelReadTiff(f, [1 2]), vol(:,:,1:2)), 'corrupt strip: undamaged slices');
    delete(f);

    % Writes that cannot succeed must error instead of reporting success.
    blocker = fullfile(tmp, 'not_a_folder.txt');
    fid = fopen(blocker, 'w');
    fwrite(fid, 'x');
    fclose(fid);
    assert(throws(@() parallelWriteTiff(fullfile(blocker, 'sub', 'out.tif'), vol)), 'write below a regular file did not error');
    delete(blocker);
    if isunix && ~ismac
        assert(throws(@() parallelWriteTiff('/dev/full', vol, 'w', 'none')), 'write to /dev/full did not error');
    end
    fprintf('PASS  damaged files\n');

    fprintf('MATLAB tests PASSED (%d round trips)\n', nchecked);
end


function writeImageJTiff(f, arr, byteorder)
%WRITEIMAGEJTIFF Minimal single-IFD ImageJ-format TIFF (test fixture).
%   One IFD describes slice 1; every slice's raw data sits contiguous after it
%   and the ImageJ description carries images=Z -- the layout ImageJ's fast
%   save produces, which the library reads through its ImageJ path. The
%   library's writer cannot produce this layout, so the test writes it by
%   hand, mirroring the fixtures in tests/roundtrip.cpp and
%   python/tests/test_python.py. arr is (y,x,z) or (y,x,c,z); byteorder is
%   'l' (little endian, like tifffile) or 'b' (big endian, like ImageJ).
    if ndims(arr) == 4
        [y, x, spp, z] = size(arr); perm = [3 2 1 4];   % file order: c, x, y, z
    else
        [y, x, z] = size(arr); spp = 1; perm = [2 1 3]; % file order: x, y, z
    end
    bits = 8 * numel(typecast(cast(0, class(arr)), 'uint8'));
    es = bits / 8;
    desc = [sprintf('ImageJ=1.11a\nimages=%d\nslices=%d\n', z, z) char(0)];
    if mod(numel(desc), 2), desc(end+1) = char(0); end

    n = 11;
    auxOff = 8 + 2 + n*12 + 4;
    bpsLen = (spp > 1) * spp * 2;
    descOff = auxOff + bpsLen;
    pixOff = descOff + numel(desc);
    pixOff = pixOff + mod(pixOff, 2);

    if spp == 1, bpsVal = bits; bpsLong = false;
    else,        bpsVal = auxOff; bpsLong = true;  end
    photometric = 1; if spp == 3, photometric = 2; end
    % tag, type (2=ASCII, 3=SHORT, 4=LONG), count, value, value-is-LONG
    ents = { ...
        256, 3, 1,           x,             false; ...  % ImageWidth
        257, 3, 1,           y,             false; ...  % ImageLength
        258, 3, spp,         bpsVal,        bpsLong; ...% BitsPerSample
        259, 3, 1,           1,             false; ...  % Compression: none
        262, 3, 1,           photometric,   false; ...  % Photometric
        270, 2, numel(desc), descOff,       true;  ...  % ImageDescription
        273, 4, 1,           pixOff,        true;  ...  % StripOffsets
        277, 3, 1,           spp,           false; ...  % SamplesPerPixel
        278, 3, 1,           y,             false; ...  % RowsPerStrip
        279, 4, 1,           x*y*spp*es,    true;  ...  % StripByteCounts
        284, 3, 1,           1,             false};     % PlanarConfig: chunky
    fid = fopen(f, 'w', byteorder);
    assert(fid > 0, 'writeImageJTiff: cannot open %s', f);
    if byteorder == 'b', fwrite(fid, 'MM', 'uint8'); else, fwrite(fid, 'II', 'uint8'); end
    fwrite(fid, 42, 'uint16'); fwrite(fid, 8, 'uint32');
    fwrite(fid, n, 'uint16');
    for i = 1:size(ents, 1)
        fwrite(fid, [ents{i,1} ents{i,2}], 'uint16');
        fwrite(fid, ents{i,3}, 'uint32');
        if ents{i,5}
            fwrite(fid, ents{i,4}, 'uint32');
        else
            fwrite(fid, [ents{i,4} 0], 'uint16');   % SHORT sits in the first 2 bytes
        end
    end
    fwrite(fid, 0, 'uint32');                       % no next IFD
    if spp > 1, fwrite(fid, repmat(bits, 1, spp), 'uint16'); end
    fwrite(fid, uint8(desc), 'uint8');
    fwrite(fid, zeros(1, pixOff - ftell(fid), 'uint8'), 'uint8');
    v = permute(arr, perm);
    fwrite(fid, v(:), class(arr));
    fclose(fid);
end


function writePlanarTiff(f, data, compression, tile)
%WRITEPLANARTIFF Planar (separate color planes) RGB/RGBA fixture written with
%   MATLAB's Tiff class: every strip or tile of the first color, then of the
%   next. data is (y, x, c, z); compression is a Tiff.Compression name and
%   tile 0 for 5-row strips. The library's writer only writes chunky color.
    [y, x, spp, z] = size(data);
    t = Tiff(f, 'w');
    closer = onCleanup(@() t.close()); %#ok<NASGU>
    for k = 1:z
        tags = struct();
        tags.ImageLength = y;
        tags.ImageWidth = x;
        tags.Photometric = Tiff.Photometric.RGB;
        tags.BitsPerSample = 8 * numel(typecast(cast(0, class(data)), 'uint8'));
        tags.SamplesPerPixel = spp;
        if isfloat(data)
            tags.SampleFormat = Tiff.SampleFormat.IEEEFP;
        else
            tags.SampleFormat = Tiff.SampleFormat.UInt;
        end
        tags.PlanarConfiguration = Tiff.PlanarConfiguration.Separate;
        tags.Compression = Tiff.Compression.(compression);
        if spp == 4
            tags.ExtraSamples = Tiff.ExtraSamples.UnassociatedAlpha;
        end
        if tile
            tags.TileWidth = tile;
            tags.TileLength = tile;
        else
            tags.RowsPerStrip = 5;
        end
        t.setTag(tags);
        t.write(data(:, :, :, k));
        if k < z
            t.writeDirectory();
        end
    end
end


function tf = throws(fn)
%THROWS True if calling fn raises an error.
    tf = false;
    try
        fn();
    catch
        tf = true;
    end
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
%   Reads the header directly instead of using the Tiff class: Tiff holds the
%   file open past the call, and Windows then refuses to delete it. Only
%   little-endian files are decoded (every platform cpp-tiff targets is little
%   endian); anything else returns -1 so the caller skips the check.
    tw = -1;
    fid = fopen(f, 'r', 'l');
    if fid < 0
        return;
    end
    try
        tw = readTileWidth(fid);
    catch
        tw = -1;
    end
    fclose(fid);
end


function tw = readTileWidth(fid)
    tw = -1;
    if ~strcmp(fread(fid, 2, '*char')', 'II')
        return;                                  % not little-endian
    end
    magic = fread(fid, 1, 'uint16');
    if magic == 42                               % classic: 12-byte entries
        off = fread(fid, 1, 'uint32');
        if fseek(fid, off, 'bof') ~= 0, return; end
        n = fread(fid, 1, 'uint16'); esz = 12; voff = 9;
    elseif magic == 43                           % BigTIFF: 20-byte entries
        fread(fid, 4, 'uint8');                  % offset size + reserved
        off = fread(fid, 1, 'uint64');
        if fseek(fid, off, 'bof') ~= 0, return; end
        n = fread(fid, 1, 'uint64'); esz = 20; voff = 13;
    else
        return;
    end
    if isempty(n) || n > 4096
        return;
    end
    tw = 0;                                      % no tile tag => striped
    for i = 1:n
        e = fread(fid, esz, '*uint8')';
        if numel(e) < esz
            tw = -1; return;
        end
        if double(e(1)) + 256*double(e(2)) == 322      % TIFFTAG_TILEWIDTH
            tw = double(e(voff)) + 256*double(e(voff+1));
            return;
        end
    end
end
