function unload_probe(mexDir, logFn, clearFirst)
% Windows, CI only (branch win-runtime-unload): writes and reads a stack with the mex files in mexDir, clears them if
% clearFirst, and returns so that matlab -batch exits. With PROBE_CDB set, it first attaches cdb to this MATLAB, which
% writes to logFn the stack of a second-chance access violation, if one happens while MATLAB exits, and the modules
% loaded and unloaded then.
addpath(mexDir, '-begin');
im = uint16(randi(60000, 256, 256, 64));
fn = [tempname, '.tif'];
parallelWriteTiff(fn, im);
back = parallelReadTiff(fn);
sz = getImageSizeMex(fn);
fprintf('write and read: equal %d, size %s, from %s\n', isequal(back, im), mat2str(sz), which('parallelWriteTiff'));
delete(fn);
if clearFirst
    clear mex
    fprintf('cleared the mex files\n');
end
cdb = getenv('PROBE_CDB');
if isempty(cdb)
    return;
end
cmdFn = [tempname, '.txt'];
writelines(["sxd av", ...
    "sxd -c2 "".echo == second-chance access violation; .ecxr; kb 40; .echo == modules; lm; .echo == end; q"" av", ...
    ".echo == attached", "g"], cmdFn);
system(sprintf(['powershell -NoProfile -NonInteractive -Command "Start-Process -FilePath ''%s'' -WindowStyle Hidden ', ...
    '-ArgumentList ''-p'',''%d'',''-G'',''-cf'',''%s'',''-logo'',''%s''"'], cdb, feature('getpid'), cmdFn, logFn));
% (the debugger runs the command file once it has attached)
t = tic;
attached = false;
while ~attached && toc(t) < 60
    pause(0.5);
    try
        attached = contains(fileread(logFn), '== attached');
    catch
    end
end
pause(2);
fprintf('cdb attached: %d after %.1f s\n', attached, toc(t));
end
