addpath('../windows');
im = rand(100,100,100,'single');
im(:,:,2:2:100) = 1;
parallelWriteTiff('test.tif',im);
%% parallelReadTiff in different modes
imT = parallelReadTiff('test.tif');
imT_1_5 = parallelReadTiff('test.tif',[1 6]); %[start end]
imT_1_2_9 = parallelReadTiff('test.tif',[1 2 11]);%[start step end]
imT_2_2_10 = parallelReadTiff('test.tif',[2 2 12]);%[start step end]
%%
figure;
montage(imT_1_5,'Size',[1 6]);
title('1 to 5');
figure;
montage(imT_1_2_9,'Size',[1 6]);
title('1 to 9 at step 2');
figure;
montage(imT_2_2_10,'Size',[1 6]);
title('2 to 10 at step 2');
%%
imSize = getImageSizeMex('test.tif');