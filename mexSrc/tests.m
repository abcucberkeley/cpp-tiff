addpath('../windows');
im = rand(100,100,100,'single');
im(:,:,2:2:100) = 1;
parallelWriteTiff('test.tif',im);
%% parallelReadTiff in different modes
imT = parallelReadTiff('test.tif');
imT_1_6 = parallelReadTiff('test.tif',[1 6]); %[start end]
imT_1_2_11 = parallelReadTiff('test.tif',[1 2 11]);%[start step end]
imT_2_2_12 = parallelReadTiff('test.tif',[2 2 12]);%[start step end]
%% Visualize tiff read by different modes
figure;
montage(imT_1_6,'Size',[1 6]);
title('1 to 6');
figure;
montage(imT_1_2_11,'Size',[1 6]);
title('1 to 11 at step 2');
figure;
montage(imT_2_2_12,'Size',[1 6]);
title('2 to 12 at step 2');
%%
imSize = getImageSizeMex('test.tif');
