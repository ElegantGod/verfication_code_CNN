clear;clc;

%% step1: 图像预处理部分
% 这里设置统一d图像尺寸
% im_h : 表示图像的高度
% im_w : 表示图像的宽度
im_h = 75;
im_w = 75;

main_dir = '.\images';
subFolder = dir(main_dir);

nClass = length(subFolder)-2;

numTrain = 0;
numTest  = 0;

for ii = 1 : length(subFolder)
    subName = subFolder(ii).name;
    if ~strcmp(subName,'.')&&~strcmp(subName, '..')
        frame = dir(fullfile(main_dir, subName, '*.jpg'));
        numTrain = numTrain + floor(0.5*length(frame));
        numTest  = numTest  + length(frame)-floor(0.5*length(frame));
    end
end


train_x = zeros(im_h, im_w, numTrain);
test_x  = zeros(im_h, im_w, numTest);
train_y = zeros(nClass, numTrain);
test_y  = zeros(nClass, numTest);
hhx = 0;
hhy = 0;

for ii = 1 : length(subFolder)
    subName = subFolder(ii).name;
    a = strcmp(subName, '.');
    if ~strcmp(subName, '.')&&~strcmp(subName, '..')
        frame = dir(fullfile(main_dir, subName, '*.jpg'));
        numImages = length(frame);
        m = randperm(numImages);
        n = floor(0.5*numImages);
        k = 0;
        for jj = 1 : numImages
            name = frame(m(jj)).name;
            imPath = fullfile(main_dir, subName, name);
            im = imread(imPath);
            if ndims(im)==3
                im = im2double(rgb2gray(im));
            else
                im = im2double(im);
            end
            
            im = imresize(im,[im_h, im_w]);
            k = k + 1;
            if k<n+1
                hhx = hhx + 1;
                train_x(:,:,hhx) = im;
                train_y(ii-2,hhx) = 1;
            else
                hhy = hhy + 1;
                test_x(:,:,hhy) = im;
                test_y(ii-2,hhy) = 1;
                
            end
        end
    end
end

%% step2: cnn测试部分
% 这里你需设置cnn测试的参 数opts
% opts.alpha: 表示权重更新的步长（这个参数一般不需要改动）
% opts.batchsize: 每次迭代的小尺寸
% opts.numepoches: 总共迭代的次数
rand('state',0)

cnn.layers = {
    struct('type', 'i') %input layer
    struct('type', 'c', 'outputmaps', 6, 'kernelsize', 6) %convolution layer
    struct('type', 's', 'scale', 2) %sub sampling layer
    struct('type', 'c', 'outputmaps', 12, 'kernelsize', 6) %convolution layer
    struct('type', 's', 'scale', 2) %subsampling layer
};


opts.alpha = 1;
opts.batchsize = 15;
opts.numepochs = 100;

cnn = cnnsetup(cnn, train_x, train_y);
cnn = cnntrain(cnn, train_x, train_y, opts);

[er, bad] = cnntest(cnn, test_x, test_y);

%plot mean squared error
figure; plot(cnn.rL);
assert(er<0.12, 'Too big error');
