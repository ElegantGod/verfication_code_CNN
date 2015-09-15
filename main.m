clear;clc;

%% step1: ͼ��Ԥ������
% ��������ͳһdͼ��ߴ�
% im_h : ��ʾͼ��ĸ߶�
% im_w : ��ʾͼ��Ŀ��
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

%% step2: cnn���Բ���
% ������������cnn���ԵĲ� ��opts
% opts.alpha: ��ʾȨ�ظ��µĲ������������һ�㲻��Ҫ�Ķ���
% opts.batchsize: ÿ�ε�����С�ߴ�
% opts.numepoches: �ܹ������Ĵ���
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
