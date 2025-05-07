PAR = [];
f=1;    
%f=0.32;  验证集
noImages = 400;  
%noImages = 400*f/4;  验证集
noImagesPerBatch = 3000;  
%noImagesPerBatch = 240;  验证集
PAR.sizeInputData = 50;
PAR.colorChannel = 1;
PAR.stride = 10;
PAR.batchSize = 128;
PAR.numberPatches = PAR.batchSize*noImagesPerBatch;
noPatches = noImages*floor((321-PAR.sizeInputData)/PAR.stride + 1)*floor((481-PAR.sizeInputData)/PAR.stride + 1);
PAR.N_AUG = ceil(PAR.numberPatches/noPatches);
PAR.sourceFolder = fullfile('D:\matlab\lmageprocess\DnCNN-master\BSDS500_RR\Grayscale','train','*.jpg');
if f~=1
    PAR.sourceFolder = fullfile('D:\matlab\lmageprocess\DnCNN-master\BSDS500_RR\Grayscale','val','*.jpg');
end
PAR.Space = noPatches*PAR.N_AUG;

%%
disp(PAR);
[inputData,labels] = genPatches(PAR);

if(f~=1)
    inputDataVal = inputData;
    clear inputData;
    labelsVal = labels;
    clear labels;
end

%%
path = fullfile('D:\matlab\lmageprocess\DnCNN-master\BSDS500_RR\Grayscale\data');
if ~exist(path,'dir')
    mkdir(path);
end
 

if f==1
    save(fullfile(path,'inputData.mat'),'inputData', '-v7.3');
    save(fullfile(path,'labels.mat'),'labels', '-v7.3');
else
    save(fullfile(path,'inputDataVal.mat'),'inputDataVal', '-v7.3');
    save(fullfile(path,'labelsVal.mat'),'labelsVal', '-v7.3');
end
