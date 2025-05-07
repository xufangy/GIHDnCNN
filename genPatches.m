function [inputData,labels] = genPatches(PAR)
sizeInputData = PAR.sizeInputData;
colorChannel = PAR.colorChannel;
stride = PAR.stride;
numberPatches = PAR.numberPatches;
N_AUG = PAR.N_AUG;
batchSize = PAR.batchSize;
sourceFolder = PAR.sourceFolder;
Space = PAR.Space;

noSamples = 0;
inputData = zeros(Space,sizeInputData,sizeInputData,colorChannel,'single');
labels = zeros(Space,sizeInputData,sizeInputData,colorChannel,'single');
disp('Space Allocated');
 

%filepaths = [];
filepaths = cat(1,dir(fullfile(sourceFolder)));

for i = 1:length(filepaths)
    I = imread(fullfile(filepaths(i).folder,filepaths(i).name));
    for j = 1:N_AUG
        if j == 1
            Iaug = I;
        elseif j ==2
            Iaug = flipud(I); %flips the rows upside down
        elseif j==3
            Iaug = rot90(I,1);
        elseif j==4
            Iaug = rot90(I,1);
            Iaug = flipud(Iaug);
        elseif j==5
            Iaug = rot90(I,2);
        elseif j==6
            Iaug = rot90(I,2);
            Iaug = flipud(Iaug);
        elseif j==7
            Iaug = rot90(I,3);
        elseif j==8
            Iaug = rot90(I,3);
            Iaug = flipud(Iaug);
        end
        
        inputI = Iaug; 
        labelI = I;
        [height,width,c] = size(inputI);
        for m = 1 : stride : (height-sizeInputData+1)
            for n = 1 : stride : (width-sizeInputData+1)
                inputIpatch = inputI(m : m+sizeInputData-1, n : n+sizeInputData-1,:);
                labelIpatch = labelI(m : m+sizeInputData-1, n : n+sizeInputData-1,:);
                noSamples = noSamples + 1;
                %sigma = 15 / 255;
                sigma = 55/255*rand;
                inputIpatch_N = imnoise(inputIpatch,'gaussian',0,sigma*sigma);
                inputData(noSamples,:,:,1:colorChannel) = im2single(inputIpatch_N);
                labels(noSamples,:,:,1:colorChannel) = im2single(labelIpatch);
            end
        end
    end
    String = ['image-',num2str(i),' completed'];
    disp(String);
end

labels = inputData - labels;
disp('checkpoint:1');

shuffleOrder = randperm(size(inputData,1));
inputData = inputData(shuffleOrder,:,:,:);
labels = labels(shuffleOrder,:,:,:);
disp('checkpoint:2');

sizeRequired = size(inputData,1) - mod(size(inputData,1),batchSize);
disp('checkpoint:3');
noSamples = min(sizeRequired,numberPatches);
inputData = inputData(1:noSamples,:,:,:);
labels = labels(1:noSamples,:,:,:);
end            
            