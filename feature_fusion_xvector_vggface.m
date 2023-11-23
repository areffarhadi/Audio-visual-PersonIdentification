load("facenet118_fold3.mat");
load("dlnet_fold3.mat");

trainingImages1 = imageDatastore('.\vox2_test_mp4\Face\fold3\Train',...
"IncludeSubfolders",true,"LabelSource","foldernames");
validationImages1 = imageDatastore('.\vox2_test_mp4\Face\fold3\Test',...
"IncludeSubfolders",true,"LabelSource","foldernames");

datasetTrain='.\vox2_test_mp4\wav\fold3\train';
datasetTest = '.\vox2_test_mp4\wav\fold3\test';

adsTrain = audioDatastore(datasetTrain,IncludeSubfolders=true);
adsTrain.Labels = categorical(extractBetween(adsTrain.Files,fullfile(datasetTrain,filesep),filesep));

adsTest = audioDatastore(datasetTest,IncludeSubfolders=true);
adsTest.Labels = categorical(extractBetween(adsTest.Files,fullfile(datasetTest,filesep),filesep));

fs = 16e3;
windowDuration = 0.03;
hopDuration = 0.01;
windowSamples = round(windowDuration*fs);
hopSamples = round(hopDuration*fs);
overlapSamples = windowSamples - hopSamples;
numCoeffs = 30;
afe = audioFeatureExtractor( ...
    SampleRate=fs, ...
    Window=hann(windowSamples,"periodic"), ...
    OverlapLength=overlapSamples, ...
    mfcc=true);
setExtractorParameters(afe,"mfcc",NumCoeffs=numCoeffs)

adsTrainTransform = transform(adsTrain,@(x)xVectorPreprocess(x,afe,Segment=false,MinimumDuration=0.5));
% features = preview(adsTrainTransform);

numPar = numpartitions(adsTrain);
features = cell(1,numPar);
parfor ii = 1:numPar
    adsPart = partition(adsTrainTransform,numPar,ii);
    N = numel(adsPart.UnderlyingDatastores{1}.Files);
    f = cell(1,N);
    for jj = 1:N
        f{jj} = read(adsPart);
    end
    features{ii} = cat(2,f{:});
end

features = cat(2,features{:});
features = cat(2,features{:});
factors = struct("Mean",mean(features,2),"STD",std(features,0,2));
clear features f

adsTrainTransform = transform(adsTrain,@(x,myInfo)xVectorPreprocess(x,afe,myInfo, ...
    Segment=false,Factors=factors,MinimumDuration=0.6), ...
    IncludeInfo=true);
featuresTable = preview(adsTrainTransform);

adsTestTransform = transform(adsTest,@(x,myInfo)xVectorPreprocess(x,afe,myInfo, ...
    Segment=false,Factors=factors,MinimumDuration=0.5), ...
    IncludeInfo=true);
% for decision making function using layer "softmax" for feature "fc_1"
[xvecsTrain,labelsTrain] = predictBatch(dlnet,adsTrainTransform,Outputs="fc_2");
numEigenvectors = 150;

% projMat = helperTrainProjectionMatrix(xvecsTrain,labelsTrain,numEigenvectors);
% xvecsTrainP = projMat*xvecsTrain;
[xvecsTest,labelsTest] = predictBatch(dlnet,adsTestTransform,Outputs="fc_2");

% projMat = helperTrainProjectionMatrix(xvecsTest,labelsTest,numEigenvectors);
% xvecsTestP = projMat*xvecsTest;

% trainingImages2 = imageDatastore('.\dev\dev_voice\train',...
% "IncludeSubfolders",true,"LabelSource","foldernames");
% validationImages2 = imageDatastore('.\dev\dev_voice\test',...
% "IncludeSubfolders",true,"LabelSource","foldernames");

imageAugmenter = imageDataAugmenter( ...
    'RandRotation',[-20,20], ...
    'RandXTranslation',[-5 5], ...
    'RandYTranslation',[-5 5]);
augImds1 = augmentedImageDatastore([224,224],trainingImages1, ...
    'DataAugmentation',imageAugmenter);
% augImds2 = augmentedImageDatastore([227,227],trainingImages2, ...
%     'DataAugmentation',imageAugmenter);
%  augImds = augmentedImageDatastore([227,227],trainingImages);
validationImages11 = augmentedImageDatastore([224,224],validationImages1);
% validationImages22 = augmentedImageDatastore([227,227],validationImages2);



% voicelayer = 'drop7';
facelayer  ='dim_proj_relu';  % for decision making layer: 'classifier_low_dim_softmax' for feature extraction layer: 'dim_proj_relu'

facefeaturesTrain = activations(facenet,trainingImages1,facelayer,'OutputAs','rows');
% voicefeaturesTrain = activations(voicenet,augImds2,voicelayer,'OutputAs','rows');

facefeaturesTest = activations(facenet,validationImages11,facelayer,'OutputAs','rows');
% voicefeaturesTest = activations(voicenet,validationImages22,voicelayer,'OutputAs','rows');
facefeaturesTrain=facefeaturesTrain';
facefeaturesTest=facefeaturesTest';

load('fold3_xvector_facefeature_for_confusion_Net_result.mat', 'trainlabel')
load('fold3_xvector_facefeature_for_confusion_Net_result.mat', 'testlabel')

Xtrain=cat(1,facefeaturesTrain,xvecsTrain);
Xtest=cat(1,facefeaturesTest,xvecsTest);



net = trainSoftmaxLayer(Xtrain,trainlabel);
Ypred = net(Xtest);
[~,D]=max(Ypred(:,1:end));
[~,E]=max(testlabel(:,1:end));

[C,order] = confusionmat(E,D);
stats = statsOfMeasure(C, 1);