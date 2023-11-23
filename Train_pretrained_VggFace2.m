% Transfer Learning Using Squeezenet
%  close all
%  clear


trainingImages = imageDatastore('.\vox2_test_mp4\Face\fold3\train',...
"IncludeSubfolders",true,"LabelSource","foldernames");
validationImages = imageDatastore('.\vox2_test_mp4\Face\fold3\test',...
"IncludeSubfolders",true,"LabelSource","foldernames");
% 
% 
% 
% 
imageAugmenter = imageDataAugmenter( ...
    'RandRotation',[-20,20], ...
    'RandXTranslation',[-5 5], ...
    'RandYTranslation',[-5 5]);
augImds = augmentedImageDatastore([224,224],trainingImages, ...
    'DataAugmentation',imageAugmenter);
%%

numTrainImages = numel(trainingImages.Labels);

%% Load Pretrained Network

 load('VggFace2.mat');
%  load('net11.mat');

%% Transfer Layers to New Network

lgraph = layerGraph(net);
numClasses = numel(categories(trainingImages.Labels));

newFCLayer =  fullyConnectedLayer(numClasses,'WeightLearnRateFactor',1,'BiasLearnRateFactor',2);
lgraph = replaceLayer(lgraph,'classifier_low_dim',newFCLayer);
newClassificatonLayer = classificationLayer('Name','new_classoutput');
lgraph = replaceLayer(lgraph,'ClassificationLayer_classifier_low_dim',newClassificatonLayer);
 %%%
%%

miniBatchSize = 64;
numIterationsPerEpoch = floor(numel(trainingImages.Labels)/miniBatchSize);
options = trainingOptions('sgdm',...
    'MiniBatchSize',miniBatchSize,...
    'MaxEpochs',15,...
    'InitialLearnRate',1e-3,...
    'Plots','training-progress',...
    'Verbose',false,...
    'ValidationData',validationImages,...
    'Shuffle', 'every-epoch', ...
    'ValidationFrequency',numIterationsPerEpoch);
    
%%

facenet = trainNetwork(augImds,lgraph,options);

save('netTransfer','facenet');

predictedLabels = classify(facenet,validationImages);
yp = predict(facenet,validationImages);

valLabels = validationImages.Labels;
accuracy = mean(predictedLabels == valLabels);
disp(accuracy);

[C,order] = confusionmat(validationImages.Labels,predictedLabels);
stats = statsOfMeasure(C, 1);

