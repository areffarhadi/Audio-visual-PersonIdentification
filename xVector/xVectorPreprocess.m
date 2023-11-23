function [output,myInfo] = xVectorPreprocess(audioData,afe,myInfo,nvargs)
% This function is only for use in this example. It may be changed or
% removed in a future release.
arguments
    audioData
    afe
    myInfo = []
    nvargs.Factors = []
    nvargs.Segment = true;
    nvargs.MinimumDuration = 1;
    nvargs.UseGPU = false;
end

% Place on GPU if requested
if nvargs.UseGPU
    audioData = gpuArray(audioData);
end

% Scale
audioData = audioData/max(abs(audioData(:)));

% Protect against NaNs
audioData(isnan(audioData)) = 0;

% Determine regions of speech
mergeDur = 0.3; % seconds
idx = detectSpeech(audioData,afe.SampleRate,MergeDistance=afe.SampleRate*mergeDur);

% If a region is less than MinimumDuration seconds, drop it.
if ~nvargs.Segment
    idxToRemove = (idx(:,2)-idx(:,1))<afe.SampleRate*nvargs.MinimumDuration;
    idx(idxToRemove,:) = [];
end
numSegments = size(idx,1);
audi=[];
for ii = 1:numSegments
    audi = [audi;audioData(idx(ii,1):idx(ii,2))];
end
% Extract features
features = cell(1,1);
    temp = (single(extract(afe,audi)))';
    if isempty(temp)
        temp = zeros(30,15,"single");
    end
    features{1} = temp;


% Standardize features
if ~isempty(nvargs.Factors)
    features = cellfun(@(x)(x-nvargs.Factors.Mean)./nvargs.Factors.STD,features,UniformOutput=false);
end

% Cepstral mean subtraction (for channel noise)
if ~isempty(nvargs.Factors)
    fileMean = mean(cat(2,features{:}),"all");
    features = cellfun(@(x)x - fileMean,features,UniformOutput=false);
end

% if ~nvargs.Segment
%     features = {cat(2,features{:})};
% end
if isempty(myInfo)
    output = features;
else
    labels = repelem(myInfo.Label,numel(features),1);

    output = table(features,labels);
end
end