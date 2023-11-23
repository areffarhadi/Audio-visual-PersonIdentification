function [sequences,labels] = preprocessMiniBatch(sequences,labels)
% This function is only for use in this example. It may be changed or
% removed in a future release.

trimDimension = 2;
lengths = cellfun(@(x)size(x,trimDimension),sequences);
minLength = min(lengths);
sequences = cellfun(@(x)randomTruncate(x,trimDimension,minLength),sequences,UniformOutput=false);
sequences = cat(3,sequences{:});
        
labels = cat(2,labels{:});
labels = onehotencode(labels,1);
labels(isnan(labels)) = 0;
end