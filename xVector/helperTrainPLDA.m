function plda = helperTrainPLDA(xvecs,YTrain,numIterations,numDimensions)
% This function is only for use in this example. It may be changed or
% removed in a future release. 
plda = audio.internal.ivector.trainPLDA(xvecs,YTrain(:), ...
    NumIterations=numIterations,NumDimensions=numDimensions,Verbose=false);
end