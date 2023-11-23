function projMat = helperTrainProjectionMatrix(xvecs,YTrain,numEigenvectors)
% This function is only for use in this example. It may be changed or
% removed in a future release. 
projMat = audio.internal.ivector.trainProjMat(xvecs,YTrain(:),eye(size(xvecs,1)), ...
    NumEigenvectors=numEigenvectors,Verbose=false);
end