function detTable = helperDetectionErrorTradeoff(xvecs,labels,enrolledLabels,plda)
% This function is only for use in this example. It may be changed or
% removed in a future release. 
enrolledLabels.Properties.VariableNames{1} = 'ivector';
detTable = audio.internal.ivector.det(xvecs,labels,enrolledLabels,plda,Verbose=false);
end