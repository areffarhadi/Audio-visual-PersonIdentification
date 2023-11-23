function [gradients,state,loss,YPred] = modelGradients(dlnet,X,Y)
% This function is only for use in this example. It may be changed or
% removed in a future release.

[YPred,state] = forward(dlnet,X);

loss = crossentropy(YPred,Y);
gradients = dlgradient(loss,dlnet.Learnables);

loss = double(gather(extractdata(loss)));

end