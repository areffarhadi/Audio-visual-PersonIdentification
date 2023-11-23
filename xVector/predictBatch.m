function [xvecs,labels] = predictBatch(dlnet,ds,nvargs)
arguments
    dlnet
    ds
    nvargs.Outputs = [];
end
if ~isempty(ver("parallel"))
    pool = gcp;
    numPartition = numpartitions(ds,pool);
else
    numPartition = 1;
end
xvecs = [];
labels = [];
outputs = nvargs.Outputs;
parfor partitionIndex = 1:numPartition
    dsPart = partition(ds,numPartition,partitionIndex);
    partitionFeatures = [];
    partitionLabels = [];
    partitionIndex;
    ii=1;
    while hasdata(dsPart)
        atable = read(dsPart);
        F = atable.features;
        dsPart.UnderlyingDatastores{1, 1}.Files{ii, 1};
        ii=ii+1;
        L = atable.labels;
        aa=numel(L);
        if numel(L) >1
            aa=1;
        end
        for kk = 1:aa
            if isempty(outputs)
                f = gather(extractdata(predict(dlnet,(dlarray(F{kk},"CTB")))));
            else
                f = gather(extractdata(predict(dlnet,(dlarray(F{kk},"CTB")),Outputs=outputs)));
            end
            l = L(kk);
            partitionFeatures = [partitionFeatures,f];
            partitionLabels = [partitionLabels,l];
        end
        
    end
    xvecs = [xvecs,partitionFeatures];
    labels = [labels,partitionLabels];
end
end