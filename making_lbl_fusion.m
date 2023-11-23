

dir1='.\vox2_test_mp4\Face\fold2\Test\';
dir2='.\vox2_test_mp4\Face\fold2\Train\';
load("VOX2_test_ID.mat")
[Spkr,~]=size(VOX2_test_ID);
ID=VOX2_test_ID;

for i=1:Spkr
    adr=strcat(dir1,ID{i,1});
    data=imageDatastore(adr);
    [num,~]=size(data.Files);
    test_ID_LBL{i,1}=ID{i,1};
    test_ID_LBL{i,2}=num;
   
end

for i=1:Spkr
    adr=strcat(dir2,ID{i,1});
    data=imageDatastore(adr);
    [num,~]=size(data.Files);
    train_ID_LBL{i,1}=ID{i,1};
    train_ID_LBL{i,2}=num;
   
end




ID_LBL=train_ID_LBL;
trainlabel=[];
B=0;
for i=1:Spkr
    trainlabel(i,B+1:B+train_ID_LBL{i,2})=1;
    [~,B]=size(trainlabel);
end


ID_LBL=test_ID_LBL;
testlabel=[];
B=0;
for i=1:Spkr
    testlabel(i,B+1:B+test_ID_LBL{i,2})=1;
    [~,B]=size(testlabel);
end
