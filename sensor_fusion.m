

A = imageDatastore('.\vox2_test_mp4\Face',...
"IncludeSubfolders",true,"LabelSource","foldernames");

 B= imageDatastore('.\vox2_test_mp4\voice',...
"IncludeSubfolders",true,"LabelSource","foldernames");
 adres='.\vox2_test_mp4\facevoice';
C=imageDatastore('.\vox2_test_mp4\facevoice',...
"IncludeSubfolders",true,"LabelSource","foldernames");
[H,~]=size(A.Files);

parfor i=1:H
D=imresize(imread(A.Files{i,1}),[184,224]);
E=imresize(imread(B.Files{i,1}),[60,224]);
a=cat(1,D,E);
a=imresize(a,[224,224]);
imwrite(a,C.Files{i,1})
i
end
