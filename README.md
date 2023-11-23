# Audio Visual Person Identification in different strategies on the Voxceleb2 Dataset

Implementation of different strategies in audio-visual person identification on the Voxceleb2 dataset.

This repo is the source code of a paper that is under review.
In that paper, we evaluate different strategies in multimodal learning for person identification.
There are different scripts:

SpeakerIdentificationUsingXVectors.xlm ----------> Speaker identification using xVector 

Train_pretrained_VggFace2.m ---------> face recognition using VGGFace2 pre-trained net, and sensory fusion with import fused image data.

 feature_fusion_xvector_vggface.m --------> train and evaluate feature fusion mode and, with some changes in the name of the destination layers, work as score fusion.

This is the training plot of the proposed Facenet from pretrained VGGface2:
![facenet_training_process_fold3](https://github.com/areffarhadi/Audio-visual_Person_Identification/assets/93467718/c33c3011-7b8a-42be-b266-8bdda982b3ff)



 ********** details will be added after the paper is published ************
