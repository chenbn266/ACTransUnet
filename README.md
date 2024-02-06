##### ACTransU-Net

#### Introduction
This work presents a novel adaptive cascaded transformer U-Net (ACTransU-Net) for brain tumor segmentation in magnetic resonance imaging, which integrates both the transformer and dynamic convolution into a cascaded U-Net architecture to adaptively capture both global information and local details of the brain tumors.The ACTransU-Net firstly cascades two three-dimensional U-Nets into a two-level network to segment the brain tumor from coarse to fine. Subsequently, it integrates a full-dimensional dynamic convolution module into the second-stage shallow encoder and decoder so as to enhance the local detail representation of various brain tumors by dynamically adjusting the convolution kernel parameters. In addition, a 3D Swain transformer module is also introduced into the second stage deep encoder and decoder to capture the long-range dependency of the image, which helps in tuning the global representation of brain tumors.

#### Dataset
We used BraTS 2020 and BraTS 2021, the data can be found again here: https://ipp.cbica.upenn.edu/ and https://www.synapse.org/#!Synapse:syn25829067/wiki/

#### Requirements
git+https://github.com/NVIDIA/dllogger


git+https://github.com/NVIDIA/mlperf-common.git  

nibabel>=3.2.1

joblib>=1.0.1

pytorch-lightning>=1.7.7

scikit-learn>=1.0 

scikit-image>=0.18.3

scipy>=1.8.1 


rich>=12.5.0


monai>=1.0.0



#### preprocessing
Each example in the BraTS dataset consists of four NIfTI files with different MRI modalities (file name suffixes flair, t1, t1ce, t2). In addition, the examples in the training dataset have an annotated NIfTI file (filename suffix seg). The first step in data preprocessing was to stack all four modalities so that each example had a shape of (4, 240, 240, 155) (with an input tensor of (C, H, W, D) layout, where C-channel, H-height, W-width and D-depth). Then, excess background voxels on the boundary of each body are cropped (voxel value is zero) as they do not provide any useful information and can be ignored by the neural network. Subsequently, for each instance, the mean and standard deviation within the non-zero region of each channel are calculated separately. All voxels were normalized by first subtracting the mean and then dividing by the standard deviation. The background voxels were not normalized, so their values remained zero. In order to distinguish between the background voxels and the normalized voxels with values close to zero, we added an input channel that encodes the foreground voxels with a single click and stacked it with the input data. Thus, each instance has 5 channels. \ 
You should run this command 'python preprocess --data inputdata --results outputdata --exec_mode training --ohe --verbose --task 13'

#### Training
Our models are implemented in python via Pytorch deep learning framework, and they are trained using the Adam optimizer with an initial learning rate of 0.0001 and a weight decay rate of 0.0001. We adopt a cosine annealing learning rate tuning strategy, with the batch size set to 4, and 1000 epochs of training using automatic mixed precision. All of the experiments are performed on a server equipped with two NVIDIA RTX3090 GPUs (2*24GB memory). \ 
You should run this command 'python main.py --brats --batch_size 2 --scheduler --learning_rate 0.0001 --epochs 1000 --fold 0 --amp --gpus 2 --task 13 --save_ckpt --tb_logs --ckpt_name name'

### Inference
In the inference process, we used Sliding Window Inference (SWI) in the inference process. We also applied the test time augmentation (TTA) method to improve the robustness and accuracy of the prediction results. For further post-processing, we converted regionally segmented brain tumors into original categories. Based on previous experience and experimental evidence, WT voxels with a probability of less than 0.4 are replaced with label 0. Similarly, TC voxels with a probability of less than 0.4 are replaced with label 2, and ET voxels with a probability of less than 0.7 are replaced with label 1. Subsequently, following the scoring rules of the BraTS competition, we identify independent tumor blocks associated with ET voxels. These blocks contained fewer than 8 voxels with an average probability of less than 0.9, which we replaced with label 1. Additionally, ET voxels with fewer than 500 total voxels and an average probability of less than 0.9 across the entire tumor block were also replaced with label 1 to ensure that they continue to be part of the tumor core. \ 
You should run this command 'python main.py --gpus 1 --amp --save_preds --exec_mode predict --brats --tta'

#### Visualization
You should change the path in visual.py and run it directly.
