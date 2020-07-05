# Hyperspectral-Image-Classification-Based-on-3D-Octave-Convolution-with-Spatial-Spectral-Attention
The official codes for paper "Hyperspectral Image Classification Based on 3D Octave Convolution With Spatial-spectral Attention Network"
## Install dependencies
    numpy
    python==3.6
    sklearn
    tensorflow==1.5
    pycharm
## dataset
    We conduct the experiments on the University of Pavia data set. To train and test our model, you should 
    download the data set and modify image's path according to your needs.
## data process 
    Since the input of our network is patch, it is necessary to preprocess the original hyperspectral image. 
    The image preprocessing includes data normalization and patch cutting. 
    In our experiments, the patch size is set to 13*13, and the training set and testing set are divided.
    For your convenience, we have uploaded the compiled training and test sets. If you want to divide the 
    data set yourself, run the program data_prosess.py
        
## train
## test
