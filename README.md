# Multimodal Emotion Recognition using facial expression and other
physiological condition



## Helpers

Helpers include scripts for creating the datasets, methods for calculating metrics, transformation of physiological signals and face detection using Haar Cascade.

## Misc

Misc include miscellaneous files for visualization of data in plots .etc.

## Models 

Models include all model architectures used, a ResNet18 implementation, a ResNet-LSTM model for predicting BVP from sequences of facial expressions, a 3D-CNN for predicting BVP from sequences of facial expressions, a 1D-CNN for predicting emotions from sequential physiological signals, and a 3D-CNN for predicting emotions from sequences of facial images.

## Prelim_experiments

Prelim_experiments include the scripts for running the experiments regarding prediction of BVP based on sequences of facial expressions using both a ResNet-LSTM model and a 3D-CNN model.

## Results

Results include text files containing calculated metrics from the different experiments and some images used.

## Main_experiments

Main_experiments include the scripts for running the two main experiments; (1) The training and validation of the multimodal emotion recognition model created from fusing the 3D-CNN and 1D-CNN models. (2) The script which labels Toadstool based of features from CK+, both with ResNet and 3D-CNN

## Train_models

Train_models include the scripts for training the ResNet model and 3D-CNN model on the CK+ dataset.




