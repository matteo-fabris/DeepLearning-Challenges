# DeepLearning-Challenges

This repo contains some scripts and notebooks developed to complete the two challenges of the course **Artificial Neural Networks and Deep Learning** at Politecnico di Milano on first semester of A.Y. 2021/2022.

## Description

The course included two challenges to start practicing with theoretical concepts seen during lessons.
The tasks covered are:
1. Image Classification
2. Time Series Forecasting

The first challenge consisted in classifing images of leaves. In particular the resulting model was required to distinguish between 14 different types of leaves.
In the second instead we were asked to develop a model that could predict the evolution of 7 features. No further information where provided about the nature of the process generating the data.

## The Team

The code was developed between november 2021 and january 2022 by:
- Samuele Portanti
- Mattia Portanti
- Matteo Fabris

We worked as a team, initially exploring different approaches then focusing on most promising ones.

## Results - Image Classification

### Dataset:
The dataset included around 10K images. Classes where imbalanced and some images from irrelevant domain where present. After some research we found that the dataset was a subset of [this](https://data.mendeley.com/datasets/tywbtsjrjv/1).

### Data Augmentation:
Images were augmented at train time defining custom augmentation functions. Transformations adopted include: noise addition, zoom in and out, shifts, rotations and patial obstruction.

### Proposed Solution:
After exploring different solutions, we adopted a transfer learning approach using EfficientNetB7. In particular, we fine tuned the network using class weights and a progressive fine tuning approach. This process was performed after exploring the architecture of EfficientNet in order to fine tune properly the network. Its worth mentioning that the learning rate tuning was fundamental, and after trying different approaches we used a reduction-on-plateau approach to enable a reduction of the learning rate at train time. All training procedures where performed on Colab with provided GPU to speed up the process.

### Performance:
The finetuned model achieved an accuracy of 99.56% on a balanced inner validation set. We achieved an accuracy of 93.40% during final test phase on unseen data. The lower level of performance can be justified by the presence of images coming from a different distribution w.r.t the one seen during training. The **concept drift** was confirmed by professors.

### Results - Time Series Forecasting

### Dataset:
The dataset included 7 features evolving with time. After a first analysis we noticed different magnitude of changes between the series and observed different levels of correlation between features. Furthermore data provided presented time intervals with constant values. 

### Proposed Solution:
We explored different solutions such as CNN, RNN, Encoder-Decoder approaches, and SCINet. After this first part we selected and adapted the SCINet architecture to perform a multivariate time series forecasting in our setting. We used the code published [here](https://github.com/Meatssauce/SCINet/blob/master/SCINet.py) as starting point and adapted it to our use case. Some models were optimized to support training on TPU achieving faster training times w.r.t. standard Colab GPU Tesla K80.

### Performance:
The final model achieved a RMSE of 3.77 on the unseen test set
