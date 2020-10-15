# HQS-Net: Neural Network-based Reconstruction in Compressed Sensing MRI Without Fully-sampled Training Data
![link to paper to be updated]()

## Abstract
Compressed Sensing MRI (CS-MRI) has shown promise in reconstructing under-sampled MR images, offering the potential to reduce scan times. Classical techniques minimize a regularized least-squares cost function using an expensive iterative optimization procedure. Recently, deep learning models have been developed that model the iterative nature of classical techniques by unrolling iterations in a neural network. While exhibiting superior performance, these methods require large quantities of ground-truth images and have shown to be non-robust to unseen data. In this paper, we explore a novel strategy to train an unrolled reconstruction network in an unsupervised fashion by adopting a loss function widely-used in classical optimization schemes. We demonstrate that this strategy achieves lower loss and is computationally cheap compared to classical optimization solvers while also exhibiting superior robustness compared to supervised models.

## Requirements
The code was tested on:
- python 3.6
- pytorch 1.1
- torchvision 0.3.0
- scikit-image 0.15.0
- scikit-learn 0.19.1
- matplotlib 3.0.2
- numpy 1.15.4
- tqdm 4.38.0

## Contact
Feel free to open an issue in github for any problems or questions.
