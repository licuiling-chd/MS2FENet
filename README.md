# Multiscale Spatial-Spectral Feature Extraction Network for Hyperspectral Image Classification

This repository is the implementation of our paper: [Multiscale Spatial-Spectral Feature Extraction Network for Hyperspectral Image Classification](https://ieeexplore.ieee.org/document/9786664). 

If you find this work helpful, please cite our paper:

  @ARTICLE{9786664,
  author={Ye, Zhen and Li, Cuiling and Liu, Qingxin and Bai, Lin and Fowler, James E.},
  journal={IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing}, 
  title={Multiscale Spatial-Spectral Feature Extraction Network for Hyperspectral Image Classification}, 
  year={2022},
  volume={15},
  number={},
  pages={4640-4652},
  doi={10.1109/JSTARS.2022.3179446}}
 
 ## Descriptions
 
Convolutional neural networks have garnered increasing interest for the supervised classification of hyperspectral imagery. However, images with a wide variety ofspatial land-cover sizes can hinder the feature-extraction ability of traditional convolutional networks. Consequently, many approaches intended to extract multiscale features have emerged; these techniques typically extract features in multiple parallel branches using convolutions of differing kernel sizes with concatenation or addition employed to fuse the features resulting from the various branches. In contrast, the present work explores a multiscale spatial-spectral feature-extraction network that operates in a more granular manner. Specifically, in the proposed network, a multibranch structure expands the convolutional receptive fields through the partitioning of input feature maps, applying hierarchical connections across the partitions, crosschannel feature fusion via pointwise convolution, and depthwise three-dimensional (3-D) convolutions for feature extraction. Experimental results reveal that the proposed multiscale spatial-spectral feature-fusion network outperforms other state-of-the-art networks at the supervised classification of hyperspectral imagery while being robust to limited training data.
 
 ![image](https://github.com/liuqingxin-chd/MF2CNet/blob/main/network.jpg)
