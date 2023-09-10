# Semi-Supervised Learning for Remote Sensing Scene Classification (SSL-for-RS)

Welcome to the GitHub repository for our study on Deep Semi-Supervised Learning (DSSL) applied to Remote Sensing Scene Classification. 
This repository contains configuration files for reproducibility and a comprehensive implementation split across four Colab notebooks. 
These notebooks cover data and software acquisition, model training, accuracy evaluation, and inference, showcasing the effectiveness of DSSL with limited labeled data.

## Overview

This repository presents a comparative study of various DSSL methods, including FixMatch [1], CoMatch [2], and Class Aware Contrastive Semi-Supervised Learning (CCSSL) [3], on two remote sensing datasets: UCM [4] and AID [5]. 
By leveraging a small number of labeled examples alongside unlabeled data, these methods demonstrate their capability to significantly improve classification accuracy.

## Key Features

- Configuration files adapted from Classification-SemiCLS GitHub [3] for reproducibility and ease of experimentation.
  They provide the details of the experiments, including dataset splits, augmentations, and training settings.
- Logs of the experimental results showcasing the performance of DSSL methods compared to supervised benchmarks.
  The compressed version of all the logs can be downloaded from google drive [here](https://drive.google.com/file/d/1QgpzJQhVqFlFsAhd8DV8yxxuUW8h86eA/view?usp=sharing).
  
## References
[1] K. Sohn, D. Berthelot, N. Carlini, Z. Zhang, H. Zhang, C. A. Raffel, E. D. Cubuk, A. Kurakin, and C. Li, "FixMatch: Simplifying Semi-Supervised Learning with Consistency and Confidence," in Advances in Neural Information Processing Systems, vol. 33, H. Larochelle et al. (eds.), Curran Associates, Inc., 2020, pp. 596-608. 

[2] J. Li, C. Xiong, and S. C. H. Hoi, "CoMatch: Semi-supervised Learning with Contrastive Graph Regularization," in Proceedings of the 2021 IEEE/CVF International Conference on Computer Vision (ICCV), 2021, pp. 9455-9464. doi: 10.1109/ICCV48922.2021.00934.

[3] F. Yang, K. Wu, S. Zhang, G. Jiang, Y. Liu, F. Zheng, W. Zhang, C. Wang, and L. Zeng, "Class-Aware Contrastive Semi-Supervised Learning," in Proceedings of the 2022 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2022, pp. 14401-14410. doi: 10.1109/CVPR52688.2022.01402.

[4] Y. Yang and S. Newsam, "Bag-of-Visual-Words and Spatial Extensions for Land-Use Classification," in Proceedings of the 18th SIGSPATIAL International Conference on Advances in Geographic Information Systems (GIS '10), ACM, 2010, pp. 270-279. doi: 10.1145/1869790.1869829. URL: https://doi.org/10.1145/1869790.1869829.

[5] G.-S. Xia, J. Hu, F. Hu, B. Shi, X. Bai, Y. Zhong, L. Zhang, and X. Lu, "AID: A Benchmark Data Set for Performance Evaluation of Aerial Scene Classification," IEEE Transactions on Geoscience and Remote Sensing, vol. 55, no. 7, pp. 3965-3981, 2017. doi: 10.1109/TGRS.2017.2685945.




