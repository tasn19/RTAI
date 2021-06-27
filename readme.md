This repository contains RTAI’s submission for the 2021 [AI Against COVID-19: Screening X-ray Images for COVID-19 Infections](https://r7.ieee.org/montreal-sight/ai-against-covid-19/#Overview) competition hosted by IEEE SIGHT (Special Interest Group on Humanitarian Technology) of the Montreal Section, Vision and Image Processing Research Group of the University of Waterloo, and DarwinAI Corp. The submission won first place in the first phase of the competition.

The challenge of the first phase of the competition was to design a machine learning algorithm to detect COVID-19 positive and negative cases when given a dataset of chest X-rays. The competition dataset, created by [COVID-Net](https://alexswong.github.io/COVID-Net/), is available [here](https://www.kaggle.com/andyczhao/covidx-cxr2). 

# How to Run
Step 1

# Our Approach
Our solution implements a neural network to identify chest X-rays as either COVID-19 positive or COVID-19-negative. We used the PyTorch torchvision package and Google Colab. We started by conducting a literature search to review existing work implementing neural networks to screen chest X-rays for abnormalities (COVID-19 and other lung diseases). The findings of our literature search inspired the design of our algorithm. The components of our approach are as follows:

#### Data Augmentation Strategy 
We crop out the outer edges of the x-ray images to prevent the neural network from picking up on artifacts which may be present on the edges of the x-ray outside the chest region. We apply a random affine transformation (including a rotation of 5 degrees and translation of 0.05 (unit?)). We also resize all images to the same size. For a network learning from chest x-rays, data augmentation may not always improve performance because the pattern to be identified is not fixed like it is in traditional computer vision tasks. For example is a classic object detection problem, the network searches for e.g. a car in an image and the shape of the car is known. How an abnormality causing COVID-19 appears in the lungs is not as well-defined. In addition, some augmentations e.g. shearing and reflection, may result in images not practically seen in clinical settings, and therefore would be misleading for the algorithm. Our selected augmentations follow those approved by radiologists: rotations between -5 and 5 degrees, equal scaling in both x and y axes, and translation [1].


#### Model Selection
We used an Xception[2] model pretrained on ImageNet with a modified classification head for binary(?) classification. We reviewed research papers and found that [3] reported high performance with the Xception model for COVID-19 detection from chest X-rays compared to other architectures. In our experiments we tested ResNet-18 and ResNet-50 models and found they did not perform as well as the Xception model. The dataset used here is imbalanced; only 13.5% of the chest x-rays are for COVID-19 positive cases. To account for this imbalance, we used weighted binary cross-entropy loss as our loss function.

#### Hyperparameter Settings and Tuning
For training, Adam optimizer was used with batches of size 32. To improve our algorithm’s performance, we experimented with different BCE weights, learning rates, number of epochs, and with and without a pretrained model and dropout. The settings which yielded the best model are: BCE weight of 50, learning rate of 0.003 using pretrained Xception trained for 10 epochs.

# Evaluation Criteria
The algorithm is evaluated according to the following score:

Score = 6PN + 5PP + 3SN + 2SP

Where 

PN = positive predictive value of the COVID-19 negative class

PP = positive predictive value of the COVID-19 positive class

SN = sensitivity of the COVID-19 negative class

SP = sensitivity of the COVID-19 positive class

# Results 
add graphs?

Our best model achieved a score of 15.960075 (SP: 1, SN: 0.995, PP: 0.995025, PN: 1) on the test set with the probability threshold for the positive class set to 0.50. On the competition set it achieved a score of 14.94 (SP: 0.88, PP: 1, SN: 0.99, PN: 0.86). 

We adjusted the probability threshold for the positive class to reduce false negatives (accounting for cases which the model predicted to have a lower probability of being positive but are in fact positive). The model achieved a score of 15.762621 (SP:1, SN:0.97, PP: 0.970874 PN: 1) on the test set with the probability threshold set to 0.35. It’s performance on the competition set was 15.30 (SP: 0.93, PP: 0.99, SN: 0.98, PN: 0.93).


## References
1. Elgendi M, Nasir MU, Tang Q, Smith D, Grenier J-P, Batte C, Spieler B, Leslie WD, Menon C, Fletcher RR, Howard N, Ward R, Parker W and Nicolaou S (2021) [The Effectiveness of Image Augmentation in Deep Learning Networks for Detecting COVID-19: A Geometric Transformation Perspective](https://www.frontiersin.org/articles/10.3389/fmed.2021.629134/full). Front. Med. 8:629134. doi: 10.3389/fmed.2021.629134
2. Chollet, François. "[Xception: Deep learning with depthwise separable convolutions](https://arxiv.org/abs/1610.02357)." Proceedings of the IEEE conference on computer vision and pattern recognition. 2017.
3. Wang D, Mo J, Zhou G, Xu L, Liu Y (2020) An efficient mixture of deep and machine learning models for COVID-19 diagnosis in chest X-ray images. PLOS ONE 15(11): e0242535. https://doi.org/10.1371/journal.pone.0242535
