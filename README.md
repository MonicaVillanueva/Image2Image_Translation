# Image2Image_Translation
Feature translation between images using Generative Adversarial Networks (GANs). It allows to modify a physical characteristic such as the hair color.

Authors
-------
This repository is being developed as part of the course [Scalable Machine Learning and Deep Learning (ID2223)](https://kth.instructure.com/courses/3687) at [KTH Royal Institute of Technology](http://kth.se), in the Fall 17 P2 round.

| Author               | GitHub                                            |
|:---------------------|:--------------------------------------------------|
| Héctor Anadón | [HectorAnadon](https://github.com/HectorAnadon)       |
| Sergio López | [Serlopal](https://github.com/Serlopal) |
| Mónica Villanueva | [MonicaVillanueva](https://github.com/MonicaVillanueva)     |


Content
-------
TODO: folder structure
 - [name](https://github.com/MonicaVillanueva/): Description

----------


Description
-------------
The idea of the project is to learn different features and translate them to images that originally do not have them, so that it generates images with modified using only one Generative Adversarial Network. We will try to reproduce the original concept, developed in the paper [“StarGAN: Uniﬁed Generative Adversarial Networks
for Multi-Domain Image-to-Image Translation”](https://arxiv.org/pdf/1711.09020.pdf), understand the underlying theory and compare the output agains the images provided in the paper.


Dataset
-------
Instead of using two datasets, we are simplifying the problem by using only one. In this case, the selected dataset is [CelebA](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html).
We will train only a handful of the features that this dataset provides, namely hair color (black/blond/brown), gender (male/female) and age (young/old).

n general, and contrary to most usual machine learning projects, this project does not require the use of an extensive dataset. Instead, we will only make use of “original photographies”, that we desire to transform, and “painting images”, that will provide the style the characteristics of which we want to extract.
Therefore, and in order to facilitate the final qualitative evaluation, we will make use of the same or similar photographies and paintings used in the paper mentioned above.

Libraries
-------
The code will be developed employing **Tensor Flow**, an open library for machine learning originally created by Google.

Experiments
-------
 - Replicate the result of the papers using the same pictures and same feature change used by them.
 - Test different results and study the veracity of the output.
 
Evaluation
-------
There will not be an exhaustive quantitative evaluation. A qualitative approach will be carried out, comparing the obtained results with their counterparts on the original paper, in order to evaluate the success of the replication process.



> **Note:**

> - The **final version** of this repository will be available on **January 2018**.

