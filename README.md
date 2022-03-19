# Instance Importance-aware Graph Convolutional Network (I<sup>2</sup>GCN)
This repository is an official PyTorch implementation of the paper **"Instance Importance-Aware Graph Convolutional Network for 3D Medical Diagnosis"** [[paper](https://www.sciencedirect.com/science/article/pii/S136184152200072X)] from Medical Image Analysis 2022.


## I<sup>2</sup>GCN for 3D Medical Diagnosis
<div align=center><img width="600" src=/fig/framework.png></div>

* Considering the high cost of collecting exhaustive annotations for 3D data, a sustainable alternative is to develop diagnosis algorithms with merely patient-level labels. We propose the Instance Importance-aware Graph Convolutional Network (I<sup>2</sup>GCN) under the multi-instance learning (MIL). 
* Using a preliminary MIL classifier, we first calculate the instance importance of each slice towards diagnosis, which is further utilized to promote the refined diagnosis branch. In the refined diagnosis branch, we devise the Instance Importance-aware Graph Convolutional Layer (I<sup>2</sup>GCLayer) to exploit complementary features in both importance-based and feature-based topologies. Moreover, the importance-based Sub-Graph Augmentation (SGA) is devised to alleviate the deficient supervision of 3D dataset.

### Download
The processed CC-CCII dataset can be downloaded from [Google Drive]. Put the downloaded .npy files in a newly-built folder ```./data/```. Please note that among the three-fold cross-validation with random split, the performance of ```split1``` and ```split2``` is slightly higher than the ```split0```.

## Dependencies
* Python 3.6
* PyTorch >= 1.3.0
* numpy 1.19.4
* scikit-learn 0.24.2
* scipy 1.3.1


## Code
Clone this repository into any place you want.
```bash
git clone https://github.com/CityU-AIM-Group/I2GCN.git
cd I2GCN
mkdir experiment; mkdir data
```
## Quickstart 
* Train the I2GCN with default settings:
```python
python ./main.py --theme default --test_split 2 
```
We provide the dataloader with two ways of loading npy files, including ```online``` and ```offline```.

## Cite
If you find our work useful in your research or publication, please cite our work:
```
@article{CHEN2022102421,
	title = {Instance Importance-Aware Graph Convolutional Network for 3D Medical Diagnosis},
	author = {Zhen Chen and Jie Liu and Meilu Zhu and Peter Y.M. Woo and Yixuan Yuan},
	journal = {Medical Image Analysis},
	pages = {102421},
	year = {2022},
	issn = {1361-8415},
	doi = {https://doi.org/10.1016/j.media.2022.102421}
}
```


## Acknowledgements
* [CC-CCII dataset](http://ncov-ai.big.ac.cn/download?lang=en) from China National Center for Bioinformation, the largest public COVID-19 dataset of 3D lung CT scans until publication.
