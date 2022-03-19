# Instance Importance-aware Graph Convolutional Network (I<sup>2</sup>GCN)
This repository is an official PyTorch implementation of the paper **"Instance Importance-Aware Graph Convolutional Network for 3D Medical Diagnosis"** [[paper](https://www.sciencedirect.com/science/article/pii/S136184152200072X)] from Medical Image Analysis 2022.


## I<sup>2</sup>GCN for 3D Medical Diagnosis
<div align=center><img width="600" src=/fig/framework.png></div>


### Download
The processed CC-CCII dataset can be downloaded from [Google Drive]. Put the downloaded .npy files in a newly-built folder ```./data/```.

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
