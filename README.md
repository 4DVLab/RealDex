# RealDex
This repository is for RealDex dataset and the code for IJCAI 2024 paper [RealDex: Towards Human-like Grasping for Robotic Dexterous Hand](https://4dvlab.github.io/RealDex_page/)
![teaser](./images/teaser.png)

## Dataset
```
pip install gdown
pip install requests
```
Before download, first check the availability of the download links:
```
python download/download_dataset.py --test
```
If all links are available:
```
python download/download_dataset.py --dest_folder /path/to/zipped-data
```
Unzip the files:
```
python download/unzip_data.py --data_folder /path/to/zipped-data --dest_folder /path/to/data
```

## Installation

## data preprocess
Generate point clouds from RGBD images
```bash
cd ./data_preprocess
python gen_pcd.py
```

