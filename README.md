# RealDex
This repository is for RealDex dataset and the code for IJCAI 2024 paper [RealDex: Towards Human-like Grasping for Robotic Dexterous Hand](https://4dvlab.github.io/RealDex_page/)
![teaser](./images/teaser.png)

## Dataset
You can download object models from this link: https://drive.google.com/file/d/1u4q9N_q-pgEYfyzr94vPk4YWkcx8vggh/view?usp=sharing
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

## data preprocess
Generate point clouds from RGBD images
```bash
cd ./data_preprocess
python gen_pcd.py
```

## Installation

## Training for grasp pose generation
```commandline
cd dexgrasp_generation/

python ./network/train.py --config-name cm_net_config \
                          --exp-dir ./runs/cm_net_train

python ./network/train.py --config-name cvae_config \
                          --exp-dir ./runs/cvae_train
```


## Acknowledgement
We have intensively borrow codes from the following repositories. Many thanks to the authors for sharing their codes.
- [Unidexgrasp](https://github.com/PKU-EPIC/UniDexGrasp.git)
- [GOAL](https://github.com/otaheri/GOAL.git)
- [GraspTTA](https://github.com/hwjiang1510/GraspTTA.git)
