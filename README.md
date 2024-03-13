# Low-Rank Rescaled Vision Transformer Fine-Tuning: A Residual Design Approach (RLRR)

------
This repository is the official implementation of RLRR. In this study, we approach the problem from the perspective of Singular Value Decomposition (SVD) of pre-trained parameter matrices, providing insights into the tuning dynamics of existing methods.
![在这里插入图片描述](https://github.com/zstarN70/RLRR/blob/main/framework.png)

## Usage

-----
### Environment
To install requirements:
```
conda env create -n RLRR -f environment.yaml
```
Before running the code, please activate this conda environment.
### Data Preparation
- FGVC & vtab-1k

You can follow [VPT](https://github.com/KMnP/vpt) to download them. 

Since the original [vtab dataset](https://github.com/google-research/task_adaptation/tree/master/task_adaptation/data) is processed with tensorflow scripts and the processing of some datasets is tricky, we also upload the extracted vtab-1k dataset in [onedrive](https://shanghaitecheducn-my.sharepoint.com/:f:/g/personal/liandz_shanghaitech_edu_cn/EnV6eYPVCPZKhbqi-WSJIO8BOcyQwDwRk6dAThqonQ1Ycw?e=J884Fp) for your convenience. You can download from here and then use them with our [vtab.py](https://github.com/dongzelian/SSF/blob/main/data/vtab.py) directly. (Note that the license is in [vtab dataset](https://github.com/google-research/task_adaptation/tree/master/task_adaptation/data)).


### Pre-trained model preparation

- For pre-trained ViT, Swin-B models on ImageNet-21K. You can also manually download them from [ViT](https://github.com/google-research/vision_transformer),[Swin Transformer](https://github.com/microsoft/Swin-Transformer).


## Train & Inference
- Clone this repo:
```bash
git clone https://github.com/zstarN70/RLRR.git
cd RLRR
```

- To fine-tune a pre-trained ViT model on VTAB, run:
```bash
CUDA_VISIBLE_DEVICES=0 python  train_vtab.py --dataset_name=kitti
```

- To fine-tune a pre-trained ViT model on FGVC, run:
```bash
CUDA_VISIBLE_DEVICES=0 python  train_fgvc.py --dataset_name=kitti
```

### Citation
If this project is helpful for you, you can cite our paper:
```

```

### Acknowledgement
The code is built upon [timm](https://github.com/jeonsworld/ViT-pytorch). The processing of the vtab-1k dataset refers to [vpt](https://github.com/KMnP/vpt), [vtab github repo](https://github.com/google-research/task_adaptation/tree/master/task_adaptation/data), and [NOAH](https://github.com/ZhangYuanhan-AI/NOAH).


