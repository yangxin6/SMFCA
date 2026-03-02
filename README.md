# SMFCA: 3D Sparse Multi-Fourier Attention Network for Plant Segmentation and Classification

![SMFCA-Net.png](figs%2FSMFCA-Net.png)
![MFCA.png](figs%2FMFCA.png)

## Environment
- Ubuntu 2204
- cuda 11.8
- python 3.8.x
- pytorch=2.1.0
```bash
sudo apt-get install libsparsehash-dev

conda env create -f environment.yaml 

cd libs/pointgroup_ops
python setup.py install
cd ../..

```

## Datasets

- Crops3D: 10.1038/s41597-024-04290-0
- Pheno4D: 10.1371/journal.pone.0256340
- Maize-SYAU: 10.1016/j.isprsjprs.2024.03.025
- Peanut

## Model Results


Train:
```bash
python tools/train.py --config-file configs/plant3d_organ_semantic/2Maize/semseg-smfca-v4m0-0-base.py --options save_path="exp/plant3d_organ_semantic/2Maize/semseg-spfa-v4m0-0-base_exp_0721"

```

Test:
```bash
python tools/test.py --config-file configs/plant3d_organ_semantic/2Maize/semseg-smfca-v4m0-0-base.py  --options save_path="exp/plant3d_organ_semantic/2Maize/semseg-spfa-v4m0-0-base_exp_0721" weight="exp/plant3d_organ_semantic/2Maize/semseg-spfa-v4m0-0-base_exp_0721/model/model_best.pth"
```

We have released the trained model weights and segmentation results for classification, semantic segmentation, and instance segmentation:  
[model_pth]()



## Reference
- [Pointcept](https://github.com/Pointcept/Pointcept)

## Citation

```
@ARTICLE{11381999,
  author={Yang, Xin and Shang, Zhijun and Huang, He and Liu, Congjun and Xu, Tongyu and Miao, Teng},
  journal={IEEE Transactions on Geoscience and Remote Sensing}, 
  title={SMFCA-Net: Sparse Multifrequency Cross-Attention Network for Single-Plant Point Cloud Classification and Segmentation}, 
  year={2026},
  volume={64},
  number={},
  pages={1-17},
  keywords={Three-dimensional displays;Point cloud compression;Transformers;Semantics;Windows;Instance segmentation;Frequency-domain analysis;Spatial resolution;Solid modeling;Robustness;3-D Fourier transform;classification;frequency transform;plant point clouds;segmentation},
  doi={10.1109/TGRS.2026.3662324}}

```

