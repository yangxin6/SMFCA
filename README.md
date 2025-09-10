# SMFA: 3D Sparse Multi-Fourier Attention Network for Plant Segmentation and Classification

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

- Crops3D
- Pheno4D
- Maize-SYAU
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

