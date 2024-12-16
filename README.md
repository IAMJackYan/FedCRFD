# The code of FedCRFD (IEEE JBHI 2024)

## Prepare datasets
1. Download datasets from fastMRI
2. Process dataset:
python preprocess_fastMRI.py

## Train
python train.py

## Test
python test.py --resume /path/to/checkpoints


## Citation

```
@article{yan2024cross,
  title={Cross-modal vertical federated learning for mri reconstruction},
  author={Yan, Yunlu and Wang, Hong and Huang, Yawen and He, Nanjun and Zhu, Lei and Xu, Yong and Li, Yuexiang and Zheng, Yefeng},
  journal={IEEE Journal of Biomedical and Health Informatics},
  year={2024},
  publisher={IEEE}
}
```
