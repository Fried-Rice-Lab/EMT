# Efficient Mixed Transformer for Single Image Super-Resolution
Ling Zheng*, [Jinchen Zhu*](https://github.com/Jinchen2028), [Jinpeng Shi](https://github.com/jinpeng-s), Shizhuang Weng^

Recently, Transformer-based methods have achieved impressive results in single image super-resolution (SISR). However, the lack of locality mechanism and high complexity limit their application in the field of super-resolution (SR). To solve these problems, we propose a new method, Efficient Mixed Transformer (EMT) in this study. Specifically, we propose the Mixed Transformer Block (MTB), consisting of multiple consecutive transformer layers, in some of which the Pixel Mixer (PM) is used to replace the Self-Attention (SA). PM can enhance the local knowledge aggregation with pixel shifting operations. At the same time, no additional complexity is introduced as PM has no parameters and floating-point operations. Moreover, we employ striped window for SA (SWSA) to gain an efficient global dependency modelling by utilizing image anisotropy. Experimental results show that EMT outperforms the existing methods on benchmark dataset and achieved state-of-the-art performance. 
> *: (Co-)first author(s)
> 
> ^: corresponding author(s)
## Dependencies & Installation
```shell
conda create -n frl python
conda activate frl
pip install torch torchvision basicsr einops timm matplotlib
```
## Dataset
You can download the datasets you need from our [OneDrive](https://1drv.ms/u/s!AqKlMh-sml1mw362MfEjdr7orzds?e=budrUU) and place the downloaded datasets in the folder `datasets`. To use the YML profile we provide, keep the local folder `datasets` in the same directory tree as the OneDrive folder `datasets`.
> 🤠 All datasets have been processed in IMDB format and do not require any additional processing. The processing of the SISR dataset refers to the [BasicSR document](https://basicsr.readthedocs.io/en/latest/api/api_scripts.html), and the processing of the denoising dataset refers to the [NAFNet document](https://github.com/megvii-research/NAFNet/tree/main/docs).
> In adiition, special thanks to [Jinpeng Shi](https://github.com/jinpeng-s)
## Demo test
```shell
python test.py -expe options/EMT.yml -task options/LSR_x4.yml
```
## Training
```shell
python train.py -expe options/EMT.yml -task options/LSR_x4.yml
```

## Citation

If EMT helps your research or work, please consider citing the following works:

----------
```BibTex
@article{zheng2023efficient,
  title={Efficient Mixed Transformer for Single Image Super-Resolution},
  author={Zheng, Ling and Zhu, Jinchen and Shi, Jinpeng and Weng, Shizhuang},
  journal={arXiv preprint arXiv:2305.11403},
  year={2023}
}
```
