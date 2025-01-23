<div id="top" align="center">

# H-vmunet: High-order Vision Mamba UNet for Medical Image Segmentation
  
  Renkai Wu, Yinghao Liu, Pengchen Liang*, and Qing Chang* </br>
  
  [![Neucom](https://img.shields.io/badge/Neucom-2025.129447-blue)](https://doi.org/10.1016/j.neucom.2025.129447)
  [![arXiv](https://img.shields.io/badge/arXiv-2403.13642-b31b1b.svg)](https://arxiv.org/abs/2403.13642)

</div>

## News🚀
(2025.01.12) ***The paper has been accepted by Neurocomputing***🔥🔥

(2024.03.21) ***Model weights have been uploaded for download***🔥

(2024.03.21) ***The project code has been uploaded***

(2024.03.20) ***The first edition of our paper has been uploaded to arXiv*** 📃

**0. Main Environments.** </br>
The environment installation procedure can be followed by [VM-UNet](https://github.com/JCruan519/VM-UNet), or by following the steps below:</br>
```
conda create -n vmunet python=3.8
conda activate vmunet
pip install torch==1.13.0 torchvision==0.14.0 torchaudio==0.13.0 --extra-index-url https://download.pytorch.org/whl/cu117
pip install packaging
pip install timm==0.4.12
pip install pytest chardet yacs termcolor
pip install submitit tensorboardX
pip install triton==2.0.0
pip install causal_conv1d==1.0.0  # causal_conv1d-1.0.0+cu118torch1.13cxx11abiFALSE-cp38-cp38-linux_x86_64.whl
pip install mamba_ssm==1.0.1  # mmamba_ssm-1.0.1+cu118torch1.13cxx11abiFALSE-cp38-cp38-linux_x86_64.whl
pip install scikit-learn matplotlib thop h5py SimpleITK scikit-image medpy yacs
```

**1. Datasets.**

*A.ISIC2017* </br>
1- Download the ISIC 2017 train dataset from [this](https://challenge.isic-archive.com/data) link and extract both training dataset and ground truth folders inside the `/data/dataset_isic17/`. </br>
2- Run `Prepare_ISIC2017.py` for data preparation and dividing data to train,validation and test sets. </br>

*B.Spleen* </br>
1- Download the Spleen dataset from [this](http://medicaldecathlon.com/) link. </br>

*C.CVC-ClinicDB* </br>
1- Download the CVC-ClinicDB dataset from [this](https://polyp.grand-challenge.org/CVCClinicDB/) link. </br>

*D. Prepare your own dataset* </br>
1. The file format reference is as follows. (The image is a 24-bit png image. The mask is an 8-bit png image. (0 pixel dots for background, 255 pixel dots for target))
- './your_dataset/'
  - images
    - 0000.png
    - 0001.png
  - masks
    - 0000.png
    - 0001.png
  - Prepare_your_dataset.py
2. In the 'Prepare_your_dataset.py' file, change the number of training sets, validation sets and test sets you want.</br>
3. Run 'Prepare_your_dataset.py'. </br>

**2. Train the H_vmunet.**
```
python train.py
```
- After trianing, you could obtain the outputs in './results/' </br>

**3. Test the H_vmunet.**  
First, in the test.py file, you should change the address of the checkpoint in 'resume_model'.
```
python test.py
```
- After testing, you could obtain the outputs in './results/' </br>

**4. Get model weights**  

*A.ISIC2017* </br>
[Google Drive](https://drive.google.com/file/d/10If43saeVW06p9q3oePAL3hOHqRxFoZV/view?usp=sharing)

*B.Spleen* </br>
[Google Drive](https://drive.google.com/file/d/18aXOv8u-nFIbBdiUwnzHdQ7ELrNIhMu3/view?usp=sharing)

*C.CVC-ClinicDB* </br>
[Google Drive](https://drive.google.com/file/d/1mG_zOlsz7OuX_qHVmB3mjMeb1GUNgtkP/view?usp=sharing)


## Citation
If you find this repository helpful, please consider citing: </br>
```
@article{wu2025h,
  title={H-vmunet: High-order vision mamba unet for medical image segmentation},
  author={Wu, Renkai and Liu, Yinghao and Liang, Pengchen and Chang, Qing},
  journal={Neurocomputing},
  pages={129447},
  year={2025},
  publisher={Elsevier}
}
```
## Acknowledgement
Thanks to [Vim](https://github.com/hustvl/Vim), [HorNet](https://github.com/raoyongming/HorNet) and [VM-UNet](https://github.com/JCruan519/VM-UNet) for their outstanding work.
