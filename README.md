
## SimpleMatch: A Simple and Strong Baseline for Semantic Correspondence

This is the official code for SimpleMatch implemented with PyTorch.

### Prepare Datasets

1) Download PF-PASCAL and PF-WILLOW datasets: [Download](https://www.di.ens.fr/willow/research/proposalflow/)
2) Download SPair-71k dataset: [Download](https://cvlab.postech.ac.kr/research/SPair-71k/data/SPair-71k.tar.gz)
3) Unzip the datasets and place them in the `data` directory
4) Organized as follows
```
./data
    PF-PASCAL
    PF-WILLOW
    SPair-71k
```

### Download pretrained parameters of backbone.
1) DINOv2: [Download](https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_pretrain.pth)
2) ibot: [Download](https://lf3-nlp-opensource.bytetos.com/obj/nlp-opensource/archive/2022/ibot/vitb_16/checkpoint_teacher.pth)
3) resnet101: [Download](https://download.pytorch.org/models/resnet101-5d3b4d8f.pth)

`mkdir checkpoints` and put them in checkpoints directory.


### Environment Settings

```
conda create -n DCM python=3.8.0
conda activate DCM
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113

pip install -r requirements.txt
```

### Download Pretrained weights 

[Google Drive](https://drive.google.com/drive/folders/1lhEEpqEVYGIdXLwmmmtUYR0G1fU72BUi?usp=sharing)

```
mkdir ckpts

# Then, place the downloaded checkpoints in the ckpts directory
```


### Evaluation on SPair-71k

(To evaluate on other datasets, replace the configuration file and the corresponding checkpoint path.)

```
python test.py \
    --config configs/task_dinov2-b14_448x448_spair.py \
    --model_path ckpts/dinov2_448x448_spair/best_model.pth  \
    --log_name infer  
```

### Evaluation on AP10k

```
python test.py --config configs/task_dinov2-b14_448x448_ap10k.py \
    --model_path ckpts/dinvov2_448x448_ap10k/best_model.pth  \
    --log_name infer \
    --cfg-options \
    test_dataloader.dataset.eval_type=cross-family  # `intra-species`, `cross-species`, `cross-family`

```


### Training on SPair-71k
```
python train.py  \
    --config configs/task_dinov2-b14_252x252_spair.py \
    --work-dir work_dirs/tmp 
```

### Training on PF-PASCAL
```
python train.py  \
    --config configs/task_ibot-b16_256x256_pfpascal.py \
    --work-dir work_dirs/tmp 
```
