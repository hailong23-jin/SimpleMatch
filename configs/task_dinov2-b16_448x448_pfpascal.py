log_level = 'INFO'
interval = 10      # record logs every n iters
device = 'cuda'
max_epochs = 100
image_size = 448

optimizer = dict(type='Adam', lr=1e-4)
lr_scheduler = dict(type='StepLR', step_size=200, gamma=0.95)

metric=dict(
    type='CorrespondenceMetric', 
    alpha=0.1, 
    img_size=image_size
)

model = dict(
    type='CorrespondenceModel',
    pair_image_encoder_cfg=dict(
        type='PairImageEncoder',
        backbone_cfg=dict(
            type='DinoVisionTransformer',
            img_size=518,
            patch_size=14,
            embed_dim=768,
            depth=12,
            num_heads=12,
            mlp_ratio=4,
            init_values=1.0,
            block_chunks=0,
        ),
        checkpoint_path='checkpoints/dinov2_vitb14_pretrain.pth',
        fine_tune_type='lora',
    ),
    task_cfg=dict(
        type='SparseCorrespondenceTask',
        img_size=image_size, 
    ),
    window_size=45
)


train_pipeline = [
    dict(type='LoadImage'),
    dict(type='RandomCrop', crop_size=(image_size, image_size)),
    dict(type='NormalAug'),

    dict(type='Exchange'),
    dict(type='PadKeyPoints', max_num=40),
    dict(type='ImageThreshold'),
    dict(type='ToTensor'),
    dict(type='Normalize')
]

test_pipeline = [
    dict(type='LoadImage'),
    dict(type='ResizeTransform', target_size=image_size),
    dict(type='PadKeyPoints', max_num=40),
    dict(type='ImageThreshold'),
    dict(type='ToTensor'),
    dict(type='Normalize')
]

dataset_type = 'PFPascalDataset'
data_root = './data'
batch_size = 8
num_workers = 8
train_dataloader = dict(
    batch_size=batch_size,
    num_workers=num_workers,
    shuffle=True,
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        split='trn',
        pipeline=train_pipeline
    )
)

val_dataloader = dict(
    batch_size=batch_size,
    shuffle=False,
    num_workers=num_workers,
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        split='val',
        pipeline=test_pipeline
    )
)

test_dataloader = dict(
    batch_size=batch_size,
    shuffle=False,
    num_workers=num_workers,
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        split='test',
        pipeline=test_pipeline
    )
)

