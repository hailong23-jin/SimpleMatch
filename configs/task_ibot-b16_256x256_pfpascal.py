log_level = 'INFO'
interval = 10      # record logs every n iters
device = 'cuda'
max_epochs = 100
img_size = 256

optimizer = dict(type='Adam', lr=1e-5)
lr_scheduler = dict(type='StepLR', step_size=100, gamma=0.95)

metric=dict(
    type='CorrespondenceMetric', 
    alpha=0.1, 
    img_size=img_size
)

model = dict(
    type='CorrespondenceModel',
    pair_image_encoder_cfg=dict(
        type='PairImageEncoder',
        backbone_cfg=dict(
            type='iBOTVisionTransformer',
            patch_size=16,
            embed_dim=768,
            num_classes=1000,
            qkv_bias=True,
        ),
        fine_tune_type=None,
        checkpoint_path='checkpoints/ibot-b16.pth',
    ),

    task_cfg=dict(
        type='SparseCorrespondenceTask',
        img_size=img_size
    ),
    window_size=45,
)

train_pipeline = [
    dict(type='LoadImage'),
    dict(type='RandomCrop', crop_size=(img_size, img_size)),
    dict(type='RandomRotation', target_size=img_size),
    dict(type='NormalAug'),

    dict(type='Exchange'),
    dict(type='PadKeyPoints', max_num=30),
    dict(type='ImageThreshold'),
    dict(type='ToTensor'),
    dict(type='Normalize')
]

test_pipeline = [
    dict(type='LoadImage'),
    dict(type='ResizeTransform', target_size=img_size),
    dict(type='PadKeyPoints', max_num=30),
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

