log_level = 'INFO'
interval = 10      # record logs every n iters
device = 'cuda'
max_epochs = 100
img_size = 256

optimizer = dict(type='Adam', lr=3e-5)
lr_scheduler = dict(type='StepLR', step_size=200, gamma=0.95)

metric=dict(
    type='CorrespondenceMetric', 
    alpha=0.1, 
    img_size=img_size
)

model = dict(
    type='CorrespondenceModelResNet',
    pair_image_encoder_cfg=dict(
        type='PairImageEncoderResNet',
        backbone_cfg=dict(
            type='ResNet',
            layers=[3, 4, 23, 3],
            checkpoint_path='/home/jinhl/.cache/torch/hub/checkpoints/resnet101-5d3b4d8f.pth'
        ),
    ),
    task_cfg=dict(
        type='SparseCorrespondenceTask',
        img_size=img_size
    ),
)

train_pipeline = [
    dict(type='LoadImage'),
    dict(type='RandomCrop', crop_size=(img_size, img_size)),
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
batch_size = 16
num_workers = 16
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

