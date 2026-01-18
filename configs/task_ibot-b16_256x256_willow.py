

log_level = 'INFO'
interval = 10      # record logs every n iters
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

dataset_type = 'PFWillowDataset'
data_root = './data'
batch_size = 16
num_workers = 16
test_pipeline = [
    dict(type='LoadImage'),
    dict(type='ResizeTransform', target_size=img_size),
    dict(type='PadKeyPoints', max_num=40),
    dict(type='WILLOWThreshold'),
    dict(type='ToTensor'),
    dict(type='Normalize')
]

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

