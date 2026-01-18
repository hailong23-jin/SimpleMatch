log_level = 'INFO'
interval = 10      # record logs every n iters
device = 'cuda'
max_epochs = 100
image_size = 252

optimizer = dict(type='Adam', lr=6e-6)

metric=dict(
    type='CorrespondenceMetric', 
    alpha=0.1, 
    img_size=image_size,
    down_factor=4
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
        type='SparseFlowTask',
        img_size=image_size, 
        down_factor=14
    ),
)

dataset_type = 'PFWillowDataset'
data_root = './data'
batch_size = 8
num_workers = 8
test_pipeline = [
    dict(type='LoadImage'),
    dict(type='ResizeTransform', target_size=image_size),
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




