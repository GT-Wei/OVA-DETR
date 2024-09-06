_base_ = [
    '../_base_/default_runtime.py'
]
backend_args = None
load_from = './pretrain_model/rtdetr_r50vd_8xb2-72e_coco_ff87da1a.pth'  # noqa
work_dir = 'work_dirs/'
# work_dir = 'work_dirs/First_State/modified_permute'


model = dict(
    type='RTDETR',
    num_queries=500,  # num_matching_queries, 900 for DINO
    with_box_refine=True,
    as_two_stage=True,
    data_preprocessor=dict(
        type='RTDETRDetDataPreprocessor',
        batch_augments=[
            dict(
                type='BatchSyncRandomResize',
                interval=1,
                interpolations=['nearest', 'bilinear', 'bicubic', 'area'],
                random_sizes=[
                    800
                ]
                )
        ],
        mean=[0, 0, 0],  # [123.675, 116.28, 103.53] for DINO
        std=[255, 255, 255],  # [58.395, 57.12, 57.375] for DINO
        bgr_to_rgb=True,
        pad_size_divisor=1),
    backbone=dict(
        type='Clip_rtdetr_Backbone',
        image_model=dict(
            type='CLIPModifiedResNet',  # ResNet for DINO
            layers=[3, 4, 6, 3],
            model_name='RN50',
            width=64,
            pretrained=True,
            frozen=False,
            load_openai=True
        ),
        text_model=dict(
            type='TextTransformer',
            model_name='RN50',
            context_length=77,
            vocab_size=49408,
            width=512,
            heads=8,
            layers=12,
            output_dim=1024,
            frozen_modules=['all'],
            frozen=True,
            load_openai=True)
        ),
    neck=dict(
        type='RT_detr_ChannelMapper',
        in_channels=[512, 1024, 2048],
        kernel_size=1,
        out_channels=256,
        act_cfg=None,
        norm_cfg=dict(type='SyncBN', requires_grad=True),  # GN for DINO
        num_outs=3,
        neck_fusion=True,
        txt_dims=1024),  # 4 for DINO
    encoder=dict(
        use_encoder_idx=[2],
        num_encoder_layers=1,
        in_channels=[256, 256, 256],
        fpn_cfg=dict(
            type='RTDETRFPN',
            in_channels=[256, 256, 256],
            out_channels=256,
            expansion=1.0,
            norm_cfg=dict(type='SyncBN', requires_grad=True)),
        layer_cfg=dict(
            self_attn_cfg=dict(embed_dims=256, num_heads=8, dropout=0.0),
            ffn_cfg=dict(
                embed_dims=256,
                feedforward_channels=1024,  # 2048 for DINO
                ffn_drop=0.0,
                act_cfg=dict(type='GELU')))),  # ReLU for DINO
    decoder=dict(
        txt_dims=1024,
        num_layers=6,
        return_intermediate=True,
        layer_cfg=dict(
            self_attn_cfg=dict(embed_dims=256, num_heads=8, dropout=0.0),
            cross_attn_cfg=dict(
                embed_dims=256,
                num_levels=3,  # 4 for DINO
                dropout=0.0),
            ffn_cfg=dict(
                embed_dims=256,
                feedforward_channels=1024,  # 2048 for DINO
                ffn_drop=0.0)),
        post_norm_cfg=None),
    bbox_head=dict(
        type='RTDETRHead',
        num_classes=80,
        txt_dims=1024,
        embed_dims=256,
        sync_cls_avg_factor=True,
        loss_cls=dict(
            type='RTDETRVarifocalLoss',  # FocalLoss in DINO
            use_sigmoid=True,
            alpha=0.75,
            gamma=2.0,
            iou_weighted=True,
            loss_weight=1.0),
        loss_bbox=dict(type='L1Loss', loss_weight=5.0),
        loss_iou=dict(type='GIoULoss', loss_weight=2.0)),
    dn_cfg=dict(  # TODO: Move to model.train_cfg ?
        label_noise_scale=0.5,
        box_noise_scale=1.0,
        group_cfg=dict(dynamic=True, num_groups=None,
                       num_dn_queries=100)),  # TODO: half num_dn_queries
    load_clip_name='RN50',
    # training and testing settings
    train_cfg=dict(
        assigner=dict(
            type='HungarianAssigner',
            match_costs=[
                dict(type='FocalLossCost', weight=2.0),
                dict(type='BBoxL1Cost', weight=5.0, box_format='xywh'),
                dict(type='IoUCost', iou_mode='giou', weight=2.0)
            ])),
    test_cfg=dict(max_per_img=500))

# train_pipeline, NOTE the img_scale and the Pad's size_divisor is different
# from the default setting in mmdet.
interpolations = ['nearest', 'bilinear', 'bicubic', 'area', 'lanczos']
train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='RandomApply',
        transforms=dict(type='PhotoMetricDistortion'),
        prob=0.5),
    # dict(type='TorchvisionRandomZoomOut', fill=0, side_range=(1.0, 4.0), p=0.5),
    # dict(type='Expand', mean=[0, 0, 0]),
    dict(
        type='Expand',
        mean=[123.675, 116.28, 103.53],
        to_rgb=True,
        ratio_range=(1, 2)),
    dict(
        type='RandomApply', transforms=dict(type='MinIoURandomCrop'),
        prob=0.8),
    dict(type='FilterAnnotations', min_gt_bbox_wh=(1, 1), keep_empty=False),
    dict(type='RandomFlip', prob=0.5),
    dict(
        type='RandomChoice',
        transforms=[[
            dict(
                type='Resize',
                scale=(800, 800),
                keep_ratio=False,
                interpolation=interpolation)
        ] for interpolation in interpolations]),
    dict(type='FilterAnnotations', min_gt_bbox_wh=(1, 1), keep_empty=False),
    dict(type='RandomLoadText',
         num_neg_samples=(48, 48),
         max_num_samples=48,
         padding_to_max=True,
         padding_value=''),
    # dict(type='LoadText'),
    dict(type='PackDetInputs',
         meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                            'scale_factor', 'flip', 'flip_direction', 'texts'))
]
test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(
        type='Resize',
        scale=(800, 800),
        keep_ratio=False,
        interpolation='bicubic'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='LoadText'),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor', 'texts'))
]


# ----------------------------------------------dota
dota_dataset_type = 'DotaDataset'
dota_zero_shot_dataset_type = "ZSDDotaDataset"
dota_zero_shot_dataset_type_unseen = "ZSDDotaDataset_Unseen"
dota_data_root = '/home/disk/datasets/DOTA_split/'
dota_train_dataset = dict(
    type='MultiModalDataset',
    dataset=dict(
        type=dota_dataset_type,
        data_root=dota_data_root,
        ann_file='annotations/DOTA_train_seen_only.json',
        data_prefix=dict(img='train_seen_only/'),
        filter_cfg=dict(filter_empty_gt=False, min_size=32)),
    class_text_path='class_text/texts/dota_seen_class_texts.json',
    pipeline=train_pipeline)
dota_val_dataset = dict(
    type='MultiModalDataset',
    dataset=dict(
        type=dota_zero_shot_dataset_type,
        data_root=dota_data_root,
        ann_file='annotations/DOTA_val_zeros_shot_final_test.json',
        data_prefix=dict(img='val/images/'),
        filter_cfg=dict(filter_empty_gt=False, min_size=32)),
    class_text_path='class_text/texts/dota_test_class_texts.json',
    pipeline=test_pipeline)
dota_val_evaluator = dict(
    type='CocoMetric',
    ann_file=dota_data_root + 'annotations/DOTA_val_zeros_shot_final_test.json',
    metric='bbox',
    format_only=False,
    classwise=True,
    backend_args=backend_args)
dota_val_dataset_zsd = dict(
    type='MultiModalDataset',
    dataset=dict(
        type=dota_zero_shot_dataset_type_unseen,
        data_root=dota_data_root,
        ann_file='annotations/DOTA_val_unseen_only.json',
        data_prefix=dict(img='val/images/'),
        filter_cfg=dict(filter_empty_gt=True, min_size=32)),
    class_text_path='class_text/texts/dota_test_class_texts_zsd.json',
    pipeline=test_pipeline)
dota_val_evaluator_zsd = dict(
    type='CocoMetric',
    ann_file=dota_data_root + 'annotations/DOTA_val_unseen_only.json',
    metric='bbox',
    format_only=False,
    classwise=True,
    backend_args=backend_args)
# ----------------------------------------------


# ----------------------------------------------DIOR
dior_dataset_type = 'DIORDataset'
dior_zero_shot_dataset_type = "ZSDDIORDataset"
dior_zero_shot_dataset_type_unseen = "ZSDDIORDataset_Unseen"
dior_data_root = '/home/disk/datasets/DIOR_coco/'
dior_train_dataset = dict(
    type='MultiModalDataset',
    dataset=dict(
        type=dior_dataset_type,
        data_root=dior_data_root,
        ann_file='annotations/dior_seen_train.json',
        data_prefix=dict(img='seen_images/'),
        filter_cfg=dict(filter_empty_gt=False, min_size=32)),
    class_text_path='class_text/texts/dior_seen_class_texts.json',
    pipeline=train_pipeline)
dior_val_dataset = dict(
    type='MultiModalDataset',
    dataset=dict(
        type=dior_zero_shot_dataset_type,
        data_root=dior_data_root,
        ann_file='annotations/dior_zsd_final_val.json',
        data_prefix=dict(img='val/'),
        filter_cfg=dict(filter_empty_gt=False, min_size=32)),
    class_text_path='class_text/texts/dior_test_class_texts.json',
    pipeline=test_pipeline)
dior_val_evaluator = dict(
    type='CocoMetric',
    ann_file=dior_data_root + 'annotations/dior_zsd_final_val.json',
    metric='bbox',
    format_only=False,
    classwise=True,
    backend_args=backend_args)
dior_val_dataset_zsd = dict(
    type='MultiModalDataset',
    dataset=dict(
        type=dior_zero_shot_dataset_type_unseen,
        data_root=dior_data_root,
        ann_file='annotations/dior_unseen_val.json',
        data_prefix=dict(img='val/'),
        filter_cfg=dict(filter_empty_gt=True, min_size=32)),
    class_text_path='class_text/texts/dior_test_class_texts_zsd.json',
    pipeline=test_pipeline)
dior_val_evaluator_zsd = dict(
    type='CocoMetric',
    ann_file=dior_data_root + 'annotations/dior_unseen_val.json',
    metric='bbox',
    format_only=False,
    classwise=True,
    backend_args=backend_args)
# ----------------------------------------------


# ----------------------------------------------DIOR
xview_train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args, imdecode_backend="tifffile"),  # 正确修改 backend_args
    *train_pipeline[1:]  # 添加除第一个元素之外的其他所有原始元素
]
xview_test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args, imdecode_backend="tifffile"),  # 正确修改 backend_args
    *test_pipeline[1:]  # 添加除第一个元素之外的其他所有原始元素
]
xview_dataset_type = 'XViewDataset'
xview_zero_shot_dataset_type = "ZSDXViewDataset"
xview_zero_shot_dataset_type_unseen = "ZSDXViewDataset_Unseen"
xview_data_root = '/home/disk/datasets/xView_split/'
xview_train_dataset = dict(
    type='MultiModalDataset',
    dataset=dict(
        type=xview_dataset_type,
        data_root=xview_data_root,
        ann_file='annotations/xView_train_seen_only.json',
        data_prefix=dict(img='train_seen_only/'),
        filter_cfg=dict(filter_empty_gt=False, min_size=32)),
    class_text_path='class_text/texts/xview_seen_class_texts_detail.json',
    pipeline=xview_train_pipeline)
xview_val_dataset = dict(
    type='MultiModalDataset',
    dataset=dict(
        type=xview_zero_shot_dataset_type,
        data_root=xview_data_root,
        ann_file='annotations/xView_val_zeros_shot_final_test.json',
        data_prefix=dict(img='val/images'),
        filter_cfg=dict(filter_empty_gt=False, min_size=32)),
    class_text_path='class_text/texts/xview_test_class_texts_detail.json',
    pipeline=xview_test_pipeline)
xview_val_evaluator = dict(
    type='CocoMetric',
    ann_file=xview_data_root + 'annotations/xView_val_zeros_shot_final_test.json',
    metric='bbox',
    format_only=False,
    classwise=True,
    backend_args=backend_args)
xview_val_dataset_zsd = dict(
    type='MultiModalDataset',
    dataset=dict(
        type=xview_zero_shot_dataset_type_unseen,
        data_root=xview_data_root,
        ann_file='annotations/Xview_val_unseen_only.json',
        data_prefix=dict(img='val/images'),
        filter_cfg=dict(filter_empty_gt=True, min_size=32)),
    class_text_path='class_text/texts/xview_test_class_texts_detail_zsd.json',
    pipeline=xview_test_pipeline)
xview_val_evaluator_zsd = dict(
    type='CocoMetric',
    ann_file=xview_data_root + 'annotations/Xview_val_unseen_only.json',
    metric='bbox',
    format_only=False,
    classwise=True,
    backend_args=backend_args)
# ----------------------------------------------


train_dataloader = dict(
    batch_size=2,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    batch_sampler=dict(type='AspectRatioBatchSampler'),
    # dataset=dota_train_dataset
    dataset=dict(
                 type='ConcatDataset', 
                 datasets=[dior_train_dataset,
                           dota_train_dataset,
                           xview_train_dataset]))

val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(type='ConcatDataset', 
                 datasets=[dior_val_dataset,
                           dior_val_dataset_zsd,
                            dota_val_dataset,
                            dota_val_dataset_zsd,
                            xview_val_dataset,
                            xview_val_dataset_zsd,
                            ]))
test_dataloader = val_dataloader


val_evaluator = dict(
    type='MultiDatasetsEvaluator',
    metrics=[dior_val_evaluator,
             dior_val_evaluator_zsd,
             dota_val_evaluator,
             dota_val_evaluator_zsd, 
             xview_val_evaluator,
             xview_val_evaluator_zsd],
    dataset_prefixes=['dior_val', 
                      'dior_val_zsd',
                      'dota_val',
                      'dota_val_zsd',
                      'xview_val',
                      'xview_val_zsd'])
test_evaluator = val_evaluator


# optimizer
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=0.0001, weight_decay=0.0001),
    clip_grad=dict(max_norm=0.1, norm_type=2),
    paramwise_cfg=dict(
        custom_keys={'backbone': dict(lr_mult=0.1)},
        norm_decay_mult=0,
        bypass_duplicate=True))

# learning policy
max_epochs = 80
train_cfg = dict(
    type='EpochBasedTrainLoop', max_epochs=max_epochs, val_interval=5)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

param_scheduler = [
    dict(
        type='LinearLR', start_factor=0.001, by_epoch=False, begin=0, end=2000)
]

# NOTE: `auto_scale_lr` is for automatically scaling LR,
# USER SHOULD NOT CHANGE ITS VALUES.
# base_batch_size = (4 GPUs) x (4 samples per GPU)
auto_scale_lr = dict(base_batch_size=16)
# base_batch_size = (8 GPUs) x (4 samples per GPU)
# auto_scale_lr = dict(base_batch_size=32)

custom_hooks = [
    dict(
        type='EMAHook',
        ema_type='ExpMomentumEMA',
        momentum=0.0001,
        update_buffers=True,
        priority=49)
]
find_unused_parameters = True
