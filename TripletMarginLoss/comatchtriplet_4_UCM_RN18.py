train = dict(
    eval_step=1024,
    total_steps=524288,
    trainer=dict(
        type='CoMatchTriplet',
        threshold=0.95,
        queue_batch=5,
        contrast_threshold=0.8,
        da_len=32,
        T=0.2,
        alpha=0.9,
        lambda_u=1.0,
        lambda_c=1.0,
        lambda_dml=1.0,
        loss_x=dict(type='cross_entropy', reduction='mean')))
num_classes = 10
model = dict(
    type='resnet18',
    width=1,
    in_channel=3,
    num_class=10,
    proj=True,
    low_dim=64)
ucm_mean = [0.485, 0.456, 0.406]
ucm_std = [0.229, 0.224, 0.225]
data = dict(
    type='MyDataset',
    num_workers=4,
    num_labeled=40,
    num_classes=10,
    batch_size=32,
    expand_labels=False,
    mu=7,
    root='/scratch/isequeir/ssl/CCSSL/data',
    labeled_names_file=
    '/scratch/isequeir/ssl/CCSSL/data/UCM/Images/UCM_train.txt',
    test_names_file='/scratch/isequeir/ssl/CCSSL/data/UCM/Images/UCM_test.txt',
    lpipelines=[[{
        'type': 'Resize',
        'size': 64
    }, {
        'type': 'RandomHorizontalFlip',
        'p': 0.5
    }, {
        'type': 'RandomResizedCrop',
        'size': 60,
        'scale': (0.2, 1.0)
    }, {
        'type': 'ToTensor'
    }, {
        'type': 'Normalize',
        'mean': [0.485, 0.456, 0.406],
        'std': [0.229, 0.224, 0.225]
    }]],
    upipelinse=[[{
        'type': 'Resize',
        'size': 64
    }, {
        'type': 'RandomHorizontalFlip'
    }, {
        'type': 'CenterCrop',
        'size': 60
    }, {
        'type': 'ToTensor'
    }, {
        'type': 'Normalize',
        'mean': [0.485, 0.456, 0.406],
        'std': [0.229, 0.224, 0.225]
    }],
                [{
                    'type': 'Resize',
                    'size': 64
                }, {
                    'type': 'RandomHorizontalFlip'
                }, {
                    'type': 'RandomResizedCrop',
                    'size': 60,
                    'scale': (0.2, 1.0)
                }, {
                    'type': 'RandAugmentMC',
                    'n': 2,
                    'm': 10
                }, {
                    'type': 'ToTensor'
                }, {
                    'type': 'Normalize',
                    'mean': [0.485, 0.456, 0.406],
                    'std': [0.229, 0.224, 0.225]
                }],
                [{
                    'type': 'Resize',
                    'size': 64
                }, {
                    'type': 'RandomResizedCrop',
                    'size': 60,
                    'scale': (0.2, 1.0)
                }, {
                    'type': 'RandomHorizontalFlip'
                }, {
                    'type':
                    'RandomApply',
                    'transforms': [{
                        'type': 'ColorJitter',
                        'brightness': 0.4,
                        'contrast': 0.4,
                        'saturation': 0.4,
                        'hue': 0.1
                    }],
                    'p':
                    0.8
                }, {
                    'type': 'RandomGrayscale',
                    'p': 0.2
                }, {
                    'type': 'ToTensor'
                }]],
    vpipeline=[
        dict(type='Resize', size=64),
        dict(type='CenterCrop', size=60),
        dict(type='ToTensor'),
        dict(
            type='Normalize',
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225])
    ],
    eval_step=1024)
scheduler = dict(
    type='cosine_schedule_with_warmup',
    num_warmup_steps=0,
    num_training_steps=524288)
ema = dict(use=True, pseudo_with_ema=False, decay=0.999)
amp = dict(use=False, opt_level='O1')
log = dict(interval=1)
ckpt = dict(interval=1)
optimizer = dict(
    type='SGD', lr=0.03, momentum=0.9, weight_decay=0.0005, nesterov=True)
resume = ''
