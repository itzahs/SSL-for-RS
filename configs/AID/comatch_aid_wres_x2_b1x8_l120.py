train = dict(
    eval_step=1024,
    total_steps=524288,
    trainer=dict(
        type='CoMatch',
        threshold=0.95,
        queue_batch=5,
        contrast_threshold=0.8,
        da_len=32,
        T=0.2,
        alpha=0.9,
        lambda_u=1.0,
        lambda_c=1.0,
        loss_x=dict(type='cross_entropy', reduction='mean')))
num_classes = 30
model = dict(
    type='wideresnet',
    depth=28,
    widen_factor=2,
    dropout=0,
    num_classes=30,
    proj=True,
    low_dim=64)
aid_mean = (0.485, 0.456, 0.406)
aid_std = (0.229, 0.224, 0.225)
data = dict(
    type='MyDataset',
    num_workers=4,
    num_labeled=120,
    num_classes=30,
    batch_size=8,
    expand_labels=False,
    mu=7,
    root='./data/AID',
    labeled_names_file='./data/AID/AID_train.txt',
    test_names_file='./data/AID/AID_test.txt',
    lpipelines=[[{
        'type': 'RandomHorizontalFlip',
        'p': 0.5
    }, {
        'type': 'RandomResizedCrop',
        'size': 224,
        'scale': (0.2, 1.0)
    }, {
        'type': 'ToTensor'
    }, {
        'type': 'Normalize',
        'mean': (0.485, 0.456, 0.406),
        'std': (0.229, 0.224, 0.225)
    }]],
    upipelinse=[[{
        'type': 'RandomHorizontalFlip'
    }, {
        'type': 'Resize',
        'size': 256
    }, {
        'type': 'CenterCrop',
        'size': 224
    }, {
        'type': 'ToTensor'
    }, {
        'type': 'Normalize',
        'mean': (0.485, 0.456, 0.406),
        'std': (0.229, 0.224, 0.225)
    }],
                [{
                    'type': 'RandomHorizontalFlip'
                }, {
                    'type': 'RandomResizedCrop',
                    'size': 224,
                    'scale': (0.2, 1.0)
                }, {
                    'type': 'RandAugmentMC',
                    'n': 2,
                    'm': 10
                }, {
                    'type': 'ToTensor'
                }, {
                    'type': 'Normalize',
                    'mean': (0.485, 0.456, 0.406),
                    'std': (0.229, 0.224, 0.225)
                }],
                [{
                    'type': 'RandomResizedCrop',
                    'size': 224,
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
        dict(type='Resize', size=256),
        dict(type='CenterCrop', size=224),
        dict(type='ToTensor'),
        dict(
            type='Normalize',
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225))
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
resume = './results_4pc/comatch/comatch_aid_wres_x2_b1x8_l120/checkpoint.pth.tar'
