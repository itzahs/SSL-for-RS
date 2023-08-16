train = dict(
    eval_step=1024,
    total_steps=1048576,
    trainer=dict(
        type='Classifier', loss_x=dict(type='cross_entropy',
                                       reduction='mean')))
num_classes = 21
model = dict(
    type='wideresnet', depth=28, widen_factor=2, dropout=0, num_classes=21)
ucm_mean = (0.485, 0.456, 0.406)
ucm_std = (0.229, 0.224, 0.225)
data = dict(
    type='MyDataset',
    num_workers=4,
    num_labeled=84,
    num_classes=21,
    batch_size=8,
    expand_labels=False,
    mu=7,
    root='./data/UCM/Images',
    labeled_names_file='./data/UCM/Images/UCM_train.txt',
    test_names_file='./data/UCM/Images/UCM_test.txt',
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
    num_training_steps=1048576)
ema = dict(use=True, pseudo_with_ema=False, decay=0.999)
amp = dict(use=False, opt_level='O1')
log = dict(interval=1)
ckpt = dict(interval=1000)
optimizer = dict(
    type='SGD', lr=0.03, momentum=0.9, weight_decay=0.001, nesterov=True)
resume = ''
