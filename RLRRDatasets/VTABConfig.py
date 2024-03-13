import ml_collections


def get_caltech101_config():
    config = ml_collections.ConfigDict()
    config.path = '/public/home/sqliu/zstar/vtab-1k/caltech101'
    config.num_classes = 102
    config.CropSize = 224
    config.num_workers = 6
    config.pin_memory = True
    config.num_gpus = 1
    config.batch_size = 32

    config.simple_aug = False
    config.lr = 1e-3
    config.drop_path = 0.1
    config.min_lr = 1e-8
    config.warmup_lr = 1e-7
    config.weight_decay = 5e-5
    return config


def get_cifar_config():
    config = ml_collections.ConfigDict()
    config.path = '/public/home/sqliu/zstar/vtab-1k/cifar'
    config.num_classes = 100
    config.CropSize = 224
    config.num_workers = 6
    config.pin_memory = True
    config.num_gpus = 1
    config.batch_size = 32
    
    config.simple_aug = False
    config.lr = 3e-3
    config.drop_path = 0.1
    config.min_lr = 1e-8
    config.warmup_lr = 1e-7
    config.weight_decay = 5e-5
    return config


def get_dtd_config():
    config = ml_collections.ConfigDict()
    config.path = '/public/home/sqliu/zstar/vtab-1k/dtd'
    config.num_classes = 47
    config.CropSize = 224
    config.num_workers = 6
    config.pin_memory = True
    config.num_gpus = 1
    config.batch_size = 32

    config.simple_aug = True
    config.lr = 2e-3
    config.drop_path = 0.1
    config.min_lr = 1e-8
    config.warmup_lr = 1e-7
    config.weight_decay = 5e-5
    return config


def get_oxford_flowers102_config():
    config = ml_collections.ConfigDict()
    config.path = '/public/home/sqliu/zstar/vtab-1k/oxford_flowers102'
    config.num_classes = 102
    config.CropSize = 224
    config.num_workers = 6
    config.pin_memory = True
    config.num_gpus = 1
    config.batch_size = 32
    
    config.simple_aug = False
    config.lr = 3e-3
    config.drop_path = 0.3
    config.min_lr = 1e-8
    config.warmup_lr = 1e-7
    config.weight_decay = 5e-2
    return config


def get_oxford_iiit_pet_config():
    config = ml_collections.ConfigDict()
    config.path = '/public/home/sqliu/zstar/vtab-1k/oxford_iiit_pet'
    config.num_classes = 37
    config.CropSize = 224
    config.num_workers = 6
    config.pin_memory = True
    config.num_gpus = 1
    config.batch_size = 32
    
    config.simple_aug = True
    config.lr = 1e-3
    config.drop_path = 0.1
    config.min_lr = 1e-8
    config.warmup_lr = 1e-7
    config.weight_decay = 5e-5
    return config

def get_svhn_config():
    config = ml_collections.ConfigDict()
    config.path = '/public/home/sqliu/zstar/vtab-1k/svhn'
    config.num_classes = 10
    config.CropSize = 224
    config.num_workers = 6
    config.pin_memory = True
    config.num_gpus = 1
    config.batch_size = 32
    
    config.simple_aug = False
    config.lr = 5e-3
    config.drop_path = 0.3
    config.min_lr = 1e-8
    config.warmup_lr = 1e-7
    config.weight_decay = 5e-5
    return config


def get_sun397_config():
    config = ml_collections.ConfigDict()
    config.path = '/public/home/sqliu/zstar/vtab-1k/sun397'
    config.num_classes = 397
    config.CropSize = 224
    config.num_workers = 6
    config.pin_memory = True
    config.num_gpus = 1
    config.batch_size = 32
    
    config.simple_aug = False
    config.lr = 1e-3
    config.drop_path = 0.1
    config.min_lr = 1e-8
    config.warmup_lr = 1e-7
    config.weight_decay = 5e-5
    return config


def get_patch_camelyon_config():
    config = ml_collections.ConfigDict()
    config.path = '/public/home/sqliu/zstar/vtab-1k/patch_camelyon'
    config.num_classes = 2
    config.CropSize = 224
    config.num_workers = 6
    config.pin_memory = True
    config.num_gpus = 1
    config.batch_size = 32

    config.simple_aug = True
    config.lr = 3e-3
    config.drop_path = 0.3
    config.min_lr = 1e-8
    config.warmup_lr = 1e-7
    config.weight_decay = 5e-2
    return config


def get_eurosat_config():
    config = ml_collections.ConfigDict()
    config.path = '/public/home/sqliu/zstar/vtab-1k/eurosat'
    config.num_classes = 10
    config.CropSize = 224
    config.num_workers = 6
    config.pin_memory = True
    config.num_gpus = 1
    config.batch_size = 32
    
    config.simple_aug = True
    config.lr = 1e-3
    config.drop_path = 0.1
    config.min_lr = 1e-8
    config.warmup_lr = 1e-7
    config.weight_decay = 5e-5
    return config


def get_resisc45_config():
    config = ml_collections.ConfigDict()
    config.path = '/public/home/sqliu/zstar/vtab-1k/resisc45'
    config.num_classes = 45
    config.CropSize = 224
    config.num_workers = 6
    config.pin_memory = True
    config.num_gpus = 1
    config.batch_size = 32
    
    config.simple_aug = False
    config.lr = 3e-3
    config.drop_path = 0.0
    config.min_lr = 1e-8
    config.warmup_lr = 1e-7
    config.weight_decay = 5e-5
    return config


def get_diabetic_retinopathy_config():
    config = ml_collections.ConfigDict()
    config.path = '/public/home/sqliu/zstar/vtab-1k/diabetic_retinopathy'
    config.num_classes = 5
    config.CropSize = 224
    config.num_workers = 6
    config.pin_memory = True
    config.num_gpus = 1
    config.batch_size = 32

    config.simple_aug = True
    config.lr = 5e-3
    config.drop_path = 0.1
    config.min_lr = 1e-8
    config.warmup_lr = 1e-7
    config.weight_decay = 5e-5
    return config

def get_clevr_count_config():
    config = ml_collections.ConfigDict()
    config.path = '/public/home/sqliu/zstar/vtab-1k/clevr_count'
    config.num_classes = 8
    config.CropSize = 224
    config.num_workers = 6
    config.pin_memory = True
    config.num_gpus = 1
    config.batch_size = 32
    
    config.simple_aug = True
    config.lr = 5e-3
    config.drop_path = 0.5
    config.min_lr = 1e-8
    config.warmup_lr = 1e-7
    config.weight_decay = 5e-5
    return config

def get_clevr_dist_config():
    config = ml_collections.ConfigDict()
    config.path = '/public/home/sqliu/zstar/vtab-1k/clevr_dist'
    config.num_classes = 6
    config.CropSize = 224
    config.num_workers = 4
    config.pin_memory = True
    config.num_gpus = 1
    config.batch_size = 32
    
    config.simple_aug = True
    config.lr = 1e-2
    config.drop_path = 0.2
    config.min_lr = 1e-8
    config.warmup_lr = 1e-7
    config.weight_decay = 5e-5
    return config


def get_dmlab_config():
    config = ml_collections.ConfigDict()
    config.path = '/public/home/sqliu/zstar/vtab-1k/dmlab'
    config.num_classes = 6
    config.CropSize = 224
    config.num_workers = 6
    config.pin_memory = True
    config.num_gpus = 1
    config.batch_size = 32
    
    config.simple_aug = True
    config.lr = 5e-3
    config.drop_path = 0.3
    config.min_lr = 1e-8
    config.warmup_lr = 1e-7
    config.weight_decay = 5e-5
    return config


def get_kitti_config():
    config = ml_collections.ConfigDict()
    config.path = '/public/home/sqliu/zstar/vtab-1k/kitti'
    config.num_classes = 4
    config.CropSize = 224
    config.num_workers = 6
    config.pin_memory = True
    config.num_gpus = 1
    config.batch_size = 32
    
    config.simple_aug = True
    config.lr = 3e-3
    config.drop_path = 0.3
    config.min_lr = 1e-8
    config.warmup_lr = 1e-7
    config.weight_decay = 5e-2
    return config

def get_dsprites_loc_config():
    config = ml_collections.ConfigDict()
    config.path = '/public/home/sqliu/zstar/vtab-1k/dsprites_loc'
    config.num_classes = 16
    config.CropSize = 224
    config.num_workers = 6
    config.pin_memory = True
    config.num_gpus = 1
    config.batch_size = 32
    
    config.simple_aug = False
    config.lr = 1e-2
    config.drop_path = 0.1
    config.min_lr = 1e-8
    config.warmup_lr = 1e-7
    config.weight_decay = 5e-5
    return config


def get_dsprites_ori_config():
    config = ml_collections.ConfigDict()
    config.path = '/public/home/sqliu/zstar/vtab-1k/dsprites_ori'
    config.num_classes = 16
    config.CropSize = 224
    config.num_workers = 6
    config.pin_memory = True
    config.num_gpus = 1
    config.batch_size = 32
    
    config.simple_aug = False
    config.lr = 5e-3
    config.drop_path = 0.1
    config.min_lr = 1e-8
    config.warmup_lr = 1e-7
    config.weight_decay = 5e-5
    return config


def get_smallnorb_azi_config():
    config = ml_collections.ConfigDict()
    config.path = '/public/home/sqliu/zstar/vtab-1k/smallnorb_azi'
    config.num_classes = 18
    config.CropSize = 224
    config.num_workers = 6
    config.pin_memory = True
    config.num_gpus = 1
    config.batch_size = 32

    config.simple_aug = False
    config.lr = 5e-3
    config.drop_path = 0.3
    config.min_lr = 1e-8
    config.warmup_lr = 1e-7
    config.weight_decay = 5e-2
    return config


def get_smallnorb_ele_config():
    config = ml_collections.ConfigDict()
    config.path = '/public/home/sqliu/zstar/vtab-1k/smallnorb_ele'
    config.num_classes = 9
    config.CropSize = 224
    config.num_workers = 6
    config.pin_memory = True
    config.num_gpus = 1
    config.batch_size = 32
    
    config.simple_aug = True
    config.lr = 1e-3
    config.drop_path = 0.3
    config.min_lr = 1e-8
    config.warmup_lr = 1e-7
    config.weight_decay = 5e-5
    return config


DATA_CONFIGS = {
    "caltech101": get_caltech101_config(),
    "cifar": get_cifar_config(),
    "clevr_count": get_clevr_count_config(),
    "clevr_dist": get_clevr_dist_config(),
    "diabetic_retinopathy": get_diabetic_retinopathy_config(),
    "dmlab": get_dmlab_config(),
    "dsprites_loc": get_dsprites_loc_config(),
    "dsprites_ori": get_dsprites_ori_config(),
    "dtd": get_dtd_config(),
    "eurosat": get_eurosat_config(),
    "kitti": get_kitti_config(),
    "oxford_flowers102": get_oxford_flowers102_config(),
    "oxford_iiit_pet": get_oxford_iiit_pet_config(),
    "patch_camelyon": get_patch_camelyon_config(),
    "resisc45": get_resisc45_config(),
    "smallnorb_azi": get_smallnorb_azi_config(),
    "smallnorb_ele": get_smallnorb_ele_config(),
    "sun397": get_sun397_config(),
    "svhn": get_svhn_config()
}