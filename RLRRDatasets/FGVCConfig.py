import ml_collections

def get_dogs_config():
    config = ml_collections.ConfigDict()
    config.dataset_name = "StanfordDogs"
    config.num_classes = 120
    config.CropSize = 224
    config.num_workers = 6
    config.pin_memory = True
    config.num_gpus = 1
    config.batch_size = 32
    
    config.drop_last = False    
    config.lr = 1e-4
    config.drop_path = 0.0
    config.min_lr = 1e-8
    config.warmup_lr = 1e-7
    config.weight_decay = 5e-2
    return config

def get_flowers_config():
    config = ml_collections.ConfigDict()
    config.dataset_name = "OxfordFlowers"
    config.num_classes = 102
    config.CropSize = 224
    config.num_workers = 6
    config.pin_memory = True
    config.num_gpus = 1
    config.batch_size = 32

    config.drop_last = False
    config.lr = 3e-3
    config.drop_path = 0.2
    config.min_lr = 1e-8
    config.warmup_lr = 1e-7
    config.weight_decay = 5e-2
    return config

def get_CUB_config():
    config = ml_collections.ConfigDict()
    config.dataset_name = "CUB_200_2011"
    config.num_classes = 200
    config.CropSize = 224
    config.num_workers = 6
    config.pin_memory = True
    config.num_gpus = 1
    config.batch_size = 32

    config.drop_last = False
    config.lr = 3e-3
    config.drop_path = 0.1
    config.min_lr = 1e-8
    config.warmup_lr = 1e-7
    config.weight_decay = 5e-2
    return config

def get_Cars_config():
    config = ml_collections.ConfigDict()
    config.dataset_name = "StanfordCars"
    config.num_classes = 196
    config.CropSize = 224
    config.num_workers = 6
    config.pin_memory = True
    config.num_gpus = 1
    config.batch_size = 32
    
    config.drop_last = False
    config.lr = 3e-3
    config.drop_path = 0.2
    config.min_lr = 1e-8
    config.warmup_lr = 1e-7
    config.weight_decay = 5e-2
    return config


def get_NABirds_config():
    config = ml_collections.ConfigDict()
    config.dataset_name = "NABirds"
    config.num_classes = 555
    config.CropSize = 224
    config.num_workers = 6
    config.pin_memory = True
    config.num_gpus = 1
    config.batch_size = 32

    config.drop_last = True
    config.lr = 5e-4
    config.drop_path = 0.3
    config.min_lr = 1e-8
    config.warmup_lr = 1e-7
    config.weight_decay = 5e-2
    return config



DATA_CONFIGS ={
    "CUB_200_2011": get_CUB_config(),
    "NABirds": get_NABirds_config(),
    "OxfordFlowers": get_flowers_config(),
    "StanfordCars": get_Cars_config(),
    "StanfordDogs": get_dogs_config()
}