'''

Image preprocessing steps

'''

import torchvision as tv

def get_transforms(split, size, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]):
    # 使用了标准的mean和std
        
    if "train" in split:
        transform = tv.transforms.Compose(
            [
                tv.transforms.Resize((256, 256)),
                tv.transforms.RandomCrop((224, 224)),
                tv.transforms.RandomHorizontalFlip(0.5),
                tv.transforms.ToTensor(),
                tv.transforms.Normalize(mean=mean, std=std),
            ]
        )
    else:
        transform = tv.transforms.Compose(
            [
                tv.transforms.Resize((256, 256)),
                tv.transforms.CenterCrop((224, 224)),
                tv.transforms.ToTensor(),
                tv.transforms.Normalize(mean=mean, std=std),
            ]
        )

    return transform

