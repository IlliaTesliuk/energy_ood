import random
import numpy as np
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

CIFAR_MEAN = [x / 255 for x in [125.3, 123.0, 113.9]]
CIFAR_STD = [x / 255 for x in [63.0, 62.1, 66.7]]

########################### Transforms ##############################

def get_transform(mean, std):
    trn = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    return trn

def get_resize_transform(mean, std, size=32):
    trn = transforms.Compose([
        transforms.Resize(size),
        transforms.CenterCrop(size),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    return trn

def get_cifar_train_transform():
    trn = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD)
    ])
    return trn

def get_cifar_test_transform():
    return get_transform(CIFAR_MEAN, CIFAR_STD)


########################## DATASETS ###########################

def get_cifar_train_valid(cifar_dir, transform_train, transform_valid, batch_size, seed):
    trainset = dset.CIFAR10(cifar_dir, train=True, transform=transform_train)
    validset = dset.CIFAR10(cifar_dir, train=True, transform=transform_valid)
        
    valid_size = 0.1
    num_train = len(trainset)
    indices = list(range(num_train))
    split = int(np.floor(valid_size * num_train))

    np.random.seed(seed)
    random.seed(seed)
    np.random.shuffle(indices)

    train_idx, valid_idx = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    trainloader = DataLoader(trainset, batch_size=batch_size, sampler=train_sampler, shuffle=False) # false because shuffled indices
    validloader = DataLoader(validset, batch_size=batch_size, sampler=valid_sampler, shuffle=False)

    print(f"[CIFAR10] Split seed={seed}")
    print(f"[CIFAR10] Train set: {len(train_sampler)} samples, {len(trainloader)} batches")
    print(f"[CIFAR10] Validation set:  {len(valid_sampler)} samples, {len(validloader)} batches")

    return trainloader, validloader


def get_cifar_generated(cifar_dir, transform, batch_size):
    gen_set = dset.ImageFolder(cifar_dir, transform=transform)
    gen_loader = DataLoader(gen_set, batch_size=batch_size, shuffle=False)
    print(f"[CIFAR10] Generated set: {len(gen_set)} samples, {len(gen_loader)} batches") 
    return gen_loader


def get_cifar_test(cifar_dir, transform, batch_size, shuffle):
    test_set = dset.CIFAR10(cifar_dir, train=False, transform=transform)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=shuffle)
    print(f"[CIFAR10][ID] Test set: {len(test_set)} samples, {len(test_loader)} batches")
    return test_loader


def get_svhn_test(svhn_dir, transform, batch_size, shuffle):
    test_set = dset.SVHN(svhn_dir, split='test', transform=transform)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=shuffle)
    print(f"[SVHN][OOD] Test set: {len(test_set)} samples, {len(test_loader)} batches") 
    return test_loader


def get_custom_test(dtd_dir, transform, batch_size, shuffle, name):
    test_set = dset.ImageFolder(dtd_dir, transform=transform)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=shuffle)
    print(f"[{name}][OOD] Test set: {len(test_set)} samples, {len(test_loader)} batches") 
    return test_loader


def get_test_dataset(ds_dir, batch_size, shuffle, name, mean=CIFAR_MEAN, std=CIFAR_STD):
    if name == "cifar_test":
        trn = get_cifar_test_transform()
        ds = get_cifar_test(ds_dir, trn, batch_size, shuffle)
    elif name == "svhn":
        trn = get_transform(mean, std) 
        ds = get_svhn_test(ds_dir, trn, batch_size, shuffle)
    else:
        if name in ["isun", "lsun-c" "lsun_r"]:
            trn = get_transform(mean, std)
        elif name in ["dtd","places"]:
            trn = get_resize_transform(mean, std)
        else:
            return None, None
        ds = get_custom_test(ds_dir, trn, batch_size, shuffle, name.upper())
    return ds, trn
    