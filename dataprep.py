from torchvision.datasets import ImageFolder
import numpy as np
from torch.utils.data import DataLoader, Subset
from pathlib import Path
import warnings
import utils.config as config
import torchvision.transforms as transforms 

opt = config.get_arguments().parse_args()

def get_transform(opt, dataset=opt.dataset, pretensor_transform=True): # for transforming images
    transforms_list = []

    # Normalization factors
    # We use the mean and standard deviation of the CIFAR-10 dataset
    # as an example.
    # The values for CINIC-10 are also provided.
    # You can change these values later as needed. 

    cifar_mean = [0.4914, 0.4822, 0.4465]
    cifar_std = [0.247, 0.243, 0.261]

    cinic_mean = [0.47889522, 0.47227842, 0.43047404]
    cinic_std = [0.24205776, 0.23828046, 0.25874835]

    if pretensor_transform:
        transforms_list.append(transforms.ToTensor())
    transforms_list.append(transforms.Resize((opt.input_height, opt.input_width)))
    transforms_list.append(transforms.Normalize(cifar_mean, cifar_std))
    return transforms.Compose(transforms_list)

def dataloader(data_root: Path='data/', dataset: str=opt.dataset, 
               batch_size: int=opt.bs, num_workers: int=opt.num_workers, 
               split: float=0.0, train=True, transform=get_transform(opt)):
    '''
    Function to load the dataset
    @param data_root: str: Path to the data folder
    @param dataset: str: Name of the dataset
    @param batch_size: int: Batch size
    @param num_workers: int: Number of workers
    @param split: int: Portion of the dataset to use as training set. Default is 0
    @param train: bool: Whether to load the training set or test set. Default is True
    @param transform: torchvision.transforms: Transformations to apply to the dataset

    @return DataLoader: DataLoader object
    '''
    if dataset == 'breast_ultrasound':
        dataset = ImageFolder(data_root + 'breast_ultrasound', transform)
    elif dataset == 'skin_cancer':
        dataset = ImageFolder(data_root + 'skin_cancer', transform)
    elif dataset == 'brain_tumor':
        dataset = ImageFolder(data_root + 'brain_tumor', transform)
    else:
        warnings.warn('Dataset not available. Please check the dataset name.')
    
    if split > 0:
        split_id = int(len(dataset) * split)
        indices = np.arange(len(dataset))
        indices = np.random.permutation(indices)
        if train:
            dataset = Subset(dataset, indices[:split_id])
        else:
            dataset = Subset(dataset, indices[split_id:])
    return DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)