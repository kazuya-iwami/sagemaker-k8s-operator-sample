import sagemaker
import torch
import torchvision
import torchvision.transforms as transforms

def _get_transform():
    return transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

def get_train_data_loader(data_dir):
    transform = _get_transform()
    trainset = torchvision.datasets.CIFAR10(root=data_dir, train=True,
                                            download=True, transform=transform)
    return torch.utils.data.DataLoader(trainset, batch_size=4,
                                       shuffle=True, num_workers=2)

def get_test_data_loader(data_dir):
    transform = _get_transform()
    testset = torchvision.datasets.CIFAR10(root=data_dir, train=False,
                                           download=True, transform=transform)
    return torch.utils.data.DataLoader(testset, batch_size=4,
                                       shuffle=False, num_workers=2)

get_train_data_loader('cifar-10-data')
get_test_data_loader('cifar-10-data')

sess = sagemaker.Session()
data_location = sess.upload_data(
    'cifar-10-data', key_prefix='pytorch-cifar10')


