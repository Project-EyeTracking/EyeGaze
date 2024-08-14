
import os
import numpy as np
import torch
import torchvision
from torchvision import transforms

from .model import L2CS

transformations = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(448),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


def prep_input_numpy(img:np.ndarray, device:str):
    """Preparing a Numpy Array as input to L2CS-Net."""

    if len(img.shape) == 4:
        imgs = []
        for im in img:
            imgs.append(transformations(im))
        img = torch.stack(imgs)
    else:
        img = transformations(img)

    img = img.to(device)

    if len(img.shape) == 3:
        img = img.unsqueeze(0)

    return img


def select_device(device=''):
    device = device.lower()

    if device == 'cpu':
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
        return torch.device('cpu')
    elif device == 'cuda':
        if torch.cuda.is_available():
            return torch.device('cuda:0')
        else:
            raise ValueError("CUDA is not available. Please choose another device.")
    else:
        # Fallback to automatic selection
        if torch.cuda.is_available():
            return torch.device('cuda:0')
        else:
            return torch.device('cpu')


def getArch(arch, bins):
    # Base network structure
    if arch == 'ResNet18':
        model = L2CS( torchvision.models.resnet.BasicBlock,[2, 2,  2, 2], bins)
    elif arch == 'ResNet34':
        model = L2CS( torchvision.models.resnet.BasicBlock,[3, 4,  6, 3], bins)
    elif arch == 'ResNet101':
        model = L2CS( torchvision.models.resnet.Bottleneck,[3, 4, 23, 3], bins)
    elif arch == 'ResNet152':
        model = L2CS( torchvision.models.resnet.Bottleneck,[3, 8, 36, 3], bins)
    else:
        if arch != 'ResNet50':
            print('Invalid value for architecture is passed! '
                'The default value of ResNet50 will be used instead!')
        model = L2CS( torchvision.models.resnet.Bottleneck, [3, 4, 6,  3], bins)
    return model
