import logging

#from helpers.configuration_container import ConfigurationContainer
from torchvision.datasets import ImageFolder

from torchvision.transforms import ToTensor, Compose, Resize, Grayscale, Normalize
from torch.utils.data import Dataset
#from data.data_loader import DataLoader
from torchvision.utils import save_image

from PIL import Image
import torch
from torch.autograd import Variable

from imblearn.over_sampling import SMOTE

#dataset = ImageFolder(root="data/datasets/base_dir/train_dir", transform=Compose(transforms))
dataset = ImageFolder(root="./datasets/base_dir/train_dir", transform=Compose([]))

print(dataset)
print(len(dataset))


mylabels = {}
totallabels = 109
tensor_list = []
labels_list = []
for img in dataset:
    tensor_list.append(img[0])
    labels_list.append(img[1])
    if not img[1] in mylabels:
        mylabels[img[1]] = 0
    if mylabels[img[1]] < totallabels:
        tensor_list.append(img[0])
        labels_list.append(img[1])
        mylabels[img[1]] = mylabels[img[1]] + 1
print("Original dataset size: " + str(len(tensor_list)))
print("Original dataset size2: " + str(len(labels_list)))
print("mylabels = "  + str(mylabels))
