import torch
from torchvision import transforms
from torchvision import datasets

from torch.nn import Module
import torch.nn as nn
import torch.nn.functional as F





class CNN_model(Module):
    
    def __init__(self, in_channels= 3, out_channels = 32,  
                 kernel_size = 3, stride = 1, padding = 1, num_classes = 10):
        
        super(CNN_model, self).__init__()

        self.conv_layer1 = Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.max_pool1 = MaxPool2d(kernel_size = kernel_size, stride = stride)
        self.relu = ReLU()

        self.conv_layer2 = Conv2d(in_channels=32, out_channels= 32, kernel_size = 3, stride = 2, padding = 0)
        self.max_pool2 = MaxPool2d(kernel_size = kernel_size, stride = stride)
    
        self.fc1 = nn.Linear(in_features= 32 * 12*12, out_features=1024)
        self.relu2 = nn.ReLU()
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, num_classes)
       


    def forward(self, x):
        x = self.conv_layer1(x)
        x= self.relu(x)
        x= self.max_pool1(x)
       # print("Output shape after maxpool1:", x.shape)
        x = self.conv_layer2(x)
        x= self.max_pool2(x)
       # print("Output shape after maxpool2:", x.shape)
#         x = nn.Flatten(x)
#         print("shape aftesr flattening: ", x.size)
        x = x.view(-1, 32 * 12*12)
        x = self.fc1(x)
        
        x = self.relu2(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        x = softmax(x)
#         outputs = torch.argmax(x, dim=3)
        return x
    
    
    


def load_data_svhn(split):
    
    # Preprocess the data by resizing, center cropping, and normalization based on the data split
    transform_train = transforms.Compose([transforms.Resize(32),
                                transforms.CenterCrop(32),
                                transforms.ToTensor(),
                                transforms.Normalize([0.5, 0.5, 0.5],
                                                     [0.5, 0.5, 0.5])])
    
    transform_test = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize((0.5, 0.5, 0.5),
                                                             (0.5, 0.5, 0.5))])
   # download the SVHN dataset
    if split == 'train':
        svhn_data = datasets.SVHN(root='path/to/data', split='train', download=True, transform=transform_train)
    
    elif split == 'test':
        svhn_data = datasets.SVHN(root='path/to/data', split='test', download=True, transform=transform_test)
    else:
        svhn_data = datasets.SVHN(root='path/to/data', split='extra', download=True, transform=transform_test)
    # load the data 
    data_loader = torch.utils.data.DataLoader(svhn_data, batch_size=32, shuffle=True, num_workers=4, drop_last = True)

    return data_loader



    

    