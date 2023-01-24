import torch
from torchvision import transforms
from torchvision import datasets

from torch.nn import Module
import torch.nn as nn
import torch.nn.functional as F



class BlockCNN(Module):
    def __init__(self,in_channels, out_channels, kernel_size):
        super(BlockCNN, self).__init__()
        self.conv_1 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size//2)
        self.conv_2 = nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, padding=kernel_size//2)
        self.conv_3 = nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, padding=kernel_size//2)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        inp = x
        x1 = self.relu(self.conv_1(x))
        x2 = self.relu(self.conv_2(x1))
        x3 = self.relu(self.conv_3(x2))
        x_out = x1 + x3
        return x_out

		

class CNN_model(Module):
    
    def __init__(self, 
    in_channels=3, 
    num_blocks=3, 
    kernel_size=3, 
    filter_size=32, 
    num_classes=10,
    pool_size=4):
        
        super(CNN_model, self).__init__()
        self.num_blocks = num_blocks
        blocks = []

        for i in range (num_blocks):
            if i == 0:
                _in_channels = in_channels
            else:
                _in_channels = filter_size
            blocks.append(BlockCNN(_in_channels, filter_size, kernel_size=kernel_size))
        
        self.cnn_blocks = nn.Sequential(*blocks)
        self.pool = nn.MaxPool2d(kernel_size=pool_size, stride=pool_size)

        self.fc1 = nn.Linear(in_features= filter_size*((32 // pool_size)**2), out_features=1024)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, num_classes)
       


    def forward(self, x):
        x = self.cnn_blocks(x)
        x = self.pool(x)
        x = x.view(x.shape[0],-1)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def load_data_svhn(split, batch_size=32):
    
    # Preprocess the data by resizing, center cropping, and normalization based on the data split
    transform_train = transforms.Compose([
                                # transforms.Resize(32),
                                # transforms.CenterCrop(32),
                                transforms.Grayscale(1),
                                transforms.ToTensor(),
                                # transforms.Normalize([0.5, 0.5, 0.5],
                                                    #  [0.5, 0.5, 0.5])
                                                     ])
    
    transform_test = transforms.Compose([
                                        transforms.Grayscale(1),
                                        transforms.ToTensor(),
                                        # transforms.Normalize((0.5, 0.5, 0.5),
                                                            #  (0.5, 0.5, 0.5))
                                                             ])
   # download the SVHN dataset
    if split == 'train':
        svhn_data = datasets.SVHN(root='path/to/data', split='train', download=True, transform=transform_train)
    
    elif split == 'test':
        svhn_data = datasets.SVHN(root='path/to/data', split='test', download=True, transform=transform_test)
    else:
        svhn_data = datasets.SVHN(root='path/to/data', split='extra', download=True, transform=transform_test)
    # load the data 
    data_loader = torch.utils.data.DataLoader(svhn_data, batch_size=batch_size, shuffle=False, num_workers=4, drop_last = False)

    return data_loader

if __name__ == "__main__":
    model = CNN_model(in_channels=3, 
                    num_blocks=3, 
                    kernel_size=3, 
                    filter_size=32, 
                    num_classes=10,
                    pool_size=4)
    inp = torch.zeros (10, 3, 32, 32)
    out = model(inp)
    print (out.shape)

    

    
