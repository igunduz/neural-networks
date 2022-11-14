import pandas as pd
import torch
from torch.utils.data import Dataset

class FashionMNIST(Dataset):
    
    '''
    This class implements the custom dataset, includes init, len and get item methods
    '''
    def __init__(self, csv_file, transform = None):
        df = pd.read_csv(csv_file)
    
        self.X = df.values[:, 0:-1]
        self.Y = df.values[:,:1]
        #self.X = self.X.astype("float32")
      
    # number of rows in the dataset
    def __len__(self):
        return len(self.X)
    
    
    # Returns each image and its label
    def __getitem__(self, idx):
        label = self.Y[idx]
        img = self.X[idx]
        return img, label
        
    
df = FashionMNIST("fashion-mnist_train.csv", transform = None)  

# Dataloader using custom Dataset class    
train_df = torch.utils.data.DataLoader(df, batch_size=64, shuffle=True )  


for imgs, labels in train_df:
    print("Batch of images has shape: ",imgs.shape)
    print("Batch of labels has shape: ", labels.shape)
