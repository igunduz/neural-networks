import pandas as pd
import torch
from torch.utils.data import Dataset

class FashionMNIST(Dataset):    
    '''
    This class implements the custom dataset, includes init, len and get item methods
    '''
    def __init__(self, csv_file, c = None):
        df = pd.read_csv(csv_file)
        self.X =torch.tensor(df.values[:, 0:-1])
        self.Y = torch.tensor(df.values[:,:1])
        self.transform = transform 
        #self.X = self.X.astype("float32")
      
    # number of rows in the dataset
    def __len__(self):
        return len(self.X)
    
    
    # Returns each image and its label
    def __getitem__(self, idx):
        label = torch.tensor(self.Y[idx])
        img = torch.tensor(self.X[idx])
        if self.transform:
            img = self.transform(img)
        return img, label
        
# Intialize class    
df = FashionMNIST("fashion-mnist_train.csv", transform = None)  

# Dataloader using custom Dataset class    
train_df = torch.utils.data.DataLoader(df, batch_size=64, shuffle=True )  


for imgs, labels in train_df:
    print("Batch of images has shape: ",imgs.shape)
    print("Batch of labels has shape: ", labels.shape)
