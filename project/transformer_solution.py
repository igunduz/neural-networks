from utils import *
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as weight_init


class AudioDataset(Dataset):
    def __init__(self, data, num_mels=13):
        X, y = data
        self.X = [extract_melspectrogram(x, sr=SAMPLING_RATE, num_mels=num_mels) for x in X]
        self.y = np.array(y.values.tolist())[:,0]

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        sample = self.X[idx], self.y[idx]
        # Perform any necessary data preprocessing here
        return sample

class PadSequence:
    def __call__(self, batch):
        # Let's assume that each element in "batch" is a tuple (data, label).
        # Sort the batch in the descending order
        sorted_batch = sorted(batch, key=lambda x: x[0].shape[1], reverse=True)
        # Get each sequence and pad it
        sequences = [torch.tensor(x[0].reshape(-1, 13), dtype=torch.float32) for x in sorted_batch]
        sequences_padded = torch.nn.utils.rnn.pad_sequence(sequences, batch_first=True)
        # Also need to store the length of each sequence
        # This is later needed in order to unpad the sequences
        lengths = [len(x) for x in sequences]
        # Don't forget to grab the labels of the *sorted* batch
        labels = torch.LongTensor([x[1] for x in sorted_batch])
        return sequences_padded, lengths, labels
    

    

class TransformerModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_heads, num_layers, dropout):
        super(TransformerModel, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_heads = num_heads
        self.num_layers = num_layers
        
        # Define input embedding layer
        self.embedding = nn.Linear(input_size, hidden_size)
        
        # Define transformer encoder layer
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(hidden_size, num_heads, dropout=dropout)
            ,num_layers=num_layers)
        
        # Define output layer
        self.output = nn.Linear(hidden_size, output_size)
        
    def forward(self, input_seq):
        # Apply input embedding
        embedded_seq = self.embedding(input_seq)
        
        # Transpose input for transformer encoder layer
        embedded_seq = embedded_seq.transpose(0, 1)
        
        # Apply transformer encoder layer
        output_seq = self.transformer_encoder(embedded_seq)
        
        # Transpose output to original shape
        output_seq = output_seq.transpose(0, 1)

        # Mean of all output token in the sequence (dim=1)
        output_seq = output_seq.mean(1)
        
        # Apply output layer
        output_seq = self.output(output_seq)
        
        return output_seq


    
    
if __name__ == "__main__":
    train, dev, test = load_and_split(None)
    num_classes = np.max(train[1].values.tolist()) + 1
    print("number of classes", num_classes)

    num_mels = 13
    # Initialize the model
    batch_size = 128
    input_size = num_mels
    hidden_size = 128
    output_size = num_classes
    num_heads = 8
    num_att_layers = 4
    dropout=0.2

    learning_rate = 1e-4
    num_epochs = 500
    single_batch_overfit = False
    
    train_data = AudioDataset(train, num_mels=num_mels)
    test_data = AudioDataset(test, num_mels=num_mels)
    dev_data = AudioDataset(dev, num_mels=num_mels)
    
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True, collate_fn=PadSequence(), num_workers=16)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle= not single_batch_overfit, collate_fn=PadSequence(), num_workers=16)
    dev_loader = DataLoader(dev_data, batch_size=batch_size, shuffle=True, collate_fn=PadSequence(), num_workers=16)
    
    

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = TransformerModel(input_size, hidden_size, output_size, num_heads, num_att_layers, dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0

        for features, lengths, label in train_loader:
            features = features.to(device) # B x Seq len x Num mels
            label = label.to(device)
            optimizer.zero_grad()

            output = model(features)

            # print (torch.argmax(output, 1))

            # import pdb;
            # pdb.set_trace()
            loss = criterion(output, label)

            loss.backward()
            optimizer.step()

            train_loss += loss.item() * features.size(0)
            train_correct += (torch.argmax(output, dim=1) == label).sum().item()
            
            if single_batch_overfit:
                break

        train_loss /= len(train_data)
        train_accuracy = train_correct / len(train_data)
        
        if single_batch_overfit:
            train_accuracy = train_correct / batch_size

        model.eval()
        val_loss = 0.0
        val_correct = 0
    
        if not single_batch_overfit:
            with torch.no_grad():
                for features, lengths, label in dev_loader:
                    features = features.to(device)
                    label = label.to(device)
                    output = model(features)
                    loss = criterion(output, label)

                    val_loss += loss.item() * features.size(0)
                    val_correct += (torch.argmax(output, dim=1) == label).sum().item()
        else:
            val_loss = 0.0
            val_correct = 0
        
        val_loss /= len(dev_data)
        val_accuracy = val_correct / len(dev_data)

        print('Epoch: {}, Train Loss: {:.4f}, Train Accuracy: {:.4f}, Val Loss: {:.4f}, Val Accuracy: {:.4f}'.format(epoch+1, train_loss, train_accuracy, val_loss, val_accuracy))
    
    