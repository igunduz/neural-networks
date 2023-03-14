from utils import *
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as weight_init
import os
import random


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
    dropout=0.4
    
    save_model_every=10
    seed = 43
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


    learning_rate = 1e-4
    num_epochs = 1000
    single_batch_overfit = False

    model_name = f"tr_hs{hidden_size}_nh{num_heads}_bs{batch_size}_nl{num_att_layers}_dr{dropout}_lr{learning_rate}"
    save_dir = f'checkpoints/{model_name}'
    os.makedirs(save_dir, exist_ok=True)
    
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

    start_epoch, ckpt, best_val = load_checkpoint(model, optimizer, save_dir)
    not_decreasing_val_cnt = 0
    train_losses = []
    valid_losses = []
    train_accs = []
    valid_accs = []

    for epoch in range(start_epoch, num_epochs):
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
        train_losses.append(train_loss)
        train_accs.append(train_accuracy)

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
        valid_losses.append(val_loss)
        valid_accs.append(val_accuracy)

        # Early stopping param update
        if best_val < val_accuracy:
            best_val = val_accuracy
            not_decreasing_val_cnt = 0
            is_best = True
        else:
            not_decreasing_val_cnt += 1
            is_best = False

        if (epoch + 1) % save_model_every == 0:
            save_checkpoint(model, optimizer, epoch, save_dir, best_val, is_best=False)
        if is_best:
            save_checkpoint(model, optimizer, epoch, save_dir, best_val, is_best=True)

        # Plot loss and acc
        if epoch + 1 >= 50 and (epoch + 1) % save_model_every == 0:
            plot_file_name = save_dir + '/{:04d}'.format(epoch+1)
            plot_losses(train_losses, valid_losses, plot_file_name + '_loss.png', val_type='Loss')
            plot_losses(train_accs, valid_accs, plot_file_name + '_acc.png', val_type='Acc')

        print('Epoch: {}, Train Loss: {:.4f}, Train Accuracy: {:.4f}, Val Loss: {:.4f}, Val Accuracy: {:.4f}'.format(epoch+1, train_loss, train_accuracy, val_loss, val_accuracy))
    
        # Early stopping
        if not_decreasing_val_cnt >= 100: 
            break
    
    print ("best validation accuracy: ", best_val)
    with open(save_dir + f'/best_val_{best_val}.txt', 'w') as f:
        f.write(f"best validation accuracy: {best_val}")