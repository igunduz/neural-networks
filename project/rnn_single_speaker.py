from utils import *
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as weight_init
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
    

    
class LSTMNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, n_rnn_layers, hidden_dim_linear, dropout, full_dropout, device):
        super(LSTMNetwork, self).__init__()
        self.rnn = nn.LSTM(input_size, hidden_size, num_layers=n_rnn_layers, dropout=dropout, batch_first=True)
        self.fc = nn.Linear(hidden_dim_linear[-1], output_size)
        self.device = device
        self.hidden_dim_linear = hidden_dim_linear
        self.linear_in = nn.Linear(hidden_size, hidden_dim_linear[0])
        if not full_dropout:
            self.linear_layers = nn.ModuleList(
                [nn.Linear(hidden_dim_linear[i], hidden_dim_linear[i+1]) for i in range(len(hidden_dim_linear)-1)])
        else:
            linear_layers = []
            for i in range(len(hidden_dim_linear)-1):
                linear_layers.append(nn.Linear(hidden_dim_linear[i], hidden_dim_linear[i+1]))
                linear_layers.append(nn.Dropout(dropout))
            self.linear_layers = nn.ModuleList(linear_layers)

        self.init_weights()
        
    def init_weights(self):
        # Initialize the weights of the forget gate to a higher value to encourage remembering
        for name, param in self.rnn.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.0)
            elif 'weight_ih' in name:
                nn.init.kaiming_normal_(param)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param)
                # nn.init.constant_(getattr(self.rnn, name+'_i'), 1.0)    
    
    def forward(self, x, seq_lengths):
        packed = nn.utils.rnn.pack_padded_sequence(x, seq_lengths, batch_first=True, enforce_sorted=True).to(self.device)
        output, _ = self.rnn(packed)
        output, _ = nn.utils.rnn.pad_packed_sequence(output, batch_first=True)
        B, N, F = output.shape
        idx = torch.tensor([l-1 for l in seq_lengths], dtype=torch.int64).to(self.device)
        
        output = torch.gather(output, dim=1, index=idx.view(B, 1, 1).expand(B, N, F))[:,-1,:]
        
        output = self.linear_in(output)
        for linear_layer in self.linear_layers:
            output = linear_layer(output)
        return self.fc(output.squeeze(0))
    
    
if __name__ == "__main__":
    train, test = load_and_split(meta_filename = "SDR_metadata.tsv", speaker= "george")
    num_classes = np.max(train[1].values.tolist()) + 1
    print("number of classes", num_classes)

    num_mels = 13
    # Initialize the model
    batch_size = 256
    input_size = num_mels
    hidden_size = 256
    output_size = num_classes
    learning_rate = 1e-3
    num_epochs = 50
    n_rnn_layers = 3
    hidden_dim_linear = [1024, 512, 256]
    single_batch_overfit = False
    dropout = 0.6
    full_dropout = True

    save_model_every=10
    seed = 43
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    model_name = f"single_speaker_rnn_hs{hidden_size}_bs{batch_size}_nl{n_rnn_layers}_dr{dropout}_lr{learning_rate}"
    if full_dropout:
        model_name += '_fdr'
    save_dir = f'checkpoints/{model_name}'
    os.makedirs(save_dir, exist_ok=True)
    
    train_data = AudioDataset(train, num_mels=num_mels)
    test_data = AudioDataset(test, num_mels=num_mels)
    #dev_data = AudioDataset(dev, num_mels=num_mels)
    
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True, collate_fn=PadSequence(), num_workers=16)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle= not single_batch_overfit, collate_fn=PadSequence(), num_workers=16)
    #dev_loader = DataLoader(dev_data, batch_size=batch_size, shuffle=True, collate_fn=PadSequence(), num_workers=16)
    
    

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = LSTMNetwork(input_size=13, hidden_size=hidden_size, output_size=num_classes, n_rnn_layers=n_rnn_layers, 
                    hidden_dim_linear=hidden_dim_linear, dropout=dropout, full_dropout=full_dropout, device=device).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Number of trainable parameters:", num_params)

    start_epoch, ckpt, best_val = load_checkpoint(model, optimizer, save_dir)
    not_decreasing_val_cnt = 0
    train_losses = []
    valid_losses = []
    train_accs = []
    valid_accs = []
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0
        total_train = 0
        for features, lengths, label in train_loader:
            features = features.to(device)
            # lengths = lengths.to(device)
            label = label.to(device)
            optimizer.zero_grad()
            output = model(features, lengths)
            loss = criterion(output, label)

            loss.backward()
            optimizer.step()

            train_loss += loss.item() * features.size(0)
            train_correct += (torch.argmax(output, dim=1) == label).sum().item()
            total_train += features.size(0)
            if single_batch_overfit:
                break

        train_loss /= total_train # fix calculation of train_loss
        train_accuracy = train_correct / total_train
        train_losses.append(train_loss)
        train_accs.append(train_accuracy)
        
        if single_batch_overfit:
            train_accuracy = train_correct / batch_size

        model.eval()
        val_loss = 0.0
        val_correct = 0
        total_val = 0
        if not single_batch_overfit:
            with torch.no_grad():
                for features, lengths, label in test_loader:
                    features = features.to(device)
                    label = label.to(device)
                    output = model(features, lengths)
                    loss = criterion(output, label)

                    val_loss += loss.item() * features.size(0)
                    val_correct += (torch.argmax(output, dim=1) == label).sum().item()
                    total_val += features.size(0)
        else:
            val_loss = 0.0
            val_correct = 0
        
        val_loss /= total_val # fix calculation of val_loss
        val_accuracy = val_correct / total_val # fix calculation of val_accuracy
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
    