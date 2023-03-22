from utils import *
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as weight_init
import os, random, math

from config_loader import get_config

import logging
from sklearn.metrics import f1_score, confusion_matrix, precision_score, recall_score

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
    
# Reference: https://pytorch.org/tutorials/beginner/transformer_tutorial.html
class PositionalEncoding(nn.Module):
    
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 1000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

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
        
        self.pos_encoding = PositionalEncoding(hidden_size, dropout)
        
        # Define transformer encoder layer
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(hidden_size, num_heads, dropout=dropout)
            ,num_layers=num_layers)
        
        # Define output layer
        self.output = nn.Linear(hidden_size, output_size)
        
    def forward(self, input_seq):
        # Apply input embedding
        embedded_seq = self.embedding(input_seq)
        embedded_seq = self.pos_encoding(embedded_seq)
        
        # Transpose input for transformer encoder layer -> [seq_len, batch_size, feat_size]
        embedded_seq = embedded_seq.transpose(0, 1)
        
        
        # Apply transformer encoder layer
        output_seq = self.transformer_encoder(embedded_seq)
        
        # Transpose output to original shape -> [batch_size, seq_len, feat_size]
        output_seq = output_seq.transpose(0, 1)

        # Mean of all output token in the sequence (dim=1)
        output_seq = output_seq.mean(1)
        
        # Apply output layer
        output_seq = self.output(output_seq)
        
        return output_seq

def evaluate(model, test_loader, device, logger):
    with torch.no_grad():
        model.eval()
        y_pred = []
        y_true = []
        for features, lengths, label in test_loader:
            features = features.to(device)
            # lengths = lengths.to(device)
            label = label.to(device)
            optimizer.zero_grad()
            output = model(features)
            
            pred = torch.argmax(output, dim=1).squeeze().cpu().numpy().tolist()
            label = label.squeeze().cpu().numpy().tolist()
            
            y_pred += pred
            y_true += label
        
        # Compute the F1 score
        f1 = f1_score(y_true, y_pred, average='micro')
        print("F1 score:", f1)
        logging.info(f"\nF1 score:{f1}")

        # Compute the precision and recall
        precision = precision_score(y_true, y_pred, average='micro')
        recall = recall_score(y_true, y_pred, average='micro')
        print("Precision:", precision)
        print("Recall:", recall)

        logging.info(f"\nPrecision: {precision}")
        logging.info(f"\nRecall: {recall}")

        # Compute the confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        print("Confusion matrix:")
        print(cm)

            
        logging.info("\nConfusion matrix:")
        logging.info(cm)
        

    
    
if __name__ == "__main__":
    speakers = ['george', 'jackson', 'lucas', 'nicolas', 'theo', 'yweweler'][:4]
    speakers_selected = ['nicolas', 'theo' , 'jackson',  'george']
    
    train, dev, test = load_and_split(meta_filename = "SDR_metadata.tsv", speaker=speakers_selected)
    num_classes = np.max(train[1].values.tolist()) + 1
    print("number of classes", num_classes, flush=True)
    
    cfg = get_config()

    num_mels = cfg.num_mels
    # Initialize the model
    batch_size = cfg.batch_size
    input_size = num_mels
    hidden_size = cfg.hidden_size
    output_size = num_classes
    num_heads = cfg.num_heads
    num_att_layers = cfg.num_att_layers
    dropout=cfg.dropout
    
    save_model_every=50
    seed = 43
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


    learning_rate = 1e-4
    num_epochs = cfg.num_epochs
    single_batch_overfit = False

    model_name = cfg.name + f"_ntr_hs{hidden_size}_nh{num_heads}_bs{batch_size}_nl{num_att_layers}_dr{dropout}_lr{learning_rate}"
    save_dir = f'checkpoints/{model_name}'
    os.makedirs(save_dir, exist_ok=True)
    
    train_data = AudioDataset(train, num_mels=num_mels)
    test_data = AudioDataset(test, num_mels=num_mels)
    dev_data = AudioDataset(dev, num_mels=num_mels)
    
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True, collate_fn=PadSequence(), num_workers=20)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle= not single_batch_overfit, collate_fn=PadSequence(), num_workers=20)
    dev_loader = DataLoader(dev_data, batch_size=batch_size, shuffle=True, collate_fn=PadSequence(), num_workers=20)
    
    

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = TransformerModel(input_size, hidden_size, output_size, num_heads, num_att_layers, dropout).to(device)

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Number of trainable parameters:", num_params, flush=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    start_epoch, ckpt, best_val = load_checkpoint(model, optimizer, save_dir)
    not_decreasing_val_cnt = 0
    train_losses = []
    valid_losses = []
    train_accs = []
    valid_accs = []
    
    logger = logging.basicConfig(filename=save_dir + f'/log.txt', level=logging.INFO, format='')
    logging.info(f"num params: {num_params}")
    
    for epoch in range(start_epoch, num_epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0
        total_train = 0
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
            total_train += features.size(0)
            if single_batch_overfit:
                break

        train_loss /= total_train # fix calculation of train_loss
        train_accuracy = train_correct / total_train
        
        if single_batch_overfit:
            train_accuracy = train_correct / batch_size
        train_losses.append(train_loss)
        train_accs.append(train_accuracy)

        model.eval()
        val_loss = 0.0
        val_correct = 0
        total_val = 0
        if not single_batch_overfit:
            with torch.no_grad():
                for features, lengths, label in dev_loader:
                    features = features.to(device)
                    label = label.to(device)
                    output = model(features)
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

        print('Epoch: {}, Train Loss: {:.4f}, Train Accuracy: {:.4f}, \
            Val Loss: {:.4f}, Val Accuracy: {:.4f}'.format(epoch+1, train_loss, train_accuracy, val_loss, val_accuracy), flush=True)
    
        # Early stopping
        if not_decreasing_val_cnt >= 15 and epoch + 1 >= 50: 
            break
    
    print ("best validation accuracy: ", best_val, flush=True)
    with open(save_dir + f'/best_val_{best_val}.txt', 'w') as f:
        f.write(f"best validation accuracy: {best_val}")
        
    evaluate(model, test_loader, device, logger)