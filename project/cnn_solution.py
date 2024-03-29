from utils import *
from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as weight_init
from config_loader import *

import logging
from sklearn.metrics import f1_score, confusion_matrix, precision_score, recall_score, classification_report


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
    

    
class SpeechToTextCNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dropout_prob):
        super(SpeechToTextCNN, self).__init__()

        # Define the layers
        self.conv1 = nn.Conv1d(in_channels=input_size, out_channels=hidden_size, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool1d(kernel_size=2)
        self.dropout1 = nn.Dropout(dropout_prob)

        self.conv2 = nn.Conv1d(in_channels=hidden_size, out_channels=hidden_size, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(hidden_size)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool1d(kernel_size=2)
        self.dropout2 = nn.Dropout(dropout_prob)

        self.conv3 = nn.Conv1d(in_channels=hidden_size, out_channels=hidden_size, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(hidden_size)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool1d(kernel_size=2)
        self.dropout3 = nn.Dropout(dropout_prob)

        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):

        x = x.transpose(1,2)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        x = self.dropout1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        x = self.dropout2(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)
        x = self.pool3(x)
        x = self.dropout3(x)
        x = x.transpose(1,2)

        x = x.mean(dim=1) # Global average pooling
        last_layer = x
        x = self.fc(x)

        return x, last_layer

def evaluate(model, test_loader, device, logger, filename):
    
    from sklearn.manifold import TSNE
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('Agg')
    
    
    with torch.no_grad():
        model.eval()
        y_pred = []
        y_true = []
        feats = []
        for features, lengths, label in test_loader:
            features = features.to(device)
            # lengths = lengths.to(device)
            label = label.to(device)
            optimizer.zero_grad()
            output, last_layer = model(features)
            feats.append(last_layer.cpu().numpy())
            pred = torch.argmax(output, dim=1).squeeze().cpu().numpy().tolist()
            label = label.squeeze().cpu().numpy().tolist()
            
            y_pred += pred
            y_true += label
        
        feats = np.concatenate (feats, axis=0)
        
        # Compute the F1 score
        f1 = f1_score(y_true, y_pred, average='macro')
        print("F1 score:", f1)
        logging.info(f"\nF1 score:{f1}")

        # Compute the precision and recall
        precision = precision_score(y_true, y_pred, average='macro')
        recall = recall_score(y_true, y_pred, average='macro')
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
        
        clf_report = classification_report(y_true, y_pred)
        print (clf_report)
        
        tsne_output = TSNE(n_components=2).fit_transform(feats)

        # Plot the reduced-dimensional data in a scatter plot, colored by the true labels
        color_map = plt.cm.get_cmap('gist_ncar', 10)

        for i in range(10):
            mask = np.array(y_true) == i
            # ax.scatter(tsne_output[mask][:,0], tsne_output[mask][:,1], c=color_map(i)[:3], label=f'{i}')        
            plt.scatter(tsne_output[mask][:,0], tsne_output[mask][:,1], c=(np.random.rand(), np.random.rand(), np.random.rand()), label=f'{i}')        
        plt.legend()

        # Add axis labels and a title
        plt.xlabel('1st dim')
        plt.ylabel('2nd dim')
        plt.savefig(filename)
        # plt.clf()
        
        
    
if __name__ == "__main__":
    speakers = ['george', 'jackson', 'lucas', 'nicolas', 'theo', 'yweweler'][:4]
    speakers_selected = ['nicolas', 'theo' , 'jackson',  'george']
    train, dev, test = load_and_split(meta_filename = "SDR_metadata.tsv", speaker=speakers_selected)
    cfg = get_config()
    num_classes = np.max(train[1].values.tolist()) + 1
    print("number of classes", num_classes)

    num_mels = cfg.num_mels
    # Initialize the model
    batch_size = cfg.batch_size
    input_size = num_mels
    hidden_size = cfg.hidden_size
    output_size = num_classes
    learning_rate = 1e-3
    num_epochs = cfg.num_epochs
    single_batch_overfit = False
    dropout=cfg.dropout
    print("variables intialized")

    save_model_every=50
    seed = 43
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    model_name = cfg.name + f"_cnn_hs{hidden_size}_bs{batch_size}_dr{dropout}_lr{learning_rate}"
    save_dir = f'checkpoints/{model_name}'
    os.makedirs(save_dir, exist_ok=True)
    
    train_data = AudioDataset(train, num_mels=num_mels)
    test_data = AudioDataset(test, num_mels=num_mels)
    dev_data = AudioDataset(dev, num_mels=num_mels)
    
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True, collate_fn=PadSequence(), num_workers=20)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle= not single_batch_overfit, collate_fn=PadSequence(), num_workers=20)
    dev_loader = DataLoader(dev_data, batch_size=batch_size, shuffle=True, collate_fn=PadSequence(), num_workers=20)
    
    

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model =  SpeechToTextCNN(input_size=13, hidden_size=hidden_size, 
                         output_size=num_classes, dropout_prob=dropout).to(device)

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print("Number of trainable parameters:", num_params)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    start_epoch, ckpt, best_val = load_checkpoint(model, optimizer, save_dir)
    not_decreasing_val_cnt = 0
    train_losses = []
    valid_losses = []
    train_accs = []
    valid_accs = []

    logger = logging.basicConfig(filename=save_dir + f'/log.txt', level=logging.INFO, format='%(message)s')
    logging.info(f"num params: {num_params}")


    print("starting the training")
    for epoch in range(start_epoch, num_epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0
        total_train = 0
        for features, lengths, label in train_loader:
            features = features.to(device)
            # lengths = lengths.to(device)
            label = label.to(device)
            optimizer.zero_grad()
            output, last_layer = model(features)
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
        # train_accuracy = train_correct / len(train_data)
    
        model.eval()
        val_loss = 0.0
        val_correct = 0
        total_val = 0

        if not single_batch_overfit:
            with torch.no_grad():
                for features, lengths, label in test_loader:
                    features = features.to(device)
                    label = label.to(device)
                    output, last_layer = model(features)
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
        if not_decreasing_val_cnt >= 15 and epoch + 1 >= 50: 
            break
        
    print ("best validation accuracy: ", best_val)
    with open(save_dir + f'/best_val_{best_val}.txt', 'w') as f:
        f.write(f"best validation accuracy: {best_val}")
    plot_file_name = save_dir + '/'
    print (plot_file_name)
    evaluate(model, test_loader, device, logger, plot_file_name + "_tsne.png")
    
    
    
    