import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import torch.nn as nn
import numpy as np
import random
import time
import random

def load_movie_reviews(data_dir, dataset_type):
    """
    Load movie reviews from the aclImdb dataset.

    Parameters:
        - data_dir (str): The directory where the aclImdb dataset is stored.
        - dataset_type (str): The type of dataset to load. Must be "train" or "test".

    Returns:
        - reviews (pandas.DataFrame): A DataFrame containing the reviews, where each row represents a single review and has two columns "review" and "sentiment"
    """
    reviews = []
    for sentiment in ["pos", "neg"]:
        sentiment_dir = os.path.join(data_dir, dataset_type, sentiment)
        for file_name in os.listdir(sentiment_dir):
            if file_name.endswith(".txt"):
                with open(os.path.join(sentiment_dir, file_name), "r", encoding="utf-8") as f:
                    review = f.read()
                    reviews.append([review, sentiment])

    return pd.DataFrame(reviews, columns=["review", "sentiment"])

class MovieReviewDataset(Dataset):
    def __init__(self, data_dir, split, max_length, vocab):
        self.reviews = load_movie_reviews(data_dir, split)
        self.shuffle_idx = list(range(len(self.reviews)))
        random.shuffle(self.shuffle_idx)
        self.shuffle_idx = np.array(self.shuffle_idx)

        self.max_length = max_length
        self.vocab = vocab

    def __len__(self):
        return len(self.reviews)

    def __getitem__(self, idx):
        review = self.reviews.iloc[self.shuffle_idx[idx]]
        review_text = review["review"]
        review_text = [self.vocab[word] if word in self.vocab else self.vocab['<unk>'] for word in review_text.split()]
        sentiment = 1 if review["sentiment"] == "pos" else 0
        
        review_text += [self.vocab['<pad>']] * self.max_length

        review_text = review_text[:self.max_length]
        review_text = torch.tensor(review_text)

        
        return review_text, sentiment


class RNNClassifier(nn.Module):
    def __init__(self, rnn_type, vocab_size, embedding_dim, hidden_dim_rnn, hidden_dim_linear, output_dim, n_rnn_layers, bidirectional, dropout):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rnn_type = rnn_type
        self.n_rnn_layers = n_rnn_layers
        if rnn_type == 'LSTM':
            self.rnn = nn.LSTM(embedding_dim, hidden_dim_rnn, num_layers=n_rnn_layers, bidirectional=bidirectional, dropout=dropout, batch_first=True)
        elif rnn_type == 'RNN':
            self.rnn = nn.RNN(embedding_dim, hidden_dim_rnn, num_layers=n_rnn_layers, bidirectional=bidirectional, dropout=dropout, batch_first=True)
        self.linear_in = nn.Linear(hidden_dim_rnn*4 if bidirectional else hidden_dim_rnn*2, hidden_dim_linear[0])

        self.linear_layers = nn.ModuleList([nn.Linear(hidden_dim_linear[i], hidden_dim_linear[i+1]) for i in range(len(hidden_dim_linear)-1)])
        
        self.fc = nn.Linear(hidden_dim_linear[-1], output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, text):
        embedded = self.dropout(self.embedding(text))
        output, hidden = self.rnn(embedded)
        
        if self.rnn_type == 'LSTM':
            hidden = self.dropout(torch.cat((hidden[0][-1,:,:],hidden[0][-2,:,:]), dim=1))
        elif self.rnn_type == 'RNN':
            hidden = self.dropout(torch.cat((hidden[-1,:,:],hidden[-2,:,:]), dim=1))
        hidden = self.linear_in(hidden)
        for linear_layer in self.linear_layers:
            hidden = linear_layer(hidden)
        return self.fc(hidden.squeeze(0))


def load_vocab(vocab_file):
    vocab = {}
    unknown_id = -100
    with open(vocab_file, 'r') as f:
        for i, line in enumerate(f):
            word = line.strip()
            vocab[word] = i + 1
            unknown_id = max(unknown_id, i)
    unknown_id = unknown_id + 1
    with open(vocab_file, 'r') as f:
        vocab_size = len(f.readlines()) + 1
    
    vocab['<unk>'] = unknown_id
    vocab['<pad>'] = 0
    
    return vocab, vocab_size


def get_data_loader(split, batch_size, num_workers, max_length, shuffle):
    data_dir = "aclImdb_v1/aclImdb"
    vocab, vocab_size  = load_vocab(data_dir + '/imdb.vocab')
    dataset = MovieReviewDataset(data_dir, split, max_length=max_length, vocab=vocab)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=8)
    return loader

def save_model(model, save_path, model_name):
    torch.save(model.state_dict(), f"{save_path}/{model_name}.pth")

def load_model(model, ckpt_path):
    ckpt = torch.load(ckpt_path)
    model.load_state_dict(ckpt)

def evaluate(model, test_loader, criterion, device, num_epochs, first_bach_training_only=False):
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    test_losses = []
    with torch.no_grad():
        for i_epoch, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_losses.append(criterion(output, target[:,None].float()).item())
            pred = output > 0
            correct += pred.eq(target.view_as(pred)).sum().item()

            # print(sum(pred), sum(target))

            # import pdb;
            # pdb.set_trace()

            total += len(data)
            if num_epochs > 0 and i_epoch > num_epochs:
                break
            if first_bach_training_only:
                break

    test_loss = np.mean(test_losses)
    accuracy = 100. * correct / total
    return test_loss, accuracy

def train_loop(num_epochs, train_loader, test_loader, model, device, model_name, criterion, optimizer, first_bach_training_only=False):
    save_path = 'checkpoints/'
    os.makedirs(save_path, exist_ok=True)

    print_every = 1
    save_interval = 5
    eval_every = 3
    num_test_epoch = -1


    test_losses = []
    train_losses = []

    # Training loop
    for i_epoch, epoch in enumerate(range(num_epochs)):
        model.train()
        epoch_losses = []
        start_epoch_time = time.time()

        for i, (text, labels) in enumerate(train_loader):
            # Move data to GPU
            text, labels = text.to(device), labels.to(device)

            # Zero gradients
            optimizer.zero_grad()

            # Forward pass
            output = model(text)

            # Compute loss
            loss = criterion(output, labels[:,None].float())

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            epoch_losses.append (loss.item())

            if first_bach_training_only:
                break
        if i_epoch % print_every == 0:
            print("Epoch train time:", time.time() - start_epoch_time)
            print(f'Epoch: {epoch+1}, Loss: {np.mean(epoch_losses):.4f}')

        if (i_epoch+1) % save_interval == 0:
            save_model(model, save_path, model_name)

        if (i_epoch+1) % eval_every == 0:
            start_eval_time = time.time()
            test_loss, accuracy = evaluate(model, test_loader, criterion, device, num_test_epoch, first_bach_training_only)
            print("Evaluation time:", time.time() - start_eval_time)
            test_losses.append(test_loss)
            print ("evaluation on test result:", "loss:", test_loss, "accuracy", accuracy)

        train_losses.append(np.mean(epoch_losses))

    return train_losses, test_losses


if __name__ == '__main__':

    torch.manual_seed(100)
    random.seed(1)

    data_dir = "aclImdb_v1/aclImdb"
    batch_size = 256    
    vocab, vocab_size  = load_vocab(data_dir + '/imdb.vocab')
    
    # Instantiate the model, dataset, and dataloader
    rnn_type = "LSTM"
    # vocab_size = len(vocab)
    embedding_dim = 64
    hidden_dim_rnn = 1024
    hidden_dim_linear = [2048, 512, 256, 128]
    output_dim = 1
    n_rnn_layers = 2
    bidirectional = False
    max_length = 128
    dropout = 0.05
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    num_workers = 8

    model_name = 'test'
    first_bach_training_only = False

    
    # training setting
    num_epochs = 30

    learning_rate = 1e-3

    
    if first_bach_training_only:
        import copy
        train_loader = get_data_loader('train', batch_size, num_workers=num_workers, max_length=max_length, shuffle=False)
        test_loader = copy.deepcopy(train_loader)
    else:
        train_loader = get_data_loader('train', batch_size, num_workers=num_workers, max_length=max_length, shuffle=True)
        test_loader = get_data_loader('train', batch_size, num_workers=num_workers, max_length=max_length, shuffle=True)
    model = RNNClassifier(rnn_type, vocab_size, embedding_dim, hidden_dim_rnn, hidden_dim_linear, output_dim, n_rnn_layers, bidirectional, dropout).to(device)

    # ckpt_path = 'checkpoints/' + model_name + '.pth'
    # if os.path.exists(ckpt_path):
    #     print("load a pretrained checkpoint:", ckpt_path)
    #     load_model(model, ckpt_path)

    # Define the loss function and optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    train_loop(num_epochs, train_loader, test_loader, model, device, model_name, criterion, optimizer, first_bach_training_only)

