import os
import random
import numpy as np
import time
import torch
from tqdm import tqdm
from torch.utils.data.dataset import random_split
from torch.utils.data import DataLoader
from gan_training import text_classification_data

from gan_training.models.EmbeddingBag import TextEmbeddingBag
from gan_training.models.BiLSTM import TextBiLSTM

seed = 2020
torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)

NGRAMS = 2
N_EPOCHS = 5
BATCH_SIZE = 32
EMBED_DIM = 64
HIDDEN_DIM = 64
N_LAYERS = 1
DROPOUT = 0.1

min_valid_loss = float('inf')

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

if not os.path.isdir('../data'):
    os.mkdir('../data')

train_dataset, test_dataset = text_classification_data.DATASETS['AG_NEWS'](
    root='../data', ngrams=NGRAMS, vocab=None)


train_len = int(len(train_dataset) * 0.95)
sub_train_, sub_valid_ = random_split(train_dataset, [train_len, len(train_dataset) - train_len])

VOCAB_SIZE = len(train_dataset.get_vocab())
NUN_CLASS = len(train_dataset.get_labels())
# model = TextEmbeddingBag(VOCAB_SIZE, EMBED_DIM, NUN_CLASS).to(device)
model = TextBiLSTM(VOCAB_SIZE, EMBED_DIM, NUN_CLASS, HIDDEN_DIM, N_LAYERS, DROPOUT).to(device)

criterion = torch.nn.CrossEntropyLoss().to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=4.0)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.9)


def train_func(sub_train_):

    # Train the model
    train_loss = 0
    train_acc = 0
    data = DataLoader(sub_train_, batch_size=BATCH_SIZE, shuffle=True,
                      collate_fn=collate_batch_lstm, drop_last=True)
    model.train()
    for i, (text, length, cls) in tqdm(enumerate(data)):
        optimizer.zero_grad()
        text, cls = text.to(device), cls.to(device)
        # output = model(text, offsets)
        output = model(text, length)
        loss = criterion(output, cls)
        train_loss += loss.item()
        loss.backward()
        optimizer.step()
        train_acc += (output.argmax(1) == cls).sum().item()

    # Adjust the learning rate
    scheduler.step()

    return train_loss / len(sub_train_), train_acc / len(sub_train_)


def test(data_):
    loss = 0
    acc = 0
    data = DataLoader(data_, batch_size=BATCH_SIZE, collate_fn=collate_batch_lstm)
    model.eval()
    for text, length, cls in data:
        text, cls = text.to(device), cls.to(device)
        with torch.no_grad():
            # output = model(text, offsets)
            output = model(text, length)
            loss = criterion(output, cls)
            loss += loss.item()
            acc += (output.argmax(1) == cls).sum().item()

    return loss / len(data_), acc / len(data_)


def generate_batch(batch):
    label = torch.tensor([entry[0] for entry in batch])
    text = [entry[1] for entry in batch]
    offsets = [0] + [len(entry) for entry in text]
    offsets = torch.tensor(offsets[:-1]).cumsum(dim=0)
    text = torch.cat(text)
    return text, offsets, label


def collate_batch_lstm(batch):
    sorted_batch = sorted(batch, key=lambda x: x[1].shape[0], reverse=True)
    sequences = [x[1] for x in sorted_batch]
    sequences_padded = torch.nn.utils.rnn.pad_sequence(sequences, batch_first=True)
    lengths = torch.LongTensor([len(x) for x in sequences])
    labels = torch.LongTensor([x[0] for x in sorted_batch])
    return sequences_padded, lengths, labels


for epoch in range(N_EPOCHS):

    start_time = time.time()
    train_loss, train_acc = train_func(sub_train_)
    valid_loss, valid_acc = test(sub_valid_)

    secs = int(time.time() - start_time)
    mins = secs / 60
    secs = secs % 60

    print('Epoch: %d' %(epoch + 1), " | time in %d minutes, %d seconds" %(mins, secs))
    print(f'\tLoss: {train_loss:.4f}(train)\t|\tAcc: {train_acc * 100:.1f}%(train)')
    print(f'\tLoss: {valid_loss:.4f}(valid)\t|\tAcc: {valid_acc * 100:.1f}%(valid)')


print('Checking the results of test dataset...')
test_loss, test_acc = test(test_dataset)
print(f'\tLoss: {test_loss:.4f}(test)\t|\tAcc: {test_acc * 100:.1f}%(test)')