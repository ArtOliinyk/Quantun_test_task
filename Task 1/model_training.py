import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)

import torch
from torch import nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from transformers import BertTokenizer, BertForTokenClassification, BertModel

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
bert = BertModel.from_pretrained('bert-base-uncased')

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
tokenizer.add_tokens(['B-Mountain', 'I-Mountain'])

data = pd.read_csv('mountain_ner_dataset.csv')

label_to_ids = {'B-Mountain': 1,
                'I-Mountain': 2,
                'O': 0}

ids_to_label = {1: 'B-Mountain',
                2: 'I-Mountain',
                0: 'O'}


def tokenize_and_preserve_labels(sentence, text_labels, tokenizer):
    """
    Word piece tokenization makes it difficult to match word labels
    back up with individual word pieces. This function tokenizes each
    word one at a time so that it is easier to preserve the correct
    label for each subword. It is, of course, a bit slower in processing
    time, but it will help our model achieve higher accuracy.
    """

    tokenized_sentence = []
    labels = []

    for word, label in zip(sentence, text_labels):

        tokenized_word = tokenizer.tokenize(word)
        n_subwords = len(tokenized_word)

        ## if sentence consist of more than 32 words, discard the later words.
        if (len(tokenized_sentence) >= 32):
            return tokenized_sentence, labels
        # Add the tokenized word to the final tokenized word list
        tokenized_sentence.extend(tokenized_word)

        # Add the same label to the new list of labels `n_subwords` times
        labels.extend([label] * n_subwords)

    return tokenized_sentence, labels


class Ner_Data(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sentence = self.data['text'][idx].strip().split()
        word_labels = self.data['labels'][idx].split(" ")
        t_sen, t_labl = tokenize_and_preserve_labels(sentence, word_labels, tokenizer)

        sen_code = tokenizer.encode_plus(t_sen,
                                         add_special_tokens=True,  # Add [CLS] and [SEP]
                                         max_length=32,  # maximum length of a sentence
                                         pad_to_max_length=True,  # Add [PAD]s
                                         return_attention_mask=True,  # Generate the attention mask
                                         )

        labels = [-100] * 32
        for i, tok in enumerate(t_labl):
            if label_to_ids.get(tok) != None:
                labels[i + 1] = label_to_ids.get(tok)

        # step 4: turn everything into PyTorch tensors
        item = {key: torch.as_tensor(val) for key, val in sen_code.items()}
        item['labels'] = torch.as_tensor(labels)

        return item


train_data = Ner_Data(data)
train_dataloader = DataLoader(train_data, batch_size=32, shuffle=False)
model2 = model = BertForTokenClassification.from_pretrained('bert-base-uncased', num_labels=len(label_to_ids))
model2.to(device)

learning_rate = 0.0001
batch_size = 128
epochs = 6

loss_fn2 = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


def train_loop(train_dataloader, model, optimizer):
    size = len(train_dataloader.dataset)
    train_loss = 0
    for i, sample in enumerate(train_dataloader):
        optimizer.zero_grad()
        ids = sample['input_ids'].to(device)
        mask = sample['attention_mask'].to(device)
        labels = sample['labels'].to(device)
        pred = model2(input_ids=ids, attention_mask=mask, labels=labels)
        loss = pred[0]

        train_loss += loss.item()
        # Backpropagation
        loss.backward()
        optimizer.step()

        if (i > 0 and i % 64 == 0):
            print(f"loss: {train_loss / i:>4f}  [{i:>5d}/{batch_size}]")
    return train_loss


train_loss = []
for t in range(epochs):
    print(f"Epoch {t + 1}\n-------------------------------")
    loss = train_loop(train_dataloader, model, optimizer)
    train_loss.append(loss)

print("Done!")

torch.save(model, './mountain_ner_model.pt')
