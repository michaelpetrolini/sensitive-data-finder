import os

import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
import seaborn as sn
from sklearn.metrics import confusion_matrix, f1_score
from tqdm import tqdm
from transformers import BertModel, BertTokenizer

TOKENIZER = "bert-base-cased"
MAX_TOKEN_COUNT = 50
EPOCHS = 2

labels = ['Health', 'Politics', 'Religion', 'Sexuality']
tokenizer = BertTokenizer.from_pretrained(TOKENIZER)


class Dataset(torch.utils.data.Dataset):

    def __init__(self, df):
        self.labels = [labels.index(label) for label in df['label']]
        self.texts = [tokenizer(text, padding='max_length', max_length=MAX_TOKEN_COUNT, truncation=True,
                                return_tensors="pt") for text in df['text']]

    def classes(self):
        return self.labels

    def __len__(self):
        return len(self.labels)

    def get_batch_labels(self, idx):
        # Fetch a batch of labels
        return np.array(self.labels[idx])

    def get_batch_texts(self, idx):
        # Fetch a batch of inputs
        return self.texts[idx]

    def __getitem__(self, idx):
        batch_texts = self.get_batch_texts(idx)
        batch_y = self.get_batch_labels(idx)

        return batch_texts, batch_y


class BertClassifier(torch.nn.Module):
    def __init__(self, dropout, learning_rate, batch_size, optimizer):
        super(BertClassifier, self).__init__()

        self.bert = BertModel.from_pretrained(TOKENIZER)
        self.dropout = torch.nn.Dropout(dropout)
        self.linear = torch.nn.Linear(768, 4)
        self.activation_function = torch.nn.ReLU()

        self.criterion = torch.nn.CrossEntropyLoss()
        self.device = torch.device("cpu")

        self.batch_size = batch_size
        if optimizer == 'Adam':
            self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        elif optimizer == 'RMSprop':
            self.optimizer = torch.optim.RMSprop(self.parameters(), lr=learning_rate)
        else:
            self.optimizer = torch.optim.SGD(self.parameters(), lr=learning_rate)

        self.result_path = f"batch_{batch_size}_learnR_{learning_rate}_dropout_{dropout}_optimizer_{optimizer}/"
        if not os.path.exists(self.result_path):
            os.mkdir(self.result_path)
        print(self.result_path)

    def forward(self, input_id, mask):
        _, pooled_output = self.bert(input_ids=input_id, attention_mask=mask, return_dict=False)
        dropout_output = self.dropout(pooled_output)
        linear_output = self.linear(dropout_output)
        final_layer = self.activation_function(linear_output)
        return final_layer

    def fit(self, train_data: Dataset):
        train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=self.batch_size, shuffle=True)

        for epoch_num in range(EPOCHS):
            for train_input, train_label in tqdm(train_dataloader):
                train_label = train_label.detach().clone().to(self.device).to(torch.long)
                mask = train_input['attention_mask'].to(self.device)
                input_id = train_input['input_ids'].squeeze(1).to(self.device)

                output = self(input_id, mask)

                batch_loss = self.criterion(output, train_label)

                self.zero_grad()
                batch_loss.backward()
                self.optimizer.step()

        torch.save(self.state_dict(), self.result_path + "model.pt")

    def evaluate(self, val_data: Dataset):
        y_pred = []
        y_true = []

        val_dataloader = torch.utils.data.DataLoader(val_data, batch_size=self.batch_size)
        total_acc_val, total_loss_val = 0, 0

        with torch.no_grad():
            for test_input, test_label in val_dataloader:
                test_label = test_label.detach().clone().to(self.device).to(torch.long)
                mask = test_input['attention_mask']
                input_id = test_input['input_ids'].squeeze(1)

                output = self(input_id, mask)

                batch_loss = self.criterion(output, test_label)
                total_loss_val += batch_loss.item()

                acc = (output.argmax(dim=1) == test_label).sum().item()
                total_acc_val += acc

                y_true.extend(test_label)
                y_pred.extend(output.argmax(dim=1))

        cf_matrix = confusion_matrix(y_true, y_pred)
        df_cm = pd.DataFrame(cf_matrix, index=labels, columns=labels, dtype=np.int64)
        plt.figure(figsize=(12, 7))
        sn.heatmap(df_cm, annot=True, fmt='d')
        plt.savefig(self.result_path + 'cm.png')

        with open(self.result_path + 'results.txt', "w") as w:
            w.write(f'Test Accuracy: {total_acc_val / len(val_data): .3f}\n')
            w.write(f'Test Loss: {total_loss_val / len(val_data): .3f}\n')
            w.write(f'Test F1-score per class: {f1_score(y_true, y_pred, average=None)}\n')
            w.write(f'Test weighted F1-score: {f1_score(y_true, y_pred, average="weighted")}\n')
