from typing import List

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import seaborn as sn
from sklearn.metrics import confusion_matrix, multilabel_confusion_matrix
from torchmetrics.functional import auroc
from tqdm import tqdm
from transformers import BertTokenizer, BertModel, AdamW, get_linear_schedule_with_warmup

TOKENIZER = "bert-base-cased"
MAX_TOKEN_COUNT = 50
THRESHOLD = 0.5

tokenizer = BertTokenizer.from_pretrained(TOKENIZER)


class Dataset(torch.utils.data.Dataset):

    def __init__(self, df, col_name: str, labels: List[str]):
        self.labels = [labels.index(label) if label in labels else len(labels) for label in df[col_name]]
        self.texts = [tokenizer(text,
                                padding='max_length', max_length=MAX_TOKEN_COUNT, truncation=True,
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

    def __init__(self, dropout, labels: List[str]):
        super(BertClassifier, self).__init__()

        self.bert = BertModel.from_pretrained(TOKENIZER)
        self.dropout = torch.nn.Dropout(dropout)
        self.linear = torch.nn.Linear(768, len(labels))
        self.relu = torch.nn.ReLU()
        self.labels = labels

    def forward(self, input_id, mask):
        _, pooled_output = self.bert(input_ids=input_id, attention_mask=mask, return_dict=False)
        dropout_output = self.dropout(pooled_output)
        linear_output = self.linear(dropout_output)
        final_layer = self.relu(linear_output)
        return final_layer

    def predict(self, test_data):
        y_pred = []

        test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=32)

        with torch.no_grad():
            for test_input, test_label in test_dataloader:
                mask = test_input['attention_mask']
                input_id = test_input['input_ids'].squeeze(1)

                output = self(input_id, mask)
                y_pred.extend(output.argmax(dim=1))
        return torch.stack(y_pred).detach().cpu()


class SensitiveDataDataset(torch.utils.data.Dataset):
    def __init__(self, data: pd.DataFrame, labels: List[str]):
        self.tokenizer = tokenizer
        self.data = data
        self.max_token_len = MAX_TOKEN_COUNT
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index: int):
        data_row = self.data.iloc[index]
        text = data_row.text
        labels = data_row[self.labels]
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_token_len,
            return_token_type_ids=False,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt')
        return dict(
            text=text,
            input_ids=encoding["input_ids"].flatten(),
            attention_mask=encoding["attention_mask"].flatten(),
            labels=torch.FloatTensor(labels)
        )


class SensitiveDataModule(pl.LightningDataModule):
    def __init__(self, train_df, val_df, test_df, labels: List[str]):
        super().__init__()
        self.batch_size = 32
        self.train_df = train_df
        self.val_df = val_df
        self.test_df = test_df
        self.tokenizer = tokenizer
        self.max_token_len = MAX_TOKEN_COUNT
        self.labels = labels

    def setup(self, stage=None):
        self.train_dataset = SensitiveDataDataset(
            self.train_df,
            self.labels
        )
        self.val_dataset = SensitiveDataDataset(
            self.val_df,
            self.labels
        )
        self.test_dataset = SensitiveDataDataset(
            self.test_df,
            self.labels
        )

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.val_dataset,
            batch_size=self.batch_size
        )

    def test_dataloader(self):
        return torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=self.batch_size
        )


class SensitiveDataTagger(pl.LightningModule):
    def __init__(self, n_classes: int, n_training_steps=None, n_warmup_steps=None):
        super().__init__()
        self.bert = BertModel.from_pretrained(TOKENIZER, return_dict=True)
        self.dropout = torch.nn.Dropout(0.5)
        self.classifier = torch.nn.Linear(self.bert.config.hidden_size, n_classes)
        self.n_training_steps = n_training_steps
        self.n_warmup_steps = n_warmup_steps
        self.criterion = torch.nn.BCELoss()
        self.device_type = torch.device('cpu')

    def forward(self, input_ids, attention_mask, labels=None):
        output = self.bert(input_ids, attention_mask=attention_mask)
        output = self.dropout(output.pooler_output)
        output = self.classifier(output)
        output = torch.sigmoid(output)
        loss = 0
        if labels is not None:
            loss = self.criterion(output, labels)
        return loss, output

    def training_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        loss, outputs = self(input_ids, attention_mask, labels)
        self.log("train_loss", loss, prog_bar=True, logger=True)
        return {"loss": loss, "predictions": outputs, "labels": labels}

    def validation_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        loss, outputs = self(input_ids, attention_mask, labels)
        self.log("val_loss", loss, prog_bar=True, logger=True)
        return loss

    def test_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        loss, outputs = self(input_ids, attention_mask, labels)
        self.log("test_loss", loss, prog_bar=True, logger=True)
        return loss

    def training_epoch_end(self, outputs):
        labels = []
        predictions = []
        for output in outputs:
            for out_labels in output["labels"].detach().cpu():
                labels.append(out_labels)
            for out_predictions in output["predictions"].detach().cpu():
                predictions.append(out_predictions)
        labels = torch.stack(labels).int()
        predictions = torch.stack(predictions)
        for i, name in enumerate(labels):
            class_roc_auc = auroc(predictions[:, i], labels[:, i])
            self.logger.experiment.add_scalar(f"{name}_roc_auc/Train", class_roc_auc, self.current_epoch)

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters())
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.n_warmup_steps,
            num_training_steps=self.n_training_steps
        )
        return dict(
            optimizer=optimizer,
            lr_scheduler=dict(
                scheduler=scheduler,
                interval='step'
            )
        )

    def predict(self, test_dataset: SensitiveDataDataset):
        predictions = []
        for item in tqdm(test_dataset):
            _, prediction = self(
                item["input_ids"].unsqueeze(dim=0).to(self.device_type),
                item["attention_mask"].unsqueeze(dim=0).to(self.device_type)
            )
            predictions.append(prediction.flatten())

        predictions = torch.stack(predictions).detach().cpu().numpy()
        predictions = np.where(predictions > THRESHOLD, 1, 0)
        return list(map(lambda x: list(x), predictions))


class ModelsEvaluator:
    def __init__(self, dataframe, labels: List[str]):
        self.dataframe = dataframe
        self.labels = labels

    def evaluate_hierarchical(self):
        def evaluate_hierarchical(x):
            if x.b_label != x.b_pred:
                return False
            if x.b_label == 'Other' and x.b_pred == 'Other':
                return True
            if x.label == x.m_pred:
                return True
            return False
        return self.dataframe.apply(lambda x: evaluate_hierarchical(x), axis=1)

    def evaluate_multilabel(self):
        def evaluate_multilabel(x):
            if x.label == 'Other' and (np.array(x.ml_pred) == [0, 0, 0, 0]).all():
                return True
            if x.label != 'Other' and x.ml_pred[self.labels.index(x.label)] == 1:
                return True
            return False
        return self.dataframe.apply(lambda x: evaluate_multilabel(x), axis=1)

    def evaluate_binary(self):
        def evaluate_binary(x):
            if x.label == 'Other' and (np.array(x.ml_pred) == [0, 0, 0, 0]).all():
                return True
            if x.label != 'Other' and (np.array(x.ml_pred) != [0, 0, 0, 0]).any():
                return True
            return False
        return self.dataframe.apply(lambda x: evaluate_binary(x), axis=1)

    def evaluate_multi(self):
        def evaluate_multi(x):
            if x.label != 'Other' and x.ml_pred[self.labels.index(x.label)] == 1:
                return True
            return False
        return self.dataframe.apply(lambda x: evaluate_multi(x), axis=1)

    def multiple_labels(self):
        def multiple_labels(x):
            if sum(x.ml_pred) > 1:
                return True
            return False
        return self.dataframe.apply(lambda x: multiple_labels(x), axis=1)


def print_confusion_matrix(conf_matrix, cm_axes, class_label, class_names, font_size=14):
    df_cm = pd.DataFrame(
        conf_matrix, index=class_names, columns=class_names,
    )

    try:
        heatmap = sn.heatmap(df_cm, annot=True, fmt="d", cbar=False, ax=cm_axes, cmap="YlGnBu")
    except ValueError:
        raise ValueError("Confusion matrix values must be integers.")
    heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=font_size)
    heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45, ha='right', fontsize=font_size)
    cm_axes.set_ylabel('True label')
    cm_axes.set_xlabel('Predicted label')
    cm_axes.set_title("Confusion Matrix for the class - " + class_label)


def hierarchical_cm(dataframe):
    binary_cm = confusion_matrix(dataframe.b_label, dataframe.b_pred)

    s_df = dataframe[dataframe.label != 'Other']
    multi_cm = confusion_matrix(s_df.label, s_df.m_pred)

    return binary_cm, multi_cm


def multilabel_cm(dataframe):
    binary_cm = [[0, 0], [0, 0]]
    y_true = dataframe.b_label
    y_pred = dataframe.ml_pred

    for i in range(len(y_true)):
        if y_true[i] == 'Other':
            if (np.array(y_pred[i]) == [0, 0, 0, 0]).all():
                binary_cm[0][0] += 1
            else:
                binary_cm[0][1] += 1
        else:
            if (np.array(y_pred[i]) == [0, 0, 0, 0]).all():
                binary_cm[1][0] += 1
            else:
                binary_cm[1][1] += 1

    s_df = dataframe[dataframe.label != 'Other']
    y_true = list(map(lambda x: np.array(x), s_df.label_list))
    y_pred = list(map(lambda x: np.array(x), s_df.ml_pred))

    multi_cm = multilabel_confusion_matrix(y_true, y_pred)

    return binary_cm, multi_cm
