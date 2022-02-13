import os

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from matplotlib import pyplot as plt
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, TQDMProgressBar
from pytorch_lightning.loggers import TensorBoardLogger
import seaborn as sn
from sklearn.metrics import multilabel_confusion_matrix, classification_report
from torchmetrics.functional import auroc
from tqdm import tqdm
from transformers import BertTokenizer, BertModel, get_linear_schedule_with_warmup

TOKENIZER = "bert-base-cased"
EPOCHS = 2
THRESHOLD = 0.5
MAX_TOKEN_COUNT = 50

lbl = ['Health',
       'Politics',
       'Religion',
       'Sexuality']
tokenizer = BertTokenizer.from_pretrained(TOKENIZER)


class SensitiveDataDataset(torch.utils.data.Dataset):
    def __init__(self, data: pd.DataFrame):
        self.tokenizer = tokenizer
        self.data = data
        self.max_token_len = MAX_TOKEN_COUNT

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index: int):
        data_row = self.data.iloc[index]
        text = data_row.text
        labels = data_row[lbl]
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
    def __init__(self, train_df, val_df, test_df, batch_size, max_token_len=128):
        super().__init__()
        self.batch_size = batch_size
        self.train_df = train_df
        self.val_df = val_df
        self.test_df = test_df
        self.tokenizer = tokenizer
        self.max_token_len = max_token_len

    def setup(self, stage=None):
        self.train_dataset = SensitiveDataDataset(
            self.train_df
        )
        self.val_dataset = SensitiveDataDataset(
            self.val_df
        )
        self.test_dataset = SensitiveDataDataset(
            self.test_df
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


def _binary_cm(y_pred, y_true):
    binary_cm = [[0, 0], [0, 0]]
    for i in range(len(y_true)):
        if (y_true[i] == [0, 0, 0, 0]).all():
            if (y_pred[i] == [0, 0, 0, 0]).all():
                binary_cm[0][0] += 1
            else:
                binary_cm[0][1] += 1
        else:
            if (y_pred[i] == [0, 0, 0, 0]).all():
                binary_cm[1][0] += 1
            else:
                binary_cm[1][1] += 1
    return binary_cm


def _print_confusion_matrix(confusion_matrix, axes, class_label, class_names, fontsize=14):
    cm_df = pd.DataFrame(
        confusion_matrix, index=class_names, columns=class_names,
    )

    try:
        heatmap = sn.heatmap(cm_df, annot=True, fmt="d", cbar=False, ax=axes)
    except ValueError:
        raise ValueError("Confusion matrix values must be integers.")
    heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=fontsize)
    heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45, ha='right', fontsize=fontsize)
    axes.set_ylabel('True label')
    axes.set_xlabel('Predicted label')
    axes.set_title("Confusion Matrix for the class - " + class_label)


class SensitiveDataTagger(pl.LightningModule):
    def __init__(self, n_classes: int, dropout, learning_rate, batch_size, optimizer, df_length):
        super().__init__()
        self.bert = BertModel.from_pretrained(TOKENIZER, return_dict=True)
        self.dropout = torch.nn.Dropout(dropout)
        self.classifier = torch.nn.Linear(self.bert.config.hidden_size, n_classes)

        steps_per_epoch = df_length // batch_size
        total_training_steps = steps_per_epoch * EPOCHS
        warmup_steps = total_training_steps // 5

        self.n_training_steps = total_training_steps
        self.n_warmup_steps = warmup_steps
        self.criterion = torch.nn.BCELoss()
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.optimizer = optimizer

        self.result_path = f"batch_{batch_size}_learnR_{learning_rate}_dropout_{dropout}_optimizer_{optimizer}/"
        if not os.path.exists(self.result_path):
            os.mkdir(self.result_path)
        print(self.result_path)

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
        self.log("train_loss", loss, prog_bar=True, logger=True, batch_size=self.batch_size)
        return {"loss": loss, "predictions": outputs, "labels": labels}

    def validation_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        loss, outputs = self(input_ids, attention_mask, labels)
        self.log("val_loss", loss, prog_bar=True, logger=True, batch_size=self.batch_size)
        return loss

    def test_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        loss, outputs = self(input_ids, attention_mask, labels)
        self.log("test_loss", loss, prog_bar=True, logger=True, batch_size=self.batch_size)
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
        for i, name in enumerate(lbl):
            class_roc_auc = auroc(predictions[:, i], labels[:, i])
            self.logger.experiment.add_scalar(f"{name}_roc_auc/Train", class_roc_auc, self.current_epoch)

    def configure_optimizers(self):
        if self.optimizer == "Adam":
            optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        elif self.optimizer == "SGD":
            optimizer = torch.optim.SGD(self.parameters(), lr=self.learning_rate)
        else:
            optimizer = torch.optim.RMSprop(self.parameters(), lr=self.learning_rate)
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

    def train_model(self, data_module):
        checkpoint_callback = ModelCheckpoint(
            dirpath=self.result_path,
            filename="best-checkpoint",
            save_top_k=1,
            verbose=True,
            monitor="val_loss",
            mode="min"
        )

        logger = TensorBoardLogger("lightning_logs", name="sensitive-data")
        early_stopping_callback = EarlyStopping(monitor='val_loss', patience=2)

        trainer = pl.Trainer(
            logger=logger,
            callbacks=[checkpoint_callback, early_stopping_callback, TQDMProgressBar(refresh_rate=30)],
            max_epochs=EPOCHS,
            gpus=0
        )

        trainer.fit(self, data_module)

    def evaluate_model(self, data_module: SensitiveDataModule):
        trained_model = SensitiveDataTagger.load_from_checkpoint(
            self.result_path,
            n_classes=len(lbl)
        )
        trained_model.eval()
        trained_model.freeze()

        device = torch.device('cpu')
        trained_model = trained_model.to(device)
        test_dataset = data_module.test_dataset

        predictions = []
        labels = []
        for item in tqdm(test_dataset):
            _, prediction = trained_model(
                item["input_ids"].unsqueeze(dim=0).to(device),
                item["attention_mask"].unsqueeze(dim=0).to(device)
            )
            predictions.append(prediction.flatten())
            labels.append(item["labels"].int())
        predictions = torch.stack(predictions).detach().cpu()
        labels = torch.stack(labels).detach().cpu()
        y_pred = predictions.numpy()
        y_pred = np.where(y_pred > THRESHOLD, 1, 0)
        y_true = labels.numpy()

        binary_cm = _binary_cm(y_pred, y_true)
        df_cm = pd.DataFrame(binary_cm, index=["Other", "Sensitive"], columns=["Other", "Sensitive"], dtype=np.int64)
        plt.figure(figsize=(12, 7))
        sn.heatmap(df_cm, annot=True, fmt='d')
        plt.savefig(self.result_path + 'binary_cm.png')

        multi_cm = multilabel_confusion_matrix(y_pred, y_true)
        fig, ax = plt.subplots(2, 2, figsize=(12, 7))
        for axes, cfs_matrix, label in zip(ax.flatten(), multi_cm, lbl):
            _print_confusion_matrix(cfs_matrix, axes, label, ["Non-" + label, label])
        fig.tight_layout()
        plt.savefig(self.result_path + "multilabel_cm.png")

        with open(self.result_path + "results.txt", "w") as w:
            w.write("Binary:\n")
            accuracy = (binary_cm[0][0] + binary_cm[1][1]) / len(y_true)
            precision = binary_cm[1][1] / (binary_cm[1][1] + binary_cm[0][1]) if (binary_cm[1][1] + binary_cm[0][
                1]) != 0 else 0
            recall = binary_cm[1][1] / (binary_cm[1][1] + binary_cm[1][0]) if (binary_cm[1][1] + binary_cm[1][
                0]) != 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0
            w.write(f"    Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F1-score: {f1}\n")

            w.write("Multi:\n")
            w.write(classification_report(y_true, y_pred, target_names=lbl, zero_division=0))
