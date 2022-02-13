from itertools import product

import numpy as np
import pandas as pd

from sensitive_data.multilabel import SensitiveDataModule, SensitiveDataTagger

SEED = 42

BATCH_SIZES = [16]
LEARNING_RATES = [1e-6, 1e-5]
DROPOUTS = [0.3, 0.5]
OPTIMIZERS = ['Adam', 'SGD', 'RMSprop']

df = pd.read_csv("train_dataset.csv", delimiter=';')
df['text'] = df.text.str.replace('[^a-zA-Z ]', '', regex=True)

labels = ['Health',
          'Politics',
          'Religion',
          'Sexuality']


def split_column(x):
    argument_columns = [0] * len(labels)
    if x != 'Other':
        argument_columns[labels.index(x)] = 1
    return argument_columns


df['label'] = df['label'].apply(split_column)
split_df = pd.DataFrame(df['label'].tolist(), columns=labels)
df = pd.concat([df, split_df], axis=1)
df.pop('label')

np.random.seed(SEED)
train_df, val_df, test_df = np.split(df.sample(frac=1, random_state=SEED), [int(.8 * len(df)), int(.801 * len(df))])

for (batch_size, learning_rate, dropout, optimizer) in product(BATCH_SIZES, LEARNING_RATES, DROPOUTS, OPTIMIZERS):
    data_module = SensitiveDataModule(
        train_df,
        val_df,
        test_df,
        batch_size=batch_size,
    )

    model = SensitiveDataTagger(
        n_classes=len(labels),
        dropout=dropout,
        learning_rate=learning_rate,
        batch_size=batch_size,
        optimizer=optimizer,
        df_length=len(train_df)
    )

    model.train_model(data_module)
    model.evaluate_model(data_module)
