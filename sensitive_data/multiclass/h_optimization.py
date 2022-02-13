from itertools import product

import numpy as np
import pandas as pd

from sensitive_data.multiclass import Dataset, BertClassifier

SEED = 42

BATCH_SIZES = [16, 32]
LEARNING_RATES = [1e-6, 1e-5]
DROPOUTS = [0.3, 0.5]
OPTIMIZERS = ['Adam', 'SGD', 'RMSprop']

df = pd.read_csv("train_dataset.csv", delimiter=';')
df = df[df['label'] != 'Other']


np.random.seed(SEED)
df_train, df_val = np.split(df.sample(frac=1, random_state=SEED), [int(.9 * len(df))])
df_train, df_val = Dataset(df_train), Dataset(df_val)


for (batch_size, learning_rate, dropout, optimizer) in product(BATCH_SIZES, LEARNING_RATES, DROPOUTS, OPTIMIZERS):
    model = BertClassifier(dropout, learning_rate, batch_size, optimizer)
    model.train_model(df_train)
    model.evaluate_model(df_val)
