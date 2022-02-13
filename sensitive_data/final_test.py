import numpy as np
import pandas as pd
import seaborn as sn
import torch
from matplotlib import pyplot as plt
from sklearn.metrics import classification_report

import sensitive_data

SEED = 42

df = pd.read_csv("E:/Documenti/Uni/Magistrale/Tesi/Data/old/test_dataset2.csv", delimiter=';')
df['b_label'] = df['label'].apply(lambda x: 'Sensitive' if x != 'Other' else x)
df.groupby(['label']).size().plot.bar()

labels = ['Other',
          'Sensitive']
binary_model = sensitive_data.BertClassifier(0.3, labels)
binary_model.load_state_dict(torch.load("binary_model.pt"))
test = sensitive_data.Dataset(df, 'b_label', labels)
df["b_pred"] = list(map(lambda x: labels[x], binary_model.predict(test)))

labels = ['Health',
          'Politics',
          'Religion',
          'Sexuality']
multiclass_model = sensitive_data.BertClassifier(0.5, labels)
multiclass_model.load_state_dict(torch.load("multiclass_model.pt"))
test = sensitive_data.Dataset(df, 'label', labels)
df["m_pred"] = list(map(lambda x: labels[x], multiclass_model.predict(test)))


def split_column(x):
    arguments_list = [0] * len(labels)
    if x != 'Other':
        arguments_list[labels.index(x)] = 1
    return arguments_list


df['label_list'] = df['label'].apply(split_column)
split_df = pd.DataFrame(df['label_list'].tolist(), columns=labels)
df = pd.concat([df, split_df], axis=1)

trained_model = sensitive_data.SensitiveDataTagger.load_from_checkpoint("multilabel_model/best-checkpoint.ckpt")
trained_model.eval()
trained_model.freeze()

test_dataset = sensitive_data.SensitiveDataDataset(df, labels)
df["ml_pred"] = trained_model.predict(test_dataset)

models_evaluator = sensitive_data.ModelsEvaluator(df, labels)
df["hierarchical_accuracy"] = models_evaluator.evaluate_hierarchical()
df["multilabel_accuracy"] = models_evaluator.evaluate_multilabel()
df["multilabel_bin_accuracy"] = models_evaluator.evaluate_binary()
df["multilabel_mul_accuracy"] = models_evaluator.evaluate_multi()
df["multiple_labels"] = models_evaluator.multiple_labels()

print(f"hierarchical binary accuracy: {len(df[df.b_label == df.b_pred]) / len(df)}")
print(f"hierarchical multi accuracy: {len(df[df.label == df.m_pred]) / len(df[df.b_label == 'Sensitive'])}")
print(f"hierarchical total accuracy: {len(df[df.hierarchical_accuracy]) / len(df)}")
print(f"multilabel binary accuracy: {len(df[df.multilabel_bin_accuracy]) / len(df)}")
print(f"multilabel multi accuracy: {len(df[df.multilabel_mul_accuracy]) / len(df[df.b_label == 'Sensitive'])}")
print(f"multilabel total accuracy: {len(df[df.multilabel_accuracy]) / len(df)}")
print(f"multiple labels: {len(df[df.multiple_labels]) / len(df[df.b_label == 'Sensitive'])}")

s_df = df[df.label != 'Other']
y_true = list(map(lambda x: np.array(x), s_df.label_list))
y_pred = list(map(lambda x: np.array(x), s_df.ml_pred))
print(classification_report(y_true, y_pred, target_names=labels, zero_division=0))


h_b_cm, h_m_cm = sensitive_data.hierarchical_cm(df)
m_b_cm, m_m_cm = sensitive_data.multilabel_cm(df)

h_b_cm = pd.DataFrame(h_b_cm, index=["Other", "Sensitive"], columns=["Other", "Sensitive"], dtype=np.int64)
plt.figure(figsize=(12, 7))
sn.heatmap(h_b_cm, annot=True, fmt='d', cmap="YlGnBu")
plt.savefig('h_b_cm.png')

h_m_cm = pd.DataFrame(h_m_cm, index=labels, columns=labels, dtype=np.int64)
plt.figure(figsize=(12, 7))
sn.heatmap(h_m_cm, annot=True, fmt='d', cmap="YlGnBu")
plt.savefig('h_m_cm.png')

m_b_cm = pd.DataFrame(m_b_cm, index=["Other", "Sensitive"], columns=["Other", "Sensitive"], dtype=np.int64)
plt.figure(figsize=(12, 7))
sn.heatmap(m_b_cm, annot=True, fmt='d', cmap="YlGnBu")
plt.savefig('m_b_cm.png')

fig, ax = plt.subplots(2, 2, figsize=(12, 7))

for axes, cfs_matrix, label in zip(ax.flatten(), m_m_cm, labels):
    sensitive_data.print_confusion_matrix(cfs_matrix, axes, label, ["Non-" + label, label])

fig.tight_layout()
plt.savefig('m_m_cm.png')
