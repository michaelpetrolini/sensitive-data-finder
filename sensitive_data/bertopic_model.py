import pandas as pd
from bertopic import BERTopic

df = pd.read_csv(r"C:\Users\Omi069\Downloads\Batch_4627728_batch_results.csv")
docs = list(df.loc[:, "Input.text"].values)

model = BERTopic(language="english")
topics, probs = model.fit_transform(docs)
model.get_topic_freq()
fig = model.visualize_topics()
fig.write_html(r"C:\Users\Omi069\Downloads\file.html")
