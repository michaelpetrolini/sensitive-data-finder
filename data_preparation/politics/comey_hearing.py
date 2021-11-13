import re

import pandas as pd

from data_preparation.configs import POLITICS_FOLDER

BASE_DIR = POLITICS_FOLDER + "josiah-comey-hearing/original/"

with open(POLITICS_FOLDER + "comey_hearing.txt", 'w', encoding='utf-8') as w:
    hearing = pd.read_csv(BASE_DIR + "hearing.csv")
    text = ' '.join(hearing["line"].tolist())
    text = re.sub("[ ]{2,}", " ", text)
    w.write(text)
