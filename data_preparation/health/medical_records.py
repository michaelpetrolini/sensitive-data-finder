import re

import pandas as pd

from data_preparation import HEALTH_FOLDER

BASE_DIR = HEALTH_FOLDER + "arvin6-medical-records-10-yrs/"

with open(HEALTH_FOLDER + "medical_records.txt", 'w', encoding='utf-8') as w:
    encounter = pd.read_csv(BASE_DIR + "encounter.csv")
    concat = encounter.fillna('')[["CC", "SOAP_Note"]].astype(str).agg('. '.join, axis=1)
    text = '\n'.join(concat.tolist())
    text = re.sub("\. \n", "", text)
    text = re.sub("s:", "", text)
    text = re.sub("[ ]{2,}", " ", text)
    w.write(text)
