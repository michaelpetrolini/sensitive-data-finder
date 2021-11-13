import re

from data_preparation.configs import *

with open(FINANCE_FOLDER + "finance.txt", "r+", encoding="utf-8") as f:
    doc = f.read()

    doc = re.sub("\t", " ", doc)
    doc = re.sub("[0-9]*", "", doc)
    doc = re.sub("\(|\)|\[|\]|\{|\}", "", doc)
    doc = re.sub("[ ]{2,}", " ", doc)

    f.seek(0)
    f.truncate(0)
    f.write(doc)
