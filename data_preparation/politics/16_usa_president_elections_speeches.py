import re

from data_preparation.configs import POLITICS_FOLDER

BASE_DIR = POLITICS_FOLDER + "josiah-16-pres-election-speeches/original/"

files_list = ["clinton_concession.txt", "trump_acceptance.txt"]

with open(POLITICS_FOLDER + "pres_speeches.txt", 'w', encoding='utf-8') as w:
    for filename in files_list:
        print(filename)
        with open(BASE_DIR + filename, 'r', encoding='utf-8') as r:
            doc = r.read()
            doc = re.sub("\n", " ", doc)
            doc = re.sub("[ ]{2,}", " ", doc)
            w.write(doc + '\n')
