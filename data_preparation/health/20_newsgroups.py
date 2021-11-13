import re
from os import listdir

from data_preparation.configs import MIXED_FOLDER, HEALTH_FOLDER

BASE_DIR = MIXED_FOLDER + "20_newsgroups/sci.med/"
files_list = listdir(BASE_DIR)

with open(HEALTH_FOLDER + "20_newsgroups.txt", 'w', encoding='utf-8') as w:
    for file_name in files_list:
        print(file_name)
        with open(BASE_DIR + file_name, 'r', encoding='utf-8') as r:
            doc = r.read()

            doc = re.sub(".*:.*", "", doc)
            doc = re.sub("<|>|\[|\]|_|-|[\.]{2,}|^|\*|=|\t", " ", doc)
            doc = re.sub("\n", " ", doc)
            doc = re.sub("[ ]{2,}", " ", doc)

            w.write(doc + '\n')
