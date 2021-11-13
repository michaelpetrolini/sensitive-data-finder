import re
from os import listdir

from data_preparation.configs import POLITICS_FOLDER

BASE_DIR = POLITICS_FOLDER + "europarl/en/"

files_list = listdir(BASE_DIR)

with open(POLITICS_FOLDER + "europarl.txt", 'w', encoding='utf-8') as w:
    for filename in files_list:
        print(filename)
        with open(BASE_DIR + filename, 'r', encoding='utf-8') as r:
            doc = r.read()
            doc = re.sub("<.*>", "", doc)
            doc = re.sub("/(.*/)", "", doc)
            doc = re.sub("\n", " ", doc)
            doc = re.sub("[ ]{2,}", " ", doc)
            w.write(doc + '\n')
