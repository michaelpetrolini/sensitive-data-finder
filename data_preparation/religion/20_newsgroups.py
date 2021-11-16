import re
from os import listdir

from data_preparation import MIXED_FOLDER, RELIGION_FOLDER

IN_FOLDER = MIXED_FOLDER + "20_newsgroups/"

CHRISTIAN_FOLDER = IN_FOLDER + "soc.religion.christian/"
MISC_FOLDER = IN_FOLDER + "talk.religion.misc/"

christians_list = listdir(CHRISTIAN_FOLDER)
misc_list = listdir(MISC_FOLDER)


def iterate_dir(base_folder, files):
    for file_name in files:
        print(base_folder + file_name)
        with open(base_folder + file_name, 'r', encoding='utf-8') as r:
            doc = r.read()

            doc = re.sub("<|>|\[|\]|_|-|[\.]{2,}|^|\*|=|\t", " ", doc)
            doc = re.sub("\n", " ", doc)
            doc = re.sub("[ ]{2,}", " ", doc)

            w.write(doc + '\n')


with open(RELIGION_FOLDER + "religion.txt", 'w', encoding='utf-8') as w:
    iterate_dir(CHRISTIAN_FOLDER, christians_list)

    iterate_dir(MISC_FOLDER, misc_list)
