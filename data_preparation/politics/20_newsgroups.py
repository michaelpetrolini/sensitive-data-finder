import re
from os import listdir

from data_preparation import MIXED_FOLDER, POLITICS_FOLDER

IN_FOLDER = MIXED_FOLDER + "20_newsgroups/"

GUNS_FOLDER = IN_FOLDER + "talk.politics.guns/"
MIDEAST_FOLDER = IN_FOLDER + "talk.politics.mideast/"
MISC_FOLDER = IN_FOLDER + "talk.politics.misc/"

guns_list = listdir(GUNS_FOLDER)
mideast_list = listdir(MIDEAST_FOLDER)
misc_list = listdir(MISC_FOLDER)


def iterate_dir(base_folder, files):
    for file_name in files:
        print(base_folder + file_name)
        with open(base_folder + file_name, 'r', encoding='utf-8') as r:
            doc = r.read()

            doc = re.sub(".*:.*", "", doc)
            doc = re.sub("<|>|\[|\]|_|-|[\.]{2,}|^|\*|=|\t", " ", doc)
            doc = re.sub("\n", " ", doc)
            doc = re.sub("[ ]{2,}", " ", doc)

            w.write(doc + '\n')


with open(POLITICS_FOLDER + "20_newsgroups.txt", 'w', encoding='utf-8') as w:
    iterate_dir(GUNS_FOLDER, guns_list)

    iterate_dir(MIDEAST_FOLDER, mideast_list)

    iterate_dir(MISC_FOLDER, misc_list)
