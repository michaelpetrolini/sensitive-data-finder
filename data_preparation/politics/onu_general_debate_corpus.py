import re
from os import listdir

from data_preparation import POLITICS_FOLDER

BASE_DIR = POLITICS_FOLDER + "ian-united-nations-general-debate-corpus/original/Converted sessions/"

folders_list = listdir(BASE_DIR)

with open(POLITICS_FOLDER + "onu.txt", 'w', encoding='utf-8') as w:
    for folder_name in folders_list:
        print(folder_name)
        current_folder = BASE_DIR + folder_name + '/'
        files_list = listdir(current_folder)
        for filename in files_list:
            print("   " + filename)
            with open(current_folder + filename, 'r', encoding='utf-8') as r:
                doc = r.read()
                doc = re.sub("\n", " ", doc)
                doc = re.sub("[ ]{2,}", " ", doc)
                w.write(doc + '\n')
