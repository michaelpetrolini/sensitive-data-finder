from os import listdir

from data_preparation import FINANCE_FOLDER

BASE_DIR = FINANCE_FOLDER + "2013-2016 CleanedParsed 10-K Filings with the SEC/"

files_list = listdir(BASE_DIR)

with open(FINANCE_FOLDER + "finance.txt", 'w', encoding='utf-8') as w:
    for filename in files_list:
        with open(BASE_DIR + filename, 'r', encoding='utf-8') as r:
            doc = r.read()
            w.write(doc + '\n')
