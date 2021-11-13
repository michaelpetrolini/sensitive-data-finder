import csv
import re
from os import listdir
from xml.etree import ElementTree as ET

from data_preparation.configs import CRIMES_FOLDER

BASE_DIR = CRIMES_FOLDER + "Legal Case Reports Dataset/corpus/"
CITATIONS_CLASS_DIR = BASE_DIR + "citations_class/"
CITATIONS_SUMM_DIR = BASE_DIR + "citations_summ/"
FULLTEXT_DIR = BASE_DIR + "fulltext/"

current_dir = FULLTEXT_DIR
files_list = listdir(current_dir)


def edit_files():
    for file_name in files_list:
        filepath = current_dir + file_name
        print(filepath)
        with open(filepath, 'r+', encoding='utf-8') as f:
            contents = f.read()
            after = re.sub(" \"id=c[0-9]*\"", "", contents)
            after = re.sub("&egrave;", "e", after)
            after = re.sub("&eacute;", "e", after)
            after = re.sub("&ndash;", "-", after)
            after = re.sub("&amp;", "", after)
            after = re.sub("&ldquo;", "\"", after)
            after = re.sub("&rdquo;", "\"", after)
            after = re.sub("&lsquo;", "\"", after)
            after = re.sub("&rsquo;", "\"", after)
            after = re.sub("&ecirc;", "e", after)
            after = re.sub("&auml;", "a", after)
            after = re.sub("&", "", after)

            f.seek(0)
            f.truncate(0)
            f.write(after)


def test_tree():
    for file_name in files_list:
        filepath = current_dir + file_name
        print(filepath)
        ET.parse(filepath)


def get_phrases():
    documents = []
    f_list = listdir(FULLTEXT_DIR)
    current_d = FULLTEXT_DIR
    for file_name in f_list:
        filepath = current_d + file_name
        tree = ET.parse(filepath)

        for catchphrase in tree.iter('catchphrase'):
            documents.append(catchphrase.text.replace("\n", " "))

        for sentence in tree.iter('sentence'):
            documents.append(sentence.text.replace("\n", " "))

    f_list = listdir(CITATIONS_SUMM_DIR)
    current_d = CITATIONS_SUMM_DIR
    for file_name in f_list:
        filepath = current_d + file_name
        tree = ET.parse(filepath)

        for citphrase in tree.iter('citphrase'):
            documents.append(citphrase.text.replace("\n", " "))

        for sentence in tree.iter('sentence'):
            if type(sentence.text) == str:
                documents.append(sentence.text.replace("\n", " "))

    f_list = listdir(CITATIONS_CLASS_DIR)
    current_d = CITATIONS_CLASS_DIR
    for file_name in f_list:
        filepath = current_d + file_name
        tree = ET.parse(filepath)

        for text in tree.iter('text'):
            if type(text.text) == str:
                documents.append(text.text.replace("\n", " "))

    return documents


def write_csv(documents):
    with open(CRIMES_FOLDER + "crimes.csv", 'w', newline='') as f:
        wr = csv.writer(f, quoting=csv.QUOTE_ALL)
        wr.writerow(documents)


def write_txt(documents):
    with open(CRIMES_FOLDER + "crimes.txt", 'w', newline='') as f:
        text = ""
        for doc in documents:
            text += doc + '\n'
        f.write(text)


docs = get_phrases()
write_txt(docs)
