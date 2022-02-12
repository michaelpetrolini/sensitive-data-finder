import nltk
import re
from langdetect import detect

FOLDER = "E:/Documenti/Uni/Magistrale/Tesi/Datasets/Altro/"
file_list = ["reddit3"]
text_type = "other"


def filter_lines():
    with open(FOLDER + text_type + "_aws3.txt", "w", encoding="utf-8") as w:
        for file in file_list:
            with open(FOLDER + file + ".txt", "r", encoding="utf-8") as r:
                for line in r:
                    line = line.encode('ascii', 'ignore').decode('utf-8')
                    line = re.sub('[ ]{2,}', ' ', line)
                    for phrase in nltk.tokenize.sent_tokenize(line):
                        if 100 <= len(phrase) <= 200:
                            try:
                                is_english = detect(phrase) == 'en'
                            except:
                                is_english = False
                            if is_english:
                                w.write(phrase + '\n')


filter_lines()
