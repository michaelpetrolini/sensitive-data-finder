import re
import string

import nltk
from nltk.corpus import stopwords

FOLDER = "E:/Documenti/Uni/Magistrale/Tesi/Datasets/Politica/"
file_list = ["20_newsgroups", "comey_hearing", "europarl", "onu", "pres_speeches"]

with open(FOLDER + "politics_final.txt", "w", encoding="utf-8") as w:
    en_stop_words = stopwords.words('english')
    en_stemmer = nltk.stem.snowball.EnglishStemmer()
    for file in file_list:
        with open(FOLDER + file + ".txt", "r", encoding="utf-8") as r:
            for line in r:
                clean_text = re.sub('[^A-Za-z0-9 ]+', '', line)
                if clean_text:
                    tokens = nltk.tokenize.sent_tokenize(line)

                    # lowercase
                    tokens = [x.lower() for x in tokens]

                    # remove punctuation
                    tokens = [x.translate(str.maketrans('', '', string.punctuation)) for x in tokens]

                    # remove stopwords
                    tokens = [' '.join([word for word in nltk.word_tokenize(phrase)
                                        if word not in en_stop_words]) for phrase in tokens]

                    # stemming
                    tokens = [' '.join([en_stemmer.stem(word) for word in nltk.word_tokenize(phrase)])
                              for phrase in tokens]

                    # save file
                    w.write('. '.join(tokens) + '\n')
