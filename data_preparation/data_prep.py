import os
import re
import string

import nltk
from nltk.corpus import stopwords

FOLDER = "E:/Documenti/Uni/Magistrale/Tesi/Datasets/Politica/"
file_list = ["20_newsgroups", "comey_hearing", "europarl", "onu", "pres_speeches", "reddit"]
text_type = "politics"


def clean():
    with open(FOLDER + text_type + "_clean.txt", "w", encoding="utf-8") as w:
        en_stop_words = stopwords.words('english')
        en_stemmer = nltk.stem.snowball.EnglishStemmer()
        for file in file_list:
            with open(FOLDER + file + ".txt", "r", encoding="utf-8") as r:
                for line in r:
                    clean_text = re.sub('[^A-Za-z ]+', ' ', line)
                    if clean_text or len(clean_text.strip()) > 20:
                        tokens = nltk.tokenize.sent_tokenize(line)

                        # lowercase
                        tokens = [x.lower() for x in tokens]

                        # remove punctuation
                        tokens = [x.translate(str.maketrans('', '', string.punctuation)) for x in tokens]

                        # remove stopwords + stemming
                        tokens = [' '.join([en_stemmer.stem(word) for word in nltk.word_tokenize(phrase)
                                            if word not in en_stop_words]) for phrase in tokens]

                        # save file
                        w.write(' '.join([token for token in tokens if len(token.strip()) > 0]) + '\n')


def manual():
    def clear():
        os.system('cls')

    counter = 0
    with open(FOLDER + text_type + "_manual.txt", "w", encoding="utf-8") as w:
        for file in file_list:
            with open(FOLDER + file + ".txt", "r", encoding="utf-8") as r:
                for line in r:
                    for phrase in nltk.tokenize.sent_tokenize(line):
                        print(f'current number of sentences: {counter}')
                        print(phrase)
                        is_correct = input(text_type + "?(y/n):")
                        if is_correct == 'y':
                            counter += 1
                            w.write(phrase + '\n')
                        clear()


def clean_split():
    with open(FOLDER + text_type + "_split.txt", "w", encoding="utf-8") as w:
        en_stop_words = stopwords.words('english')
        en_stemmer = nltk.stem.snowball.EnglishStemmer()
        for file in file_list:
            with open(FOLDER + file + ".txt", "r", encoding="utf-8") as r:
                for line in r:
                    for phrase in nltk.tokenize.sent_tokenize(line):
                        phrase = re.sub('[^A-Za-z ]+', ' ', phrase).strip()
                        phrase = ' '.join(
                            [word.strip() for word in nltk.word_tokenize(phrase) if len(word.strip()) > 0])
                        if phrase or len(phrase) > 50:
                            # lowercase
                            phrase = phrase.lower()

                            # remove punctuation
                            phrase = phrase.translate(str.maketrans('', '', string.punctuation))

                            # remove stopwords + stemming
                            phrase = ' '.join([en_stemmer.stem(word) for word in nltk.word_tokenize(phrase)
                                               if word not in en_stop_words])

                            # save file
                            if len(phrase) > 20 and len(nltk.word_tokenize(phrase)) > 1:
                                w.write(phrase + '\n')


manual()
