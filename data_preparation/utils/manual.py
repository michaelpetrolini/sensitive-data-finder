import os
import nltk


FOLDER = "E:/Documenti/Uni/Magistrale/Tesi/Datasets/Salute/"
file_list = ["reddit"]
text_type = "medical"
MIN_LEN = 100
MAX_LEN = 200


def manual():
    def clear():
        os.system('cls')

    counter = 0
    with open(FOLDER + text_type + "_manual.txt", "a", encoding="utf-8") as w:
        for file in file_list:
            with open(FOLDER + file + ".txt", "r", encoding="utf-8") as r:
                for line in r:
                    for phrase in nltk.tokenize.sent_tokenize(line):
                        if MIN_LEN <= len(phrase) <= MAX_LEN:
                            print(f'current number of sentences: {counter}')
                            print(phrase)
                            response = input(text_type + "?(y/n):")
                            if response == 'y':
                                counter += 1
                                w.write(phrase + '\n')
                                w.flush()
                            clear()


manual()