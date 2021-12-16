import os
import matplotlib.pyplot as plt
import nltk


FOLDER = "E:/Documenti/Uni/Magistrale/Tesi/Datasets/Salute/"
file_list = ["reddit"]
text_type = "medical"


def histogram():
    len_map = {}
    max_len = 0
    for file in file_list:
        with open(FOLDER + file + ".txt", "r", encoding="utf-8") as r:
            for line in r:
                for phrase in nltk.tokenize.sent_tokenize(line):
                    length = len(phrase)

                    if max_len < length:
                        max_len = length
                    if length in len_map:
                        len_map[length] += 1
                    else:
                        len_map[length] = 1

    lengths = [0]* (max_len + 1)

    for k, v in len_map.items():
        lengths[k] = v

    total = 0
    for elem in lengths[100:200]:
        total += elem

    print(f"total: {total}")
    plt.hist(x=lengths, bins=len(lengths)//100)
    plt.show()


histogram()