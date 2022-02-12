import matplotlib.pyplot as plt
import nltk

FOLDER = "C:/Users/Omi069/Downloads/"
file_list = ["total_aws2"]


def histogram():
    lengths = []
    for file in file_list:
        with open(FOLDER + file + ".txt", "r", encoding="utf-8") as r:
            for line in r:
                length = len(nltk.tokenize.word_tokenize(line))
                lengths.append(length)

    plt.hist(lengths, bins=750, color='palegreen', edgecolor='darkseagreen', linewidth=1.2)
    plt.show()


histogram()
