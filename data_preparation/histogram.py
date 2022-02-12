import matplotlib.pyplot as plt
import nltk

arguments = ['health', 'religion', 'politics', 'sexuality', 'other']


def words_count():
    lengths = []
    for argument in arguments:
        with open(f"{argument}.txt", "r", encoding="utf-8") as r:
            for line in r:
                length = len(nltk.tokenize.word_tokenize(line))
                lengths.append(length)
    return lengths


def chars_count():
    lengths = []
    for argument in arguments:
        with open(f"{argument}.txt", "r", encoding="utf-8") as r:
            for line in r:
                lengths.append(len(line))
    return lengths


plt.hist(words_count(), bins=750, color='palegreen', edgecolor='darkseagreen', linewidth=1.2)
plt.savefig("words_count.png")

plt.hist(chars_count(), bins=750, color='wheat', edgecolor='tan', linewidth=1.2)
plt.savefig("chars_count.png")
