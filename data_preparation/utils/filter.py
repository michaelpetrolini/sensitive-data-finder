import nltk

FOLDER = "E:/Documenti/Uni/Magistrale/Tesi/Datasets/Economia/"
file_list = ["reddit"]
text_type = "finance"
MIN_LEN = 100
MAX_LEN = 200


def filter_lines():
    with open(FOLDER + text_type + "_aws.txt", "w", encoding="utf-8") as w:
        for file in file_list:
            with open(FOLDER + file + ".txt", "r", encoding="utf-8") as r:
                for line in r:
                    line = line.encode('ascii', 'ignore').decode('utf-8')
                    for phrase in nltk.tokenize.sent_tokenize(line):
                        if MIN_LEN <= len(phrase) <= MAX_LEN:
                            w.write(phrase + '\n')


filter_lines()
