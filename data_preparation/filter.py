import nltk
import re
from langdetect import detect

MIN_LEN, MAX_LEN = 100, 200
arguments = ['health', 'religion', 'politics', 'sexuality', 'other']

for argument in arguments:
    with open(f"{argument}_aws.txt", "w", encoding="utf-8") as w:
        with open(f"{argument}.txt", "r", encoding="utf-8") as r:
            for line in r:
                line = line.encode('ascii', 'ignore').decode('utf-8')
                line = re.sub('[ ]{2,}', ' ', line)

                for phrase in nltk.tokenize.sent_tokenize(line):
                    if MIN_LEN <= len(phrase) <= MAX_LEN:
                        try:
                            if detect(phrase) == 'en':
                                w.write(phrase + '\n')
                        except Exception as e:
                            print(e)
                            continue
