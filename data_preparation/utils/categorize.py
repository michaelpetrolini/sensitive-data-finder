import os

INPUT_PATH = "C:/Users/Omi069/Downloads/input.csv"
MIDDLE_PATH = "C:/Users/Omi069/Downloads/middle.csv"
OUTPUT_PATH = "C:/Users/Omi069/Downloads/output.csv"

with open(INPUT_PATH, "r", encoding="utf-8") as r:
    data = r.read().splitlines(True)

for i in range(1, len(data)):
    def clear():
        os.system('cls')

    text = data[i]
    print(text)
    response = input('Which category? (f)inance, (p)olitical, (s)exuality, (h)ealth, (r)eligion, other?')

    if response == 'f':
        category = 'Finance'
    elif response == 'p':
        category = 'Political'
    elif response == 's':
        category = 'Sexuality'
    elif response == 'h':
        category = 'Health'
    elif response == 'r':
        category = 'Religion'
    else:
        category = 'Other'

    with open(OUTPUT_PATH, "a", encoding="utf-8") as w:
        w.write(text[:-1] + ',' + category + '\n')

    with open(MIDDLE_PATH, "w", encoding="utf-8") as w:
        w.writelines([data[0]] + data[(i + 1):])
    clear()



    