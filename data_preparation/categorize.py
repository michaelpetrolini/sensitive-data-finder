import os


with open("input.csv", "r", encoding="utf-8") as r:
    data = r.read().splitlines(True)

categories = {'h': 'Health',
              'p': 'Politics',
              'r': 'Religion',
              's': 'Sexuality'}

for i in range(1, len(data)):
    def clear():
        os.system('cls')

    text = data[i]
    print(text)
    response = input('Which category? (f)inance, (p)olitical, (s)exuality, (h)ealth, (r)eligion, other?')

    if response in categories.keys():
        category = categories[response]
    else:
        category = 'Other'

    with open("output.csv", "a", encoding="utf-8") as w:
        w.write(text[:-1] + ',' + category + '\n')

    with open("middle.csv", "w", encoding="utf-8") as w:
        w.writelines([data[0]] + data[(i + 1):])
    clear()
