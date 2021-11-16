from personal_data import PersonalData


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    text_list = []

    for text in text_list:
        personal_data = PersonalData(text)
        print(personal_data)

        if not personal_data.is_empty():
            print("text contains personal data")
    print_hi('PyCharm')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
