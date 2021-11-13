import re
import string
from typing import List, Dict

import resources

name_page = "https://en.wikipedia.org/w/api.php?action=query&list=categorymembers" \
            "&cmtitle=Category:English%20masculine%20given%20names&cmprop=title&format=json&cmlimit=500"


def look_for_pattern(text: str, pattern: str) -> List[str]:
    return re.findall(pattern, text)


def check_dictionary(tokens: List[str], dictionary: Dict[str, bool]) -> List[str]:
    result = []

    for token in tokens:
        if token in dictionary:
            result.append(token)

    return result


def tokenize(text: str) -> List[str]:
    return text.translate(str.maketrans('', '', string.punctuation)).split(' ')


class PersonalDataFinder:

    def __init__(self):
        self.email_pattern = resources.get_email()
        self.ipv4 = resources.get_ipv4_address()
        self.ipv6 = resources.get_ipv6_address()
        self.address = resources.get_address()

        self.names = resources.get_names()
        self.surnames = resources.get_surnames()

    def look_for_emails(self, text: str) -> List[str]:
        return look_for_pattern(text, self.email_pattern)

    def look_for_ip_address(self, text: str) -> List[str]:
        return look_for_pattern(text, self.ipv4) + look_for_pattern(text, self.ipv6)

    def look_for_address(self, text: str) -> List[str]:
        return look_for_pattern(text, self.address)

    def look_for_names(self, text: str) -> List[str]:
        return check_dictionary(tokenize(text), self.names)

    def look_for_surnames(self, text: str) -> List[str]:
        return check_dictionary(tokenize(text), self.surnames)
