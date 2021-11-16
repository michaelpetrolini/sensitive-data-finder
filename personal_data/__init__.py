import re
import string
from typing import List, Dict

import resources


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
        self.phone_number = resources.get_phone_number()
        self.ssn = resources.get_social_security_number()

        self.names = resources.get_names()
        self.surnames = resources.get_surnames()

    def look_for_emails(self, text: str) -> List[str]:
        return look_for_pattern(text, self.email_pattern)

    def look_for_ip_address(self, text: str) -> List[str]:
        return look_for_pattern(text, self.ipv4) + look_for_pattern(text, self.ipv6)

    def look_for_address(self, text: str) -> List[str]:
        return look_for_pattern(text, self.address)

    def look_for_phone_number(self, text: str) -> List[str]:
        return list(map(lambda x: x[0], look_for_pattern(text, self.phone_number)))

    def look_for_ssn(self, text: str) -> List[str]:
        return look_for_pattern(text, self.ssn)

    def look_for_names(self, text: str) -> List[str]:
        return check_dictionary(tokenize(text), self.names)

    def look_for_surnames(self, text: str) -> List[str]:
        return check_dictionary(tokenize(text), self.surnames)


_personal_data_finder = PersonalDataFinder()


class PersonalData:

    def __init__(self, text: str):
        self.text = text

        self.emails = _personal_data_finder.look_for_emails(text)
        self.ip_addresses = _personal_data_finder.look_for_ip_address(text)
        self.addresses = _personal_data_finder.look_for_address(text)
        self.phone_numbers = _personal_data_finder.look_for_phone_number(text)
        self.ssn = _personal_data_finder.look_for_ssn(text)
        self.names = _personal_data_finder.look_for_names(text)
        self.surnames = _personal_data_finder.look_for_surnames(text)

    def is_empty(self) -> bool:
        return len(self.emails + self.ip_addresses + self.addresses + self.phone_numbers + self.ssn + self.names +
                   self.surnames) == 0
