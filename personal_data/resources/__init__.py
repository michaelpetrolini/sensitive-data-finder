import importlib.resources as pkg_resources
from typing import Dict


def get_template(file_name: str) -> str:
    return pkg_resources.read_text(__name__, file_name)


def get_dict(file_name: str) -> Dict[str, bool]:
    dictionary = {}

    obj_list = get_template(file_name).split(',')

    for obj in obj_list:
        dictionary[obj] = True

    return dictionary


def get_email() -> str:
    return get_template('email.txt')


def get_ipv4_address() -> str:
    return get_template('ipv4_address.txt')


def get_ipv6_address() -> str:
    return get_template('ipv6_address.txt')


def get_address() -> str:
    return get_template('address.txt')


def get_phone_number() -> str:
    return get_template('phone_number.txt')


def get_social_security_number() -> str:
    return get_template('social_security_number.txt')


def get_names() -> Dict[str, bool]:
    return get_dict('names.txt')


def get_surnames() -> Dict[str, bool]:
    return get_dict('surnames.txt')
