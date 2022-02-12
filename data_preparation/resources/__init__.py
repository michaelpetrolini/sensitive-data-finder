import importlib.resources as pkg_resources
import json
from typing import List, Dict


def get_template(file_name: str) -> str:
    return pkg_resources.read_text(__name__, file_name)


def health_subreddits() -> List[str]:
    return get_template("health.txt").split(',')


def politics_subreddits() -> List[str]:
    return get_template("politics.txt").split(',')


def religion_subreddits() -> List[str]:
    return get_template("religion.txt").split(',')


def sexuality_subreddits() -> List[str]:
    return get_template("sexuality.txt").split(',')


def reddit_specs() -> Dict[str, str]:
    return json.loads(get_template('reddit_specs.json'))
