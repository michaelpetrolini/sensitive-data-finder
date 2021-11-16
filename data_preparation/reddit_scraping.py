import importlib.resources as pkg_resources
import json

import praw

import data_preparation

FOLDER = "E:/Documenti/Uni/Magistrale/Tesi/Datasets/Sessualità/"
# religion = ["religion", "Christianity", "Christian", "TrueChristian", "atheism", "islam"]

# sexuality = ["lgbt", "gay", "lesbian", "bisexual", "asktransgender", "transgender", "askgaybros", "actuallesbians",
# "ainbow", "LesbianActually", "gaybros", "LGBTeens", "queer", "sexuality", "sex", "relationships",
#               "ldssexuality", "asexuality"]

reddit_specs = json.loads(pkg_resources.read_text(data_preparation, 'reddit_specs.json'))
reddit = praw.Reddit(client_id=reddit_specs['client_id'],
                     client_secret=reddit_specs['client_secret'],
                     user_agent=reddit_specs['user_agent'],
                     username=reddit_specs['username'],
                     password=reddit_specs['password'])


subreddits = ["lgbt", "gay", "lesbian", "bisexual", "asktransgender", "transgender", "askgaybros", "actuallesbians",
              "ainbow", "LesbianActually", "gaybros", "LGBTeens", "queer", "sexuality", "sex", "relationships",
              "ldssexuality", "asexuality"]
with open(FOLDER + "reddit.txt", "a", encoding="utf-8") as w:
    for subreddit in subreddits:
        hot_posts = reddit.subreddit(subreddit).hot(limit=10000)
        for post in hot_posts:
            print(post.title)
            submission = reddit.submission(id=post.id)
            submission.comments.replace_more(limit=0)

            entry = (post.title.strip() + '. ' + post.selftext.strip() + '. ' + '. '.join([
                comment.body.strip() for comment in submission.comments.list()])).replace("\r", " ").replace("\n", " ")

            w.write(entry + "\n")
