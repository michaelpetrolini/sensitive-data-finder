import praw

from data_preparation import resources

reddit_specs = resources.reddit_specs()
reddit = praw.Reddit(client_id=reddit_specs['client_id'],
                     client_secret=reddit_specs['client_secret'],
                     user_agent=reddit_specs['user_agent'],
                     username=reddit_specs['username'],
                     password=reddit_specs['password'])

health = resources.health_subreddits()
politics = resources.politics_subreddits()
religion = resources.religion_subreddits()
sexuality = resources.sexuality_subreddits()

subreddits = {'health': health,
              'politics': politics,
              'religion': religion,
              'sexuality': sexuality}

keep_out = health + politics + religion + sexuality


def gather_subreddits_posts(category: str):
    with open(f"{category}.txt", "w", encoding="utf-8") as w:
        for subreddit in subreddits[category]:
            print(subreddit)
            hot_posts = reddit.subreddit(subreddit).hot(limit=10000)

            for submission in hot_posts:
                write_submission(submission, w)


def gather_unrelated_posts():
    with open("other.txt", "w", encoding="utf-8") as w:
        hot_posts = reddit.subreddit("all").hot(limit=50000)

        for submission in hot_posts:
            s_name = str(submission.subreddit)
            if s_name not in keep_out:
                print(s_name)
                write_submission(submission, w)


def write_submission(submission, w):
    submission.comments.replace_more(limit=0)
    post = f"{submission.title.strip()}. {submission.selftext.strip()}. "
    comments = [comment.body.strip() for comment in submission.comments.list()]
    entry = (post.join(comments)).replace("\r", " ").replace("\n", " ")
    w.write(entry + "\n")


for argument in subreddits.keys():
    gather_subreddits_posts(argument)

gather_unrelated_posts()
