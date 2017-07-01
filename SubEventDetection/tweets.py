# encoding=utf8
import tweepy
import time
import csv
import pandas as pd

TWITTER_CONSUMER_KEY="" 
TWITTER_CONSUMER_SECRET=""
TWITTER_ACCESS_KEY=""
TWITTER_ACCESS_SECRET=""

# Create an output csv file and define headers
output = csv.writer(open('tweets.csv', 'w'))
output.writerow(['Tweet Created At', 'Tweet Text'])


# Twitter Authentication
auth = tweepy.OAuthHandler(TWITTER_CONSUMER_KEY, TWITTER_CONSUMER_SECRET)
auth.set_access_token(TWITTER_ACCESS_KEY, TWITTER_ACCESS_SECRET)
api = tweepy.API(auth)

# Define the search parameters for tweepy
search_tag = 'Brexit'
since_date = '2016-06-23'


def clean_nonascii(tweet):
    if tweet:
        try:
            value = ''.join(char for char in tweet if 32 <= ord(char) < 128)
            return value
        except:
            value = unicode(tweet, "utf-8")
            return value
        else:
            pass
    else:
        return tweet

# count = 0


def save_tweets():
    count = 1
    rate_limit_count = 1
    t_obj = tweepy.Cursor(api.search, q=search_tag, lang="en",
                          since_id=since_date).items()
    # until=until_date
    # Twitter has rate limit. Check if there is any rate limit happening.

    while True:
        try:
            tweet = t_obj.next()
            
        except tweepy.TweepError:
            print(rate_limit_count, ". Rate Limited")
            time.sleep(60 * 15)
            continue
        except StopIteration:
            break

        if tweet:
            # User Details
            print(count, ". Got next tweet object")
            count += 1
            # Tweet Details
            tweet_text = clean_nonascii(tweet.text)
            tweet_created_at = tweet.created_at
            if tweet_created_at:
                tweet_created_at = pd.to_datetime(tweet_created_at)

            # Write data to csv file
            # print tweet
            # print "*********\n"
            output.writerow([tweet_created_at, tweet_text.encode('utf-8')])


if __name__ == '__main__':
    save_tweets()
