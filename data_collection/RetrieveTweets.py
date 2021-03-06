"""
Downloads all tweets from a given user.
Uses twitter.Api.GetUserTimeline to retreive the last 3,200 tweets from a user.
Twitter doesn't allow retreiving more tweets than this through the API, so we get
as many as possible.
t.py should contain the imported variables.
"""
from __future__ import print_function

import pandas as pd
import twitter
import os
API_KEY = os.environ["API_KEY"]
API_SECRET = os.environ["API_SECRET"]
ACCESS_TOKEN = os.environ["ACCESS_TOKEN"]
TOKEN_SECRET = os.environ["TOKEN_SECRET"]
def get_tweets(api=None, screen_name=None):
    timeline = api.GetUserTimeline(screen_name=screen_name, count=100)
    earliest_tweet = min(timeline, key=lambda x: x.id).id

    # while True:
    #     tweets = api.GetUserTimeline(
    #         screen_name=screen_name, max_id=earliest_tweet, count=10
    #     )
    #     new_earliest = min(tweets, key=lambda x: x.id).id

    #     if not tweets or new_earliest == earliest_tweet:
    #         break
    #     else:
    #         earliest_tweet = new_earliest
    #         timeline += tweets

    return timeline

def create_dataframe(columns, handles, api):
    data = []   
    for handle in handles:
        print(f"Saving tweets from account @{handle}")
        timeline = get_tweets(api, handle)
        for tweet in timeline:
            tweet = tweet._json # convert to py dict
            row = list()
            row.append(tweet["text"].replace("\n", "").replace("\t", "").replace("\r", ""))
            row += [tweet[col] for col in columns if col not in ("username", "text")]
            row.append(handle)

            data.append(row)

    df = pd.DataFrame(data=data, columns=columns)

    return df

if __name__ == "__main__":
    api = twitter.Api(
        API_KEY, API_SECRET, ACCESS_TOKEN, TOKEN_SECRET
    )

    columns = ["text", "is_quote_status", "retweet_count", "favorite_count", "favorited", "retweeted", "username"]
    
    handles = [
      "lisarinna",
      "KyleRichards",
      "erikajayne",
      "GarcelleB",
      "doritkemsley1",
      "crystalsminkoff",
      "DENISE_RICHARDS",
      "SuttonBStracke",
      "YolandaHadid",
    ]

    df = create_dataframe(columns, handles, api)
    df.to_csv("tweets.csv", sep=",", mode="w+", index=False)

