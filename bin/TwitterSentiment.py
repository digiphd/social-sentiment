import tweepy
import configparser
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import pipeline
from scipy.special import softmax
import time
import math
import pandas as pd


# read config

config = configparser.ConfigParser(interpolation=None)
config.read('config.ini')

api_key = config['twitter']['api_key']
api_key_secret = config['twitter']['api_key_secret']
access_token = config['twitter']['access_token']
access_token_secret = config['twitter']['access_token_secret']
bearer_token = config['twitter']['bearer_token']
MODEL = "cardiffnlp/twitter-roberta-base-sentiment-latest"


# authentication with twitter

class twitterSentiment():
    """
    Calculates the average sentiment of a particular search query via analysing many tweets, output is Positive %
    Neutral % and Negative %.
    """

    def __init__(self, query, max_results=100):
        self.max_results = max_results
        self.query = query

        self.client = tweepy.Client(
            bearer_token=bearer_token,
            consumer_key=api_key,
            consumer_secret=api_key_secret,
            access_token=access_token,
            access_token_secret=access_token_secret
        )



    def searchTweets(self):
        client = self.client
        tweet_fields = ['created_at', 'text', 'author_id', 'context_annotations','geo', 'lang', 'public_metrics']
        # tweet_fields = ['public_mentrics']


        # Get Restuls with Pagination
        if self.max_results >100:
            #calculate limit

            public_tweets = []
            limit = math.ceil(max_results/100)
            paginator = tweepy.Paginator(
                client.search_recent_tweets,
                query=self.query,
                max_results=100,
                tweet_fields=tweet_fields,
                expansions=['author_id'],
                limit=limit
            )
            for tweet in paginator.flatten(limit=max_results):  # Total number of tweets to retrieve

                public_tweets.extend([tweet])
        else:
            public_tweets_response = client.search_recent_tweets(query=self.query, max_results=max_results, tweet_fields=tweet_fields,
                                                    expansions=['author_id'])
            public_tweets = public_tweets_response.data

        tweets_dict = []
        # print(public_tweets)
        for tweet_object in public_tweets:

            tweet_words = []
            for word in tweet_object['text'].split(' '):
                if word.startswith('@') and len(word) > 1:
                    word = '@user'
                elif word.startswith('http'):
                    word = 'http'
                tweet_words.append(word)

            tweet_join = " ".join(tweet_words)
            tweet_proc = tweet_join.replace('\n', '')
            # user_metrics = getUserMetrics(client, tweet_object)
            tweets_dict.extend([{'author_id':tweet_object.author_id, 'text':  tweet_proc, 'created_at': tweet_object.created_at}])
        tweet_dict, time_days = self.getUserMetrics(tweets_dict)


        return tweets_dict, time_days

    def getUserMetrics(self, tweet_dict):
        user_fields = ['location', 'public_metrics']
        # print(tweet_object)
        tweet_df = pd.DataFrame.from_dict(tweet_dict)

        df_len = tweet_df.shape[0]

        index = 0
        if df_len > 100:
            chunk_size = math.ceil(df_len / 100)
            n = 100
            list_df = [tweet_df[i:i + n] for i in range(0, tweet_df.shape[0], n)]


            for df in list_df:
                author_ids = df['author_id'].values.tolist()
                user_details = self.client.get_users(ids=author_ids, user_fields=user_fields)
                for i, user in enumerate(user_details.data):
                    location = user.location
                    tweet_dict[index]['user_metrics'] = {'location': location,
                                                     'followers': user.public_metrics['followers_count'],
                                                     'tweet_count': user.public_metrics['tweet_count']}
                    index+=1
        else:

            author_ids = tweet_df['author_id'].values.tolist()
            user_details = self.client.get_users(ids=author_ids, user_fields=user_fields)

            for i,user in enumerate(user_details.data):
                location = user.location
                tweet_dict[i]['user_metrics'] = {'location': location, 'followers':user.public_metrics['followers_count'], 'tweet_count':user.public_metrics['tweet_count']}

        min_max_dates = tweet_df['created_at'].agg(['min', 'max'])
        time_days = min_max_dates[1] - min_max_dates[0]

        # import pdb; pdb.set_trace()
        return tweet_dict, time_days



    def classifyTweet(self, tweet):

        # Preprocess tweet
        tweet_words = []
        for word in tweet.split(' '):
                if word.startswith('@') and len(word) > 1:
                    word = '@user'
                elif word.startswith('http'):
                    word = 'http'
                tweet_words.append(word)

        tweet_proc = " ".join(tweet_words)

        # load model

        model = AutoModelForSequenceClassification.from_pretrained(MODEL)
        # model.save_pretrained(roberta)
        tokenizer = AutoTokenizer.from_pretrained(MODEL)
        # tokenizer.save_pretrained(roberta)

        labels = ['Negative', 'Neutral', 'Postitive']

        # sentiment analysis
        encoded_tweet = tokenizer(tweet_proc, return_tensors='pt')

        output = model(encoded_tweet['input_ids'], encoded_tweet['attention_mask'])

        scores = output[0][0].detach().numpy()

        scores = softmax(scores)

        for i in range(len(scores)):
            l = labels[i]
            s = scores[i]

            print(l, s)

    def classifyTweetsPipeline(self, tweets_dict):
        sentiment_task = pipeline("sentiment-analysis", model=MODEL, tokenizer=MODEL)

        classified_tweets = []
        for tweet in tweets_dict:
            results = sentiment_task(tweet['text'])
            tweet['sentiment_values'] = results[0]
            tweet['sentiment'] = results[0]['label']

            classified_tweets.extend([tweet])
        # print(classified_tweets)
        return classified_tweets

    def calculateAverage(self, classified_tweets):

        positive = 0
        neutral = 0
        negative = 0
        for tweet in classified_tweets:

            if tweet['sentiment'] == 'Positive':
                positive+=1
            elif tweet['sentiment'] == 'Neutral':
                neutral+=1
            elif tweet['sentiment'] == 'Negative':
                negative+=1
            else:
                print(tweet['sentiment'])

        print("Overall Sentiment: ", "Positive %: "+str(positive/len(classified_tweets)), "Neutral %: "+str(neutral/len(classified_tweets)), "Negative %: "+str(negative/len(classified_tweets)))


    def launchAnalysis(self):
        st = time.time()
        tweets_dict, time_days = self.searchTweets()

        classified_tweets = self.classifyTweetsPipeline(tweets_dict)
        # Print individual tweet results for debugging purposes
        # for i, tweet in enumerate(classified_tweets):
        #     # print(i, tweet['user_metrics']['location'], tweet['text'], tweet['sentiment'], '\n')
        #     # print(i, ",Locale:" + str(tweet['user_metrics']['location']), ",Sentiment:" + tweet['sentiment'])

        print(str(len(tweets_dict)) + " Tweets processed from the last " + str(time_days) + " for the query: "+ query )
        self.calculateAverage(classified_tweets)

        et = time.time()

        # get the execution time
        elapsed_time = et - st

        time_per_tweet = elapsed_time/len(tweets_dict)
        print('Execution time:', elapsed_time, 'seconds, Approx '+str(time_per_tweet)+' secs per tweet')

        return classified_tweets


# query = "#greens OR #alp lang:en"
query = "#lnp lang:en"
max_results = 1000

out = twitterSentiment(query, max_results)
out.launchAnalysis()
