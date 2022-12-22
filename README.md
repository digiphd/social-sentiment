# WHAT IS THIS APP?
Check out the live version of [this app](https://digiphd-social-sentiment-main-zdymf5.streamlit.app/).

This app was a fun project to simply understand twitters community sentiment accross any issue. 

Note: Sentiment scores do in no way indicate my views or oppinions or that of Twitter or Streamlit

# HOW TO RUN LOCALLY

## Add Your Twitter Credentials
1. Go to the [Twitter Dev Portal](https://developer.twitter.com/en/portal/dashboard) and create a project + app and get your keys and tokens
2. Create a file `.streamlit/secrets.toml` that looks like this
```
[twitter]

api_key = "enter your credentials"
api_key_secret = "enter your credentials"
access_token = "enter your credentials"
access_token_secret = "enter your credentials"
bearer_token = "enter your credentials"

```

# WHAT I LEARNED
- Streamlit
- Twitter API
- Hugging Face sentiment models

# Notes:
The [hugging face model](https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment-latest) this uses is 500MB 
and will download the first time you load the application in Streamlit locally. So just keep that mind.
Fun Fact: It is trained on over 124M Tweets from Jan 2018 to Dec 2021.