import pandas as pd
import streamlit as st

from bin.TwitterSentiment import twitterSentiment

st.set_page_config(page_title="Transcribe", page_icon="💬‍")
st.markdown("# Check Social Sentiment on Any Issue 💬")

c1, c2, c3 = st.columns([1, 4, 1])

with c2:
    input = st.text_input("Search Term or Hashtag", value="football")
    input = '\"'+input+'\"'
    value = st.slider('Number of tweets to Analyse', min_value=100, max_value=500)
    result = st.button("Get Sentiment")

    if result:
        out = twitterSentiment(input, value)
        result = out.launchAnalysis()
if result:

    # Work out Final Sentiment and Map to an Emoji
    m_df = pd.DataFrame([result['averages']])
    sentiment = m_df.idxmax(axis=1)[0]
    st.title('Community seems '+sentiment+' on '+input)
    col1, col2, col3 = st.columns(3)
    if sentiment == 'negative':
        col2.metric("Negative", str(round(result['averages']['negative'] * 100.00, 2)) + '%')
        col2.image('images/negative.png', width=200)
    elif sentiment == 'neutral':
        col2.metric("Neutral", str(round(result['averages']['neutral'] * 100.00, 2)) + '%')
        col2.image('images/neutral.png', width=200)
    else:
        col2.metric("Positive", str(round(result['averages']['positive'] * 100.00, 2)) + '%')
        col2.image('images/positive.png', width=200)
    with st.expander("More Info"):
        col1, col2, col3 = st.columns(3)
        col1.metric("Positive", str(round(result['averages']['positive'] * 100.00, 2)) + '%')
        col2.metric("Neutral", str(round(result['averages']['neutral'] * 100.00, 2)) + '%')
        col3.metric("Negative", str(round(result['averages']['negative'] * 100.00, 2)) + '%')
        st.write(result['message'])
        st.write(result['time_message'])
    st.write("Note: Sentiment on topics presented here does not express my views or those of Twitter or Streamlit")







