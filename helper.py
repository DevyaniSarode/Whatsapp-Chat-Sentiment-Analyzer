# Importing modules
import streamlit as st
from urlextract import URLExtract
from wordcloud import WordCloud
import pandas as pd
from collections import Counter
import emoji
import string
import re
from datetime import datetime
from nltk.sentiment import SentimentIntensityAnalyzer
from textblob import TextBlob


# Object
extract = URLExtract()


def fetch_stats(selected_user, df):
    if selected_user == 'Overall':
        df = df[df['user'] != selected_user]

    # fetch the number of messages
    num_messages = df.shape[0]

    # fetch the total number of words
    words = []
    for message in df['message']:
        words.extend(message.split())

    # fetch number of media messages
    num_media_messages = df[df['message'] == '<Media omitted>\n'].shape[0]

    #   fetch number of links shared
    links = []
    for message in df['message']:
        links.extend(extract.find_urls(message))

    emoji_counts = []
    for message in df['message']:
       emoji_counts.extend(extract.find_urls(message))

    

    

    return num_messages, len(words), num_media_messages, len(links), len(emoji_counts)

    




def most_busy_users(df):
    x = df['user'].value_counts().head()
    df = round((df['user'].value_counts() / df.shape[0]) * 100, 2).reset_index().rename(
        columns={'index': 'name', 'user': 'percent'})
    return x, df


def remove_stop_words(message):
    f = open('stop_hinglish.txt', 'r')
    stop_words = f.read()
    y = []
    for word in message.lower().split():
        if word not in stop_words:
            y.append(word)
    return " ".join(y)


def remove_punctuation(message):
    x = re.sub('[%s]' % re.escape(string.punctuation), '', message)
    return x


# -1 => Negative
# 0 => Neutral
# 1 => Positive

# Will return count of messages of selected user per day having k(0/1/-1) sentiment
def week_activity_map(selected_user, df, k):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]
    df = df[df['value'] == k]
    return df['day_name'].value_counts()


# Will return count of messages of selected user per month having k(0/1/-1) sentiment
def month_activity_map(selected_user, df, k):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]
    df = df[df['value'] == k]
    return df['month'].value_counts()


# Will return hear map containing count of messages having k(0/1/-1) sentiment
def activity_heatmap(selected_user, df, k):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]
    df = df[df['value'] == k]

    # Creating heat map
    user_heatmap = df.pivot_table(index='day_name', columns='period', values='message', aggfunc='count').fillna(0)
    return user_heatmap


# Will return count of messages of selected user per date having k(0/1/-1) sentiment
def daily_timeline(selected_user, df, k):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]
    df = df[df['value'] == k]
    # count of message on a specific date

    daily_timeline = df.groupby('only_date').count()['message'].reset_index()
    return daily_timeline

def monthlyTimeline(selected_user, df):
    if selected_user != "Overall":
        df = df[df['user'] == selected_user]
    timeline = df.groupby(['year', 'month_num', 'month']).count()[
        'message'].reset_index()

    time = []
    for i in range(timeline.shape[0]):
        time.append(timeline['month'][i] + "-" + str(timeline['year'][i]))
    timeline['time'] = time
    return timeline

def dailyTimeline(selected_user, df):
    if selected_user != "Overall":
        df = df[df['user'] == selected_user]
    df['onlyDate'] = pd.to_datetime(df['date']).dt.date
    dailyTimeline = df.groupby("onlyDate").count()['message'].reset_index()
    return dailyTimeline







# Will return count of messages of selected user per {year + month number + month} having k(0/1/-1) sentiment
def monthly_timeline(selected_user, df, k):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]
    df = df[df['value'] == -k]
    timeline = df.groupby(['year', 'month_num', 'month']).count()['message'].reset_index()
    time = []
    for i in range(timeline.shape[0]):
        time.append(timeline['month'][i] + "-" + str(timeline['year'][i]))
    timeline['time'] = time
    return timeline


# Will return percentage of message contributed having k(0/1/-1) sentiment
def percentage(df, k):
    df = round((df['user'][df['value'] == k].value_counts() / df[df['value'] == k].shape[0]) * 100,
               2).reset_index().rename(
        columns={'index': 'name', 'user': 'percent'})
    return df


def create_wordcloud2(selected_user, df):
    f = open('stop_hinglish.txt', 'r')
    stop_words = f.read()

    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    temp = df[df['user'] != 'group_notification']
    temp = temp[temp['message'] != '<Media omitted>\n']

    def remove_stop_words(message):
        y = []
        for word in message.lower().split():
            if word not in stop_words:
                y.append(word)
        return " ".join(y)

    wc = WordCloud(width=500, height=500, min_font_size=10, background_color='white')
    temp['message'] = temp['message'].apply(remove_stop_words)
    df_wc = wc.generate(temp['message'].str.cat(sep=" "))
    return df_wc


# Return wordcloud from words in message
def create_wordcloud(selected_user, df, k):
    f = open('stop_hinglish.txt', 'r')
    stop_words = f.read()
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    # Remove entries of no significance
    temp = df[df['user'] != 'group_notification']
    temp = temp[temp['message'] != '<Media omitted>\n']

    # Remove stop words according to text file "stop_hinglish.txt"
    def remove_stop_words(message):
        y = []
        for word in message.lower().split():
            if word not in stop_words:
                y.append(word)
        return " ".join(y)

    # Dimensions of wordcloud
    wc = WordCloud(width=500, height=500, min_font_size=10, background_color='white')

    # Actual removing
    temp['message'] = temp['message'].apply(remove_stop_words)
    temp['message'] = temp['message'][temp['value'] == k]

    # Word cloud generated
    df_wc = wc.generate(temp['message'].str.cat(sep=" "))
    return df_wc


def most_common_words2(selected_user, df):
    f = open('stop_hinglish.txt', 'r')
    stop_words = f.read()

    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    temp = df[df['user'] != 'group_notification']
    temp = temp[temp['message'] != '<Media omitted>\n']

    words = []

    for message in temp['message']:
        for word in message.lower().split():
            if word not in stop_words:
                words.append(word)

    most_common_df = pd.DataFrame(Counter(words).most_common(20))
    return most_common_df


# Return set of most common words having k(0/1/-1) sentiment
def most_common_words(selected_user, df, k):
    f = open('stop_hinglish.txt', 'r')
    stop_words = f.read()
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]
    temp = df[df['user'] != 'group_notification']
    temp = temp[temp['message'] != '<Media omitted>\n']
    words = []
    for message in temp['message'][temp['value'] == k]:
        for word in message.lower().split():
            if word not in stop_words:
                words.append(word)

    # Creating data frame of most common 20 entries
    most_common_df = pd.DataFrame(Counter(words).most_common(20))
    return most_common_df


def mostEmoji(selected_user, df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]
    emojis = []
    for message in df['message']:
        if isinstance(message, str):
            message_emojized = emoji.emojize(message, language='alias')
            emojis.extend(
                [c for c in message_emojized if c in emoji.UNICODE_EMOJI['en']])

    emoji_counts = Counter(emojis)
    emoji_df = pd.DataFrame(list(emoji_counts.items()),
                            columns=['Emoji', 'Count'])
    emoji_df['Emoji'] = emoji_df['Emoji'].apply(
        lambda x: emoji.emojize(x, language='alias'))
    emoji_df = emoji_df.sort_values(
        'Count', ascending=False).reset_index(drop=True)

    return emoji_df


def weekActivity(selectedUser, df):
    if selectedUser != "Overall":
        df = df[df['user'] == selectedUser]
    week_activity = df.groupby("day_name").count()['message'].reset_index()
    return df['day_name'].value_counts(), week_activity


def monthActivity(selected_user, df):
    if selected_user != "Overall":
        df = df[df['user'] == selected_user]
    month_activity = df.groupby("month").count()['message'].reset_index()
    return df['month'].value_counts(), month_activity


def hourActivity(selected_user, df):
    if selected_user != "Overall":
        df = df[df['user'] == selected_user]
    return df.groupby(['day_name', 'hour'])['message'].count(), df.groupby(['day_name', 'hour'])[
        'message'].count().reset_index()


def activity_heatmap2(selected_user,df):

    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    user_heatmap = df.pivot_table(index='day_name', columns='period', values='message', aggfunc='count').fillna(0)

    return user_heatmap

def analyze_sentiment(selected_user, df):
    sia = SentimentIntensityAnalyzer()
    user_sentiments = {}

    for index, row in df.iterrows():
        user = row['user']
        message = row['message']

        if user not in user_sentiments:
            user_sentiments[user] = {'compound': 0.0, 'pos': 0.0, 'neu': 0.0, 'neg': 0.0}
        
        # Analyze text sentiment
        text_sentiment = sia.polarity_scores(message)
        for key in user_sentiments[user].keys():
            user_sentiments[user][key] += text_sentiment[key]

        # Analyze emoji sentiment
        emojis = [c for c in message if emoji.demojize(c) != c]
        for emo in emojis:
            emoji_sentiment = {
                                '😀': 1.0,  # Positive sentiment
                                '😢': -1.0,  # Negative sentiment
                                '😠': -1.0,  # Negative sentiment
                                '😍': 1.0,  # Positive sentiment
                                '😐': 0.0,  # Neutral sentiment
                            }
            if emo in emoji_sentiment:
                user_sentiments[user]['compound'] += emoji_sentiment[emo]

    # Average the sentiment scores
    for user, sentiment_scores in user_sentiments.items():
        total_messages = df[df['user'] == user].shape[0]
        for key in sentiment_scores.keys():
            sentiment_scores[key] /= total_messages

    if selected_user == 'Overall':
        return user_sentiments
    else:
        return {selected_user: user_sentiments.get(selected_user, {'compound': 0.0, 'pos': 0.0, 'neu': 0.0, 'neg': 0.0})}


def sentiment_score(user_sentiments, selected_user):
    if selected_user == 'Overall':
        pos = sum([user['pos'] for user in user_sentiments.values()])/len(user_sentiments)
        neg = sum([user['neg'] for user in user_sentiments.values()])/len(user_sentiments)
        neu = sum([user['neu'] for user in user_sentiments.values()])/len(user_sentiments)
    elif selected_user in user_sentiments:
        sentiment_scores = user_sentiments[selected_user]
        pos = sentiment_scores['pos']
        neg = sentiment_scores['neg']
        neu = sentiment_scores['neu']
    else:
        return "No sentiment analysis results for the selected user"

    if (pos > neg) and (pos > neu):
        return "Positive 😊"
    elif (neg > pos) and (neg > neu):
        return "Negative 😠"
    else:
        return "Neutral 🙂"



    





