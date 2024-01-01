# Importing modules
import nltk
import streamlit as st
import re
from matplotlib import cm
from wordcloud import WordCloud
import emoji
import preprocessor, helper
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from textblob import TextBlob
def categorize_sentiment(score):
    if score > 0:
        return 'Positive'
    elif score == 0:
        return 'Neutral'
    else:
        return 'Negative'




# App title
st.sidebar.title("Whatsapp Chat Analyzer🔎")

# VADER : is a lexicon and rule-based sentiment analysis tool that is specifically attuned to sentiments.
nltk.download('vader_lexicon')

# File upload button
uploaded_file = st.sidebar.file_uploader("Choose a file 🗃️")

# Main heading
st.markdown("<h1 style='text-align: center; color: grey;'>Whatsapp Chat Analyzer 🔎</h1>",
            unsafe_allow_html=True)

if uploaded_file is not None:

    # Getting byte form & then decoding
    bytes_data = uploaded_file.getvalue()
    d = bytes_data.decode("utf-8")

    # Perform preprocessing
    data = preprocessor.preprocess(d)
    st.dataframe(data)
    # Importing SentimentIntensityAnalyzer class from "nltk.sentiment.vader"
    from nltk.sentiment.vader import SentimentIntensityAnalyzer

    # Object
    sentiments = SentimentIntensityAnalyzer()

    # Creating different columns for (Positive/Negative/Neutral)
    data["po"] = [sentiments.polarity_scores(i)["pos"] for i in data["message"]]  # Positive
    data["ne"] = [sentiments.polarity_scores(i)["neg"] for i in data["message"]]  # Negative
    data["nu"] = [sentiments.polarity_scores(i)["neu"] for i in data["message"]]  # Neutral


    # To indentify true sentiment per row in message column
    def sentiment(d):
        if d["po"] >= d["ne"] and d["po"] >= d["nu"]:
            return 1
        if d["ne"] >= d["po"] and d["ne"] >= d["nu"]:
            return -1
        if d["nu"] >= d["po"] and d["nu"] >= d["ne"]:
            return 0


    # Creating new column & Applying function
    data['value'] = data.apply(lambda row: sentiment(row), axis=1)

    # User names list
    user_list = data['user'].unique().tolist()

    # Sorting
    user_list.sort()

    # Insert "Overall" at index 0
    user_list.insert(0, "Overall")

    # Selectbox
    selected_user = st.sidebar.selectbox("Show analysis 🤔", user_list)

    if st.sidebar.button("Show Analysis 🔢"):
        # Stats Area
        num_messages, words, num_media_messages, num_links, emoji_counts = helper.fetch_stats(selected_user, data)
        st.title("Top Statistics 📈")

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.header("Total Messages🤳🏻")
            st.title(num_messages)

        with col2:
            st.header("Total Words💭")
            st.title(words)

        with col3:
            st.header("Media Shared🎥")
            st.title(num_media_messages)
        with col4:
            st.header("Links Shared🔗")
            st.title(num_links)
        with col1:
            st.header("Emojis Shared😳")
            st.title(emoji_counts)
        

        # activity map
        st.title('Activity Map')
        col1, col2 = st.columns(2)

        with col1:
            st.header("Most busy day")
            busy_day = helper.week_activity_map(selected_user, data, k=0)
            fig, ax = plt.subplots()
            ax.bar(busy_day.index, busy_day.values, color='purple')
            plt.xticks(rotation='vertical')
            st.pyplot(fig)

        with col2:
            st.header("Most busy month")
            busy_month = helper.month_activity_map(selected_user, data, -1)
            fig, ax = plt.subplots()
            ax.bar(busy_month.index, busy_month.values, color='orange')
            plt.xticks(rotation='vertical')
            st.pyplot(fig)

        # finding the busiest users in the group(Group level)
        if selected_user == 'Overall':
            st.title('Most Busy Users 🗣️')
            x, new_df = helper.most_busy_users(data)
            fig, ax = plt.subplots()
            col1, col2 = st.columns(2)

            with col1:
                ax.bar(x.index, x.values, color='red')
                plt.xticks(rotation='vertical')
                st.pyplot(fig)
            with col2:
                st.dataframe(new_df)

        # Most Positive,Negative,Neutral User...
        if selected_user == 'Overall':
            # Getting names per sentiment
            x = data['user'][data['value'] == 1].value_counts().head(10)
            y = data['user'][data['value'] == -1].value_counts().head(10)
            z = data['user'][data['value'] == 0].value_counts().head(10)

            col1, col2, col3 = st.columns(3)
            with col1:
                # heading
                st.markdown("<h3 style='text-align: center; color: black;'>Most Positive Users</h3>",
                            unsafe_allow_html=True)

                # Displaying
                fig, ax = plt.subplots()
                ax.bar(x.index, x.values, color='green')
                plt.xticks(rotation='vertical')
                st.pyplot(fig)
            with col2:
                # heading
                st.markdown("<h3 style='text-align: center; color: black;'>Most Neutral Users</h3>",
                            unsafe_allow_html=True)

                # Displaying
                fig, ax = plt.subplots()
                ax.bar(z.index, z.values, color='grey')
                plt.xticks(rotation='vertical')
                st.pyplot(fig)
            with col3:
                # heading
                st.markdown("<h3 style='text-align: center; color: black;'>Most Negative Users</h3>",
                            unsafe_allow_html=True)

                # Displaying
                fig, ax = plt.subplots()
                ax.bar(y.index, y.values, color='red')
                plt.xticks(rotation='vertical')
                st.pyplot(fig)
        
        # Percentage contributed
        if selected_user == 'Overall':
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown("<h3 style='text-align: center; color: black;'>Most Positive Contribution</h3>",
                            unsafe_allow_html=True)
                x = helper.percentage(data, 1)

                # Displaying
                st.dataframe(x)
            with col2:
                st.markdown("<h3 style='text-align: center; color: black;'>Most Neutral Contribution</h3>",
                            unsafe_allow_html=True)
                y = helper.percentage(data, 0)

                # Displaying
                st.dataframe(y)
            with col3:
                st.markdown("<h3 style='text-align: center; color: black;'>Most Negative Contribution</h3>",
                            unsafe_allow_html=True)
                z = helper.percentage(data, -1)

                # Displaying
                st.dataframe(z)

        # activity map
        st.title("Week Activity📊")
        col1, col2 = st.columns(2)
        weekActivitySeries, week_activity = helper.weekActivity(selected_user, data)
        week_activity = week_activity.sort_values('message')
        days = week_activity['day_name']
        messages = week_activity['message']

        with col2:
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.pie(messages, labels=days, autopct='%1.1f%%', colors=plt.cm.Dark2.colors)
            ax.axis('equal')
            plt.style.use('dark_background')
            st.pyplot(fig)

        with col1:
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.bar(days, messages)
            ax.set_xlabel('Day of the Week', color="yellow")
            ax.set_ylabel('Number of Messages', color='yellow')
            plt.style.use('dark_background')
            st.pyplot(fig)

        # hourly activity
        st.title("Hour Activity⌛")
        h1, h2 = helper.hourActivity(selected_user, data)

        fig, ax = plt.subplots(figsize=(12, 3))
        h1.unstack('day_name').plot(ax=ax)
        ax.set_xlabel('Hour of the Day', color='yellow')
        ax.set_ylabel('Number of Messages', color='yellow')
        ax.set_title('Messages Sent by Hour of the Day', color='white')
        plt.style.use('dark_background')
        st.pyplot(fig)
        
        st.title("Month Activity📊")
        col1, col2 = st.columns(2)
        monthActivitySeries, month_activity = helper.monthActivity(selected_user, data)
        month_activity = month_activity.sort_values('message')
        month = month_activity['month']
        messages = month_activity['message']

        with col2:
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.pie(messages, labels=month, autopct='%1.1f%%', colors=plt.cm.Dark2.colors)
            ax.axis('equal')
            plt.style.use('dark_background')
            st.pyplot(fig)

        with col1:
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.bar(month, messages)
            ax.set_xlabel('Month of the Year', color="yellow")
            ax.set_ylabel('Number of Messages', color='yellow')
            plt.style.use('dark_background')
            st.pyplot(fig)

        # Monthly activity map
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("<h3 style='text-align: center; color: black;'>Monthly Activity map(Positive)</h3>",
                        unsafe_allow_html=True)

            busy_month = helper.month_activity_map(selected_user, data, 1)

            fig, ax = plt.subplots()
            ax.bar(busy_month.index, busy_month.values, color='green')
            plt.xticks(rotation='vertical')
            st.pyplot(fig)
        with col2:
            st.markdown("<h3 style='text-align: center; color: black;'>Monthly Activity map(Neutral)</h3>",
                        unsafe_allow_html=True)

            busy_month = helper.month_activity_map(selected_user, data, 0)

            fig, ax = plt.subplots()
            ax.bar(busy_month.index, busy_month.values, color='grey')
            plt.xticks(rotation='vertical')
            st.pyplot(fig)
        with col3:
            st.markdown("<h3 style='text-align: center; color: black;'>Monthly Activity map(Negative)</h3>",
                        unsafe_allow_html=True)

            busy_month = helper.month_activity_map(selected_user, data, -1)

            fig, ax = plt.subplots()
            ax.bar(busy_month.index, busy_month.values, color='red')
            plt.xticks(rotation='vertical')
            st.pyplot(fig)

        # Daily activity map
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("<h3 style='text-align: center; color: black;'>Daily Activity map(Positive)</h3>",
                        unsafe_allow_html=True)

            busy_day = helper.week_activity_map(selected_user, data, 1)

            fig, ax = plt.subplots()
            ax.bar(busy_day.index, busy_day.values, color='green')
            plt.xticks(rotation='vertical')
            st.pyplot(fig)
        with col2:
            st.markdown("<h3 style='text-align: center; color: black;'>Daily Activity map(Neutral)</h3>",
                        unsafe_allow_html=True)

            busy_day = helper.week_activity_map(selected_user, data, 0)

            fig, ax = plt.subplots()
            ax.bar(busy_day.index, busy_day.values, color='grey')
            plt.xticks(rotation='vertical')
            st.pyplot(fig)
        with col3:
            st.markdown("<h3 style='text-align: center; color: black;'>Daily Activity map(Negative)</h3>",
                        unsafe_allow_html=True)

            busy_day = helper.week_activity_map(selected_user, data, -1)

            fig, ax = plt.subplots()
            ax.bar(busy_day.index, busy_day.values, color='red')
            plt.xticks(rotation='vertical')
            st.pyplot(fig)

        # daily timeline
        st.title("Daily Timeline📅")
        dailyTimeline = helper.dailyTimeline(selected_user, data)
        plt.style.use('dark_background')
        plt.figure(figsize=(14, 3))
        plt.plot(dailyTimeline['onlyDate'], dailyTimeline['message'])
        plt.xticks(rotation='vertical')
        plt.title('Daily Message Count', color='yellow')
        plt.xlabel('Date', color='white')
        plt.ylabel('Message Count', color='white')
        st.pyplot(plt)

        # Daily timeline
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("<h3 style='text-align: center; color: black;'>Daily Timeline(Positive)</h3>",
                        unsafe_allow_html=True)

            daily_timeline = helper.daily_timeline(selected_user, data, 1)

            fig, ax = plt.subplots()
            ax.plot(daily_timeline['only_date'], daily_timeline['message'], color='green')
            plt.xticks(rotation='vertical')
            st.pyplot(fig)
        with col2:
            st.markdown("<h3 style='text-align: center; color: black;'>Daily Timeline(Neutral)</h3>",
                        unsafe_allow_html=True)

            daily_timeline = helper.daily_timeline(selected_user, data, 0)

            fig, ax = plt.subplots()
            ax.plot(daily_timeline['only_date'], daily_timeline['message'], color='grey')
            plt.xticks(rotation='vertical')
            st.pyplot(fig)
        with col3:
            st.markdown("<h3 style='text-align: center; color: black;'>Daily Timeline(Negative)</h3>",
                        unsafe_allow_html=True)

            daily_timeline = helper.daily_timeline(selected_user, data, -1)

            fig, ax = plt.subplots()
            ax.plot(daily_timeline['only_date'], daily_timeline['message'], color='red')
            plt.xticks(rotation='vertical')
            st.pyplot(fig)





        # monthly timeline
        st.title("Monthly Timeline⌚")
        timeline = helper.monthlyTimeline(selected_user, data)
        plt.style.use('dark_background')
        plt.figure(figsize=(12, 3))
        plt.plot(timeline['time'], timeline['message'])
        plt.xticks(rotation='vertical')
        plt.title(f"{selected_user}", color='yellow')
        st.pyplot(plt)

        # Monthly timeline
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("<h3 style='text-align: center; color: black;'>Monthly Timeline(Positive)</h3>",
                        unsafe_allow_html=True)

            timeline = helper.monthly_timeline(selected_user, data, 1)

            fig, ax = plt.subplots()
            ax.plot(timeline['time'], timeline['message'], color='green')
            plt.xticks(rotation='vertical')
            st.pyplot(fig)
        with col2:
            st.markdown("<h3 style='text-align: center; color: black;'>Monthly Timeline(Neutral)</h3>",
                        unsafe_allow_html=True)

            timeline = helper.monthly_timeline(selected_user, data, 0)

            fig, ax = plt.subplots()
            ax.plot(timeline['time'], timeline['message'], color='grey')
            plt.xticks(rotation='vertical')
            st.pyplot(fig)
        with col3:
            st.markdown("<h3 style='text-align: center; color: black;'>Monthly Timeline(Negative)</h3>",
                        unsafe_allow_html=True)

            timeline = helper.monthly_timeline(selected_user, data, -1)

            fig, ax = plt.subplots()
            ax.plot(timeline['time'], timeline['message'], color='red')
            plt.xticks(rotation='vertical')
            st.pyplot(fig)

        

        
        # WordCloud
        st.title("Wordcloud 🌍")
        df_wc = helper.create_wordcloud(selected_user, data, k=0)
        fig, ax = plt.subplots()
        ax.imshow(df_wc)
        st.pyplot(fig)

        # WORDCLOUD......
        col1, col2, col3 = st.columns(3)
        with col1:
            try:
                # heading
                st.markdown("<h3 style='text-align: center; color: black;'>Positive WordCloud</h3>",
                            unsafe_allow_html=True)

                # Creating wordcloud of positive words
                df_wc = helper.create_wordcloud(selected_user, data, 1)
                fig, ax = plt.subplots()
                ax.imshow(df_wc)
                st.pyplot(fig)
            except:
                # Display error message
                st.image('error.webp')
        with col2:
            try:
                # heading
                st.markdown("<h3 style='text-align: center; color: black;'>Neutral WordCloud</h3>",
                            unsafe_allow_html=True)

                # Creating wordcloud of neutral words
                df_wc = helper.create_wordcloud(selected_user, data, 0)
                fig, ax = plt.subplots()
                ax.imshow(df_wc)
                st.pyplot(fig)
            except:
                # Display error message
                st.image('error.webp')
        with col3:
            try:
                # heading
                st.markdown("<h3 style='text-align: center; color: black;'>Negative WordCloud</h3>",
                            unsafe_allow_html=True)

                # Creating wordcloud of negative words
                df_wc = helper.create_wordcloud(selected_user, data, -1)
                fig, ax = plt.subplots()
                ax.imshow(df_wc)
                st.pyplot(fig)
            except:
                # Display error message
                st.image('error.webp')

        # most common word
        most_common_df = helper.most_common_words(selected_user, data, k=0)

        fig, ax = plt.subplots()

        ax.barh(most_common_df[0], most_common_df[1])
        plt.xticks(rotation='vertical')

        st.title('Most common words 🥇')
        st.pyplot(fig)

        # Most common positive words
        col1, col2, col3 = st.columns(3)
        with col1:
            try:
                # Data frame of most common positive words.
                most_common_df = helper.most_common_words(selected_user, data, 1)

                # heading
                st.markdown("<h3 style='text-align: center; color: black;'>Positive Words</h3>", unsafe_allow_html=True)
                fig, ax = plt.subplots()
                ax.barh(most_common_df[0], most_common_df[1], color='green')
                plt.xticks(rotation='vertical')
                st.pyplot(fig)
            except:
                # Disply error image
                st.image('error.webp')
        with col2:
            try:
                # Data frame of most common neutral words.
                most_common_df = helper.most_common_words(selected_user, data, 0)

                # heading
                st.markdown("<h3 style='text-align: center; color: black;'>Neutral Words</h3>", unsafe_allow_html=True)
                fig, ax = plt.subplots()
                ax.barh(most_common_df[0], most_common_df[1], color='grey')
                plt.xticks(rotation='vertical')
                st.pyplot(fig)
            except:
                # Disply error image
                st.image('error.webp')
        with col3:
            try:
                # Data frame of most common negative words.
                most_common_df = helper.most_common_words(selected_user, data, -1)

                # heading
                st.markdown("<h3 style='text-align: center; color: black;'>Negative Words</h3>", unsafe_allow_html=True)
                fig, ax = plt.subplots()
                ax.barh(most_common_df[0], most_common_df[1], color='red')
                plt.xticks(rotation='vertical')
                st.pyplot(fig)
            except:
                # Disply error image
                st.image('error.webp')

        # emoji analysis
        emoji_df = helper.mostEmoji(selected_user, data)
        if (emoji_df.shape[0] != 0):
            st.title("Emoji Analysis😳")

        col1, col2 = st.columns(2)

        with col1:
            st.dataframe(emoji_df)
        with col2:
            fig, ax = plt.subplots()
            color = ['#FFC107', '#2196F3', '#4CAF50', '#F44336', '#9C27B0']

            ax.pie(emoji_df['Count'].head(), labels=emoji_df['Emoji'].head(), autopct="%0.2f", colors=color)
            ax.set_title("Emoji Distribution", color='yellow')
            fig.set_facecolor('#121212')
            st.pyplot(fig)

        
        st.title("Weekly Activity Map 🗺️")
        user_heatmap = helper.activity_heatmap(selected_user, data, k=0)
        fig, ax = plt.subplots()
        ax = sns.heatmap(user_heatmap,cmap='viridis', linewidths=0.001, linecolor='white', cbar_kws={'label': 'Activity Level'})
        plt.xlabel('Hour of the Day', fontsize=10)
        plt.ylabel('Day of the Week', fontsize=10)
        st.pyplot(fig)


        # User Sentiment Analysis
        st.title("Sentiment Analysis")

        # Perform sentiment analysis for the selected user or overall
        user_sentiments = helper.analyze_sentiment(selected_user, data)

        # If the selected user is 'Overall', display sentiment for all users
        if selected_user == 'Overall':
          overall_sentiment = {key: sum([user[key] for user in user_sentiments.values()])/len(user_sentiments) for key in ['pos', 'neu', 'neg', 'compound']}
          st.subheader("Overall")

          # Create a horizontal bar chart for the overall sentiment scores
          fig, ax = plt.subplots()
          ax.barh(list(overall_sentiment.keys()), list(overall_sentiment.values()), color=['green' if v >= 0 else 'red' for v in overall_sentiment.values()])
          ax.set_xlabel('Score')
          ax.set_title('Overall sentiment scores')
          st.pyplot(fig)
        else:
          # If the selected user exists in the sentiment analysis results
          if selected_user in user_sentiments:
            sentiment_scores = user_sentiments[selected_user]

            st.subheader(f"User: {selected_user}")

            # Create a horizontal bar chart for the sentiment scores
            fig, ax = plt.subplots()
            ax.barh(list(sentiment_scores.keys()), list(sentiment_scores.values()), color=['green' if v >= 0 else 'red' for v in sentiment_scores.values()])
            ax.set_xlabel('Score')
            ax.set_title(f'Sentiment scores for {selected_user}')
            st.pyplot(fig)
          else:
               st.write(f"No sentiment analysis results for {selected_user}")

        ## Final results for the sentiment analysis

        # Perform sentiment analysis for the selected user or overall
        user_sentiments = helper.analyze_sentiment(selected_user, data)

       # Display the sentiment score
        sentiment = helper.sentiment_score(user_sentiments, selected_user)
        st.subheader(f"The sentiment of {selected_user} is {sentiment}")
        

        

        
         


    
     



    
        


    
    
