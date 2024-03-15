import pandas as pd
import numpy as np
import requests
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split


# Part 3: Text mining.

# Return a pandas dataframe containing the data set.
# Specify a 'latin-1' encoding when reading the data.
# data_file will be populated with a string 
# corresponding to a path containing the wholesale_customers.csv file.
def read_csv_3(data_file):
	df = pd.read_csv(data_file, encoding='latin-1')
	return df

df = read_csv_3('coronavirus_tweets.csv')


# Return a list with the possible sentiments that a tweet might have.
def get_sentiments(df):
	return df['Sentiment'].unique().tolist()

# Return a string containing the second most popular sentiment among the tweets.
def second_most_popular_sentiment(df):
	sentiment_count = df['Sentiment'].value_counts()
	# Check if there are at least two unique sentiments
	if len(sentiment_count) < 2:
		raise ValueError("There must be at least two unique sentiments in the DataFrame.")
	# Return the second most popular sentiment (assuming index starts from 0)
	return sentiment_count.index[1]



# Return the date (string as it appears in the data) with the greatest number of extremely positive tweets.
def date_most_popular_tweets(df):
	df_extremely_pos = df[df['Sentiment'] == 'Extremely Positive']
	date_with_most_tweets = df_extremely_pos['TweetAt'].value_counts().idxmax()
	return date_with_most_tweets

# Modify the dataframe df by converting all tweets to lower case. 
def lower_case(df):
	df['OriginalTweet'] = df['OriginalTweet'].str.lower()
	print(df['OriginalTweet'])

# Modify the dataframe df by replacing each characters which is not alphabetic or whitespace with a whitespace.
def remove_non_alphabetic_chars(df):
	# make sure it is replacing as intended to
	df['OriginalTweet'] = df['OriginalTweet'].str.replace(r'[^a-zA-Z@\s]+', ' ')


# Modify the dataframe df with tweets after removing characters which are not alphabetic or whitespaces.
def remove_multiple_consecutive_whitespaces(df):
	df['OriginalTweet'] = df['OriginalTweet'].str.replace(r'\s+', ' ')


# Given a dataframe where each tweet is one string with words separated by single whitespaces,
# tokenize every tweet by converting it into a list of words (strings).
def tokenize(df):
	df['OriginalTweet'] = df['OriginalTweet'].str.split()

# Given dataframe tdf with the tweets tokenized, return the number of words in all tweets including repetitions.
def count_words_with_repetitions(tdf):
	return sum(tdf['OriginalTweet'].apply(len))

# Given dataframe tdf with the tweets tokenized, return the number of distinct words in all tweets.
def count_words_without_repetitions(tdf):
	distinct_words = set()
	for tweet in tdf['OriginalTweet']:
		distinct_words.update(tweet)
	return len(distinct_words)

# Given dataframe tdf with the tweets tokenized, return a list with the k distinct words that are most frequent in the tweets.
def frequent_words(tdf,k):
	all_words = [word for tweet in tdf['OriginalTweet'] for word in tweet]
	word_counts = pd.Series(all_words).value_counts()
	return word_counts.index[:k].tolist()

# Given dataframe tdf with the tweets tokenized, remove stop words and words with <=2 characters from each tweet.
# The function should download the list of stop words via:
# https://raw.githubusercontent.com/fozziethebeat/S-Space/master/data/english-stop-words-large.txt
def remove_stop_words(tdf):
	stop_words_url = 'https://raw.githubusercontent.com/fozziethebeat/S-Space/master/data/english-stop-words-large.txt'
	stop_words = set(requests.get(stop_words_url).text.split())
	tdf['OriginalTweet'] = tdf['OriginalTweet'].apply(lambda tweet: [word for word in tweet if word.lower() not in stop_words and len(word) > 2])


# Given dataframe tdf with the tweets tokenized, reduce each word in every tweet to its stem.
def stemming(tdf):
	porter = PorterStemmer()
	tdf['OriginalTweet'] = tdf['OriginalTweet'].apply(lambda tweet: [porter.stem(word) for word in tweet])

# Given a pandas dataframe df with the original coronavirus_tweets.csv data set,
# build a Multinomial Naive Bayes classifier. 
# Return predicted sentiments (e.g. 'Neutral', 'Positive') for the training set
# as a 1d array (numpy.ndarray). 
def mnb_predict(df):
	count_vectorizer = CountVectorizer()
	X = count_vectorizer.fit_transform(df['OriginalTweet'].apply(' '.join))
	y = df['Sentiment']
	clf = MultinomialNB()
	clf.fit(X, y)
	return clf.predict(X)

# Given a 1d array (numpy.ndarray) y_pred with predicted labels (e.g. 'Neutral', 'Positive') 
# by a classifier and another 1d array y_true with the true labels, 
# return the classification accuracy rounded in the 3rd decimal digit.
def mnb_accuracy(y_pred,y_true):
	return round((y_pred == y_true).mean(), 3)



# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df['OriginalTweet'], df['Sentiment'], test_size=0.2, random_state=42)

# Convert tokens to strings for CountVectorizer
df['OriginalTweet'] = df['OriginalTweet'].apply(lambda tweet: ' '.join(tweet))

# Build the Multinomial Naive Bayes classifier
clf = MultinomialNB()

# Initialize CountVectorizer
count_vectorizer = CountVectorizer()

# Transform the training and testing sets
X_train_counts = count_vectorizer.fit_transform(X_train)
X_test_counts = count_vectorizer.transform(X_test)

# Train the classifier
clf.fit(X_train_counts, y_train)

# Predict on the training set
train_pred = clf.predict(X_train_counts)

# Predict on the testing set
test_pred = clf.predict(X_test_counts)

# Calculate training and testing accuracy
train_accuracy = mnb_accuracy(train_pred, y_train)
test_accuracy = mnb_accuracy(test_pred, y_test)

print("Training accuracy:", train_accuracy)
print("Testing accuracy:", test_accuracy)




