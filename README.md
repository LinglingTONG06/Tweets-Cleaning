# Tweets-Cleaning
# 1.1 Import library

import re
import string
from typing import Any, Union

import pandas as pd
import numpy as np
import nltk
import ssl
import matplotlib.pyplot as plt
from HanTa import HanoverTagger as ht
from pandas import DataFrame, Series
from pandas.io.parsers import TextFileReader

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context
nltk.download('stopwords')


# 1.2 Load Tweets csv file
tweets = pd.read_csv("/Users/tonglingling/Documents/GitHub/ms-st-20-12-filterblase/data/twitter_politics.csv")
print(tweets.head())
tweets.info()


# 2. Pre-processing text data in the column of Inhalt
# 2.1. Define functions for cleaning
def clean_text(df, text_field):
    df[text_field] = df[text_field].fillna('').apply(str)
    df[text_field] = df[text_field].apply(lambda elem: re.sub(r"(@[A-Za-z0-9iäöüÄÖÜ]+)|([^0-9A-Za-zäöüÄÖÜ \s])", "",elem))  # remove tweeted at and non letters and numbers
    hyperlink_pattern = r"(?:\@|http?\://|https?\://|www)\S*'"
    df[text_field] = df[text_field].apply(lambda elem: re.sub(hyperlink_pattern, "", elem)) # remove links
    df[text_field] = df[text_field].apply(lambda elem: re.sub(r"[^a-zA-ZäöüÄÖÜ]", " ", elem))  # remove numbers
    return df


# 2.2. Use cleaning functions to the column Inhalt

tweets["Cleaned_Inhalt"] = tweets["Inhalt"].fillna('').apply(str)

# remove tweeted at and non letters and numbers
tweets["Cleaned_Inhalt"] = tweets["Cleaned_Inhalt"].apply(lambda elem: re.sub(r"(@[A-Za-z0-9iäöüÄÖÜ]+)|([^0-9A-Za-zäöüÄÖÜ \s])", "",elem))

# remove links
hyperlink_pattern = r"(?:\@|http?\://|https?\://|www)\S*"
tweets["Cleaned_Inhalt"] = tweets["Cleaned_Inhalt"].apply(lambda elem: re.sub(hyperlink_pattern, "", elem,flags=re.MULTILINE))

# remove numbers
tweets["Cleaned_Inhalt"] = tweets["Cleaned_Inhalt"].apply(lambda elem: re.sub(r"[^a-zA-ZäöüÄÖÜ]", " ", elem))

print(tweets["Cleaned_Inhalt"].head(10))

# 3. Tokenization and tokens cleaning

# 3.1 Remove german and english stopwords

# Transfer all words to lowercase
tweets["Cleaned_Inhalt"] = tweets["Cleaned_Inhalt"].str.lower()

# remove stopwords
stop_words_list = set(nltk.corpus.stopwords.words('english') + nltk.corpus.stopwords.words('german'))
tweets["Cleaned_Inhalt_tokens"] = tweets["Cleaned_Inhalt"].apply(lambda x:[item for item in str(x).split() if item not in stop_words_list])


# 3.2  Tokenization and Lemmatizing
# Based on the information from the website http://textmining.wp.hs-hannover.de/Preprocessing.html
# and https://github.com/wartaal/HanTa/blob/master/Demo.ipynb
#nlp = spacy.load('de')
#lemma = nlp.WordNetLemmatizer()
#d = [lemma.lemmatize(word) for word in d]  # identify the correct form of the word in the dictionary
#d = " ".join(d)

# Tokenize
#tweets["Cleaned_Inhalt_tokens"] = tweets["Cleaned_Inhalt"].apply(lambda x: nltk.tokenize.word_tokenize(x, language='german'))

# tagger = ht.HanoverTagger('morphmodel_ger.pgz')
# tweets["Cleaned_Inhalt_tokens"] =tweets["Cleaned_Inhalt_tokens"].apply(lambda x: tagger.tag_sent(x,taglevel= 1))
# print(tweets["Cleaned_Inhalt_tokens"].head(10))

# 4. Descriptive statistics
# Convert the data type of the column Datum from object to datetime
tweets["Datum"] = pd.to_datetime(tweets["Datum"], format='%Y-%m-%d')

# Add features of tweets
tweets['count_punct'] = tweets["Inhalt"].str.count(".!?")
tweets['count_exclamation_mark'] = tweets["Inhalt"].str.count("!")
# tweets['count_question_mark'] = tweets["Inhalt"].str.count("?")

# Split Hashtags
tweets["Splited_Hashtags"] = tweets["Hashtags"].str.split(' ')
print(tweets["Splited_Hashtags"].head(5))
tweets["Splited_Hashtags"] = tweets["Splited_Hashtags"].replace(np.nan, '', regex=True)
hashtags_list = np.hstack(tweets["Splited_Hashtags"]).tolist()

# Plot top ten Hashtags
n = np.random.rand(len(hashtags_list))
a = np.random.choice(hashtags_list, p=n/n.sum(),size=400)
s = pd.Series(a)
s.value_counts()[:10].plot(kind="bar")
plt.show()
plt.savefig("/Users/Tweets_clean/output/Top_10_Hashtags.png")

# Plot the top ten most tweeted accounts
tweets['Nutzername'].value_counts()[:10].plot(kind="bar")
plt.show()
plt.savefig("/Users/Tweets_clean/output/Top_10_most_tweeted_accounts.png")

# Save as csv file
tweets.to_csv("/Users/Tweets_clean/output/twitter_politics_cleaned.csv")
