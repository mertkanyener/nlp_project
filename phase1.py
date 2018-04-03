import pandas as pd
import random

#from nltk.stem import *
#from nltk.tokenize import RegexpTokenizer, sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk import *


df = pd.read_csv('amazon_d.csv', header=None)


df.columns = [col for col in df.iloc[0, :]]
reviews = df.iloc[1:, 19]  # only review.texts

reviews_ratings = df.loc[:, ('reviews.text', 'reviews.rating')]  # get reviews and ratings together
df_rr_cleared = reviews_ratings.dropna(axis=0)  # delete rows with the missing values
df_rr = df_rr_cleared.iloc[1:, :]  # do not include first row which is: review.text review.rating
df_rr = df_rr.reset_index(drop=True)
length = df_rr.count()[1]  # row count of dataframe


# Find the stem of words in text

df_rev = df_rr.iloc[:, 0]  # only reviews
rand_review = df_rev.loc[random.randint(0, length)]

tokenizer = RegexpTokenizer(r'\w+')
ps = PorterStemmer()
rand_review = tokenizer.tokenize(rand_review)
stems = [ps.stem(word) for word in rand_review]

# Eliminate the stopwords in stem list using English stopword list

eliminated = [stem for stem in stems if stem not in stopwords.words('english')]

# Find the most frequent stems in list

most_freq = FreqDist(eliminated)

print(rand_review)
print(stems)
print(eliminated)
print(most_freq.most_common(10))


#tokenizer = RegexpTokenizer(r'\w+')




