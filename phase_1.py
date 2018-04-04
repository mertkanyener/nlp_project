
import pandas as pd
import random

from nltk.corpus import stopwords
from nltk import *
from nltk.tokenize import RegexpTokenizer




tokenizer = RegexpTokenizer(r'\w+')

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

#review1 = df_rev.loc[5]
ps = PorterStemmer()



print(rand_review)
print("--------------------")

rt = tokenizer.tokenize(rand_review)
stems = [ps.stem(word) for word in rt]
eliminated = [stem for stem in stems if stem not in stopwords.words('english')]
most_freq = FreqDist(eliminated)

print(stems)
print(eliminated)
print(most_freq.most_common(10))

nglist = ngrams(eliminated, 3)
for ng in nglist:
    print(ng)







