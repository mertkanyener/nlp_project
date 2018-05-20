import pandas as pd
import random


from nltk.corpus import stopwords
from nltk import *
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score


df = pd.read_csv('amazon_d.csv', header=None)

df.columns = [col for col in df.iloc[0, :]]

reviews_ratings = df.loc[:, ('reviews.text', 'reviews.rating')]  # get reviews and ratings together
df_rr_cleared = reviews_ratings.dropna(axis=0)  # delete rows with the missing values
df_rr = df_rr_cleared.iloc[1:, :]  # do not include first row which is: review.text review.rating
df_rr = df_rr.reset_index(drop=True)
X = df_rr.loc[:, 'reviews.text']
y = df_rr.loc[:, 'reviews.rating']
length = df_rr.count()[1]  # row count of dataframe

# Find the stem of words in text

#tfidf = TfidfVectorizer(sublinear_tf=True, min_df=5, norm='l2',
#                        encoding='latin-1', ngram_range=(1, 2), stop_words='english')
#features = tfidf.fit_transform(X).toarray()
count_vect = CountVectorizer()
tfidf_transformer = TfidfTransformer()
X_counts = count_vect.fit_transform(X)
X_tfidf = tfidf_transformer.fit_transform(X_counts)

X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y, test_size=0.2, random_state=6, stratify=y)

NB = MultinomialNB().fit(X_train, y_train)
y_pred = NB.predict(X_test)
print('Accuracy: %.3f' % accuracy_score(y_test, y_pred))