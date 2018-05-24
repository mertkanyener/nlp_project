import pandas as pd
import numpy as np
import gzip


from nltk.corpus import stopwords
from nltk import *
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA, TruncatedSVD, KernelPCA


def parse(path):
  g = gzip.open(path, 'rb')
  for l in g:
    yield eval(l)


def getDF(path):
  i = 0
  df = {}
  for d in parse(path):
    df[i] = d
    i += 1
  return pd.DataFrame.from_dict(df, orient='index')


def read_data(filename):
    df = getDF(filename)
    X = df.loc[:10000, 'reviewText'].values
    y = df.loc[:10000, 'overall'].values

    return X, y


def text_representation(X):

    count_vect = CountVectorizer()
    tfidf_transformer = TfidfTransformer()
    X_counts = count_vect.fit_transform(X)
    X_tfidf = tfidf_transformer.fit_transform(X_counts)

    return X_tfidf


def run_classifiers(X, y, random_state, pipelines):

    pipe_nb, pipe_lr, pipe_svm, pipe_forest, pipe_knn = pipelines
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state, stratify=y)

    pipe_nb.fit(X_train, y_train)
    y_pred = pipe_nb.predict(X_test)
    print('Naive Bayes Accuracy: %.3f' % accuracy_score(y_test, y_pred))

    pipe_forest.fit(X_train, y_train)
    y_pred = pipe_forest.predict(X_test)
    print('Random Forest Accuracy: %.3f' % accuracy_score(y_test, y_pred))

    pipe_lr.fit(X_train, y_train)
    y_pred = pipe_lr.predict(X_test)
    print('Logistic Regression Accuracy: %.3f' % accuracy_score(y_test, y_pred))

    pipe_svm.fit(X_train, y_train)
    y_pred = pipe_svm.predict(X_test)
    print('SVM Accuracy: %.3f' % accuracy_score(y_test, y_pred))

    pipe_knn.fit(X_train, y_train)
    y_pred = pipe_knn.predict(X_test)
    print('KNN Accuracy: %.3f' % accuracy_score(y_test, y_pred))


def make_pipelines():

    pipe_nb = make_pipeline(
                            MultinomialNB())

    pipe_lr = make_pipeline(TruncatedSVD(n_components=2),
                            LogisticRegression(random_state=1))

    pipe_svm = make_pipeline(
                            SVC(random_state=1))

    pipe_forest = make_pipeline(
                                RandomForestClassifier(n_estimators=100, random_state=1))

    pipe_knn = make_pipeline(
                             #TruncatedSVD(n_components=2),
                             KNeighborsClassifier(n_neighbors=5))

    pipelines = pipe_nb, pipe_lr, pipe_svm, pipe_forest, pipe_knn

    return pipelines


auto = 'reviews_Automotive.json.gz'
phones = 'reviews_Cell_Phones_and_Accessories.json.gz'
games = 'reviews_Video_Games.json.gz'

X, y = read_data(auto)
X_bow = text_representation(X)

pipe_nb = make_pipeline(MultinomialNB())

pipe_lr = make_pipeline(
                        TruncatedSVD(n_components=5000),
                        LogisticRegression(random_state=1))
X_train, X_test, y_train, y_test = train_test_split(X_bow, y, test_size=0.2, random_state=1, stratify=y)
print(X_train.shape)
pipe_nb.fit(X_train, y_train)
y_pred = pipe_nb.predict(X_test)
print('Accuracy: %.3f' % accuracy_score(y_test, y_pred))


"""
pipelines = make_pipelines()
run_classifiers(X_bow, y, 4, pipelines)
"""
