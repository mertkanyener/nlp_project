import pandas as pd
import random

from nltk.corpus import stopwords
from nltk import *



def preprocessing(filename):

    # Read and preprocess data
    df = pd.read_csv(filename, header=None)

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

    print("Review text: ", rand_review)
    print("---------------------------")

    tokenizer = RegexpTokenizer(r'\w+')  # remove punctuation with regex
    ps = PorterStemmer()
    rand_review = tokenizer.tokenize(rand_review)
    stems = [ps.stem(word) for word in rand_review]
    print("Stems list: ", stems)
    print("---------------------------")

    # Eliminate the stopwords in stem list using English stopword list

    eliminated = [stem for stem in stems if stem not in stopwords.words('english')]
    print("Stopwords eliminated: ", eliminated)
    print("---------------------------")

    return eliminated


def most_frequent(token_list, n):

    most_freq = FreqDist(token_list)
    result = most_freq.most_common(n)
    return result


def listNgrams(token_list, n):

    nglist = ngrams(token_list, n)
    for ng in nglist:
        print(ng)


def list_freq_bigram(token_list, frequency, n):

    bglist = bigrams(token_list)
    bg_freq = FreqDist(bglist)
    bglist = [key for key, value in bg_freq.items() if value == frequency]
    res = []
    if 0 < n <= len(bglist):
        res = bglist[:n]
    elif n > len(bglist):
        res = bglist
    else:
        print("Invalid n value!")
    return res


def scored_bigram(bglist):

    bigram_measures = collocations.BigramAssocMeasures()
    finder = BigramCollocationFinder.from_words(bglist)
    scored = finder.score_ngrams(bigram_measures.raw_freq)
    return scored


def sorted_bigram(scored):
    return sorted(bg for bg, score in scored)


tl = preprocessing('amazon_d.csv')
most_freq = most_frequent(tl, 10)
bglist = list_freq_bigram(tl, 2, 5)
print("Bigrams : ", bglist)
print("---------------------------")
scored = scored_bigram(bglist)
print("Scored bigrams: ", scored)
print("---------------------------")
sorted_bgs = sorted_bigram(scored)
print("Sorted scored bigrams: ", sorted_bgs)