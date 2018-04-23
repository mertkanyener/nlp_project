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
    raw_review = df_rev.loc[random.randint(0, length)]

    print("Review text: ", raw_review)
    print("---------------------------")

    tokenizer = RegexpTokenizer(r'\w+')  # remove punctuation with regex
    ps = PorterStemmer()
    rand_review = tokenizer.tokenize(raw_review)
    stems = [ps.stem(word) for word in rand_review]
    print("Stems list: ", stems)
    print("---------------------------")

    # Eliminate the stopwords in stem list using English stopword list

    eliminated = [stem for stem in stems if stem not in stopwords.words('english')]
    print("Stopwords eliminated: ", eliminated)
    print("---------------------------")

    return eliminated, rand_review


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
    bglist = [key for key, value in bg_freq.items() if value >= frequency]
    res = []
    if 0 < n <= len(bglist):
        res = bglist[:n]
    elif n > len(bglist):
        res = bglist
    else:
        print("Invalid n value!")
    return res


def scored_bigram(token_list, bglist):

    word_fd = FreqDist(tl)
    bigram_fd = FreqDist(bglist)
    bigram_measures = collocations.BigramAssocMeasures()
    finder = BigramCollocationFinder(word_fd, bigram_fd)
    scored = finder.score_ngrams(bigram_measures.raw_freq)
    return scored


def sorted_bigram(scored):
    return sorted(bg for bg, score in scored)


def pos_tagger(rev_text):
    return pos_tag(rev_text)


def find_words(tagged_list, POS):
    words = []
    if POS == 'noun':
        words = [i[0] for i in tagged_list if i[1] == 'NNP' or i[1] == 'NN' or i[1] == 'NNS']
    elif POS == 'verb':
        words = [i[0] for i in tagged_list if i[1] == 'VB' or i[1] == 'VBN' or i[1] == 'VBD'
                 or i[1] == 'VBG' or i[1] == 'VBP' or i[1] == 'VBZ']
    elif POS == 'adjective':
        words = [i[0] for i in tagged_list if i[1] == 'JJ' or i[1] == 'JJR' or i[1] == 'JJS']
    elif POS == 'adverb':
        words = [i[0] for i in tagged_list if i[1] == 'RB' or i[1] == 'RBR' or i[1] == 'RBS'
                 or i[1] == 'WRB']
    elif POS == 'pronoun':
        words = [i[0] for i in tagged_list if i[1] == 'PRP' or i[1] == 'PRP$' or i[1] == 'WP'
                 or i[1] == 'WPS']
    elif POS == 'other':
        words = [i[0] for i in tagged_list if i[1] == 'CC' or i[1] == 'CD' or i[1] == 'DT'
                 or i[1] == 'EX' or i[1] == 'IN' or i[1] == 'LS' or i[1] == 'MD'
                 or i[1] == 'PDT' or i[1] == 'POS' or i[1] == 'RP' or i[1] == 'TO'
                 or i[1] == 'UH' or i[1] == 'WDT']
    else:
        print('Wrong tag name!')

    return most_frequent(words, 10)

tl, raw_text = preprocessing('amazon_d.csv')
most_freq = most_frequent(tl, 10)
bglist = list_freq_bigram(tl, 2, 5)

print("Bigrams : ", bglist)
print("---------------------------")
scored = scored_bigram(tl, bglist)
print("Scored bigrams: ", scored)
print("---------------------------")
sorted_bgs = sorted_bigram(scored)
print("Sorted scored bigrams: ", sorted_bgs)
POS = pos_tagger(raw_text)
print("---------------------------")
print("POS tags: ", POS)
POS_tags = [i[1] for i in POS]
print("---------------------------")
print("Only tags: ", POS_tags)
most_freq_POS = most_frequent(POS_tags, 10)
print("---------------------------")
print("10 most common POS tags: ", most_freq_POS)
tag_name = 'noun'
most_freq_POS_words = find_words(POS, tag_name)
print("---------------------------")
print("10 most common ", tag_name, "s: ", most_freq_POS_words)





