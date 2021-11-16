import pandas as pd
import numpy as np
import preprocess_kgptalkie as ps
import re

#libraries for the algorithm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC

#cleaning function
def get_clean(x):
    x = str(x).lower().replace('\\', '').replace('_', ' ')
    x = ps.cont_exp(x)
    x = ps.remove_emails(x)
    x = ps.remove_urls(x)
    x = ps.remove_html_tags(x)
    x = ps.remove_rt(x)
    x = ps.remove_accented_chars(x)
    x = ps.remove_special_chars(x)
    x = re.sub("(.)\\1{2,}", "\\1", x)
    return x


def getReviewScore(reviewText):
    reviewScore = 0
    cleaned_reviewText = get_clean(reviewText)
    df = pd.read_csv('https://raw.githubusercontent.com/laxmimerit/Amazon-Musical-Reviews-Rating-Dataset/master/Musical_instruments_reviews.csv' , usecols=['reviewText', 'overall'])

    #apply cleaning function
    df['reviewText'] = df['reviewText'].apply(lambda x: get_clean(x))
    tfidf = TfidfVectorizer(max_features=20000, ngram_range=(1,5), analyzer='char')
    X = tfidf.fit_transform(df['reviewText'])
    Y = df['overall']
    X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.8, random_state = 0)

    clf = LinearSVC( C = 20, class_weight="balanced")
    clf.fit(X_train, Y_train)

    y_pred = clf.predict(X_test)

    vec = tfidf.transform([cleaned_reviewText])
    reviewScore = vec[0][0]
    

    return reviewScore