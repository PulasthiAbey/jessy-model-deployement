#installing the dependencies
import pandas as pd
import preprocess_kgptalkie as ps
import numpy as np

#model
from sklearn.linear_model import LinearRegression 
from sklearn.model_selection import train_test_split

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


def scheduler():
    data = pd.read_csv('https://raw.githubusercontent.com/laxmimerit/Amazon-Musical-Reviews-Rating-Dataset/master/Musical_instruments_reviews.csv' )

    #training the model with these instructions
    train = data.drop(['description', 'additional-information'], axis=1)
    test = data['distance']

    #testing
    X_train, X_test, y_train, y_test = train_test_split(train, test, test_size=0.3, random_state=2)

    #reggression model of the dataset
    regr = LinearRegression()

    #reggression - Checking the fitted values 
    regr = regr.fit(X_train, y_train)

    #prediction values of the dataset
    pred = regr.predict(X_test)

    #output
    return pred

