from sklearn.datasets import fetch_20newsgroups
from sklearn.metrics import f1_score ,classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Perceptron

categories = ['rec.sport.hockey' , 'rec.sport.baseball', 'rec.autos']
news_group_train = fetch_20newsgroups(subset='train' ,categories = categories ,
                                      remove = ('header','footer','quotes'))
news_group_test = fetch_20newsgroups(subset='test' ,categories = categories ,
                                      remove = ('header','footer','quotes'))
vectorizer = TfidfVectorizer()
X_train = vectorizer.fit_transform(news_group_train)
X_test = vectorizer.transform(news_group_test)

classifier = Perceptron(max_iter=100,eta0=0.1)

