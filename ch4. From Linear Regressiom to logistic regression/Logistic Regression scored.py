import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model.logistic import LogisticRegression
from sklearn.model_selection import train_test_split as tts ,cross_val_score as cvs
df = pd.read_csv('sms.csv')

X_train_raw , X_test_raw , y_train, y_test = tts(df['message'],df['label'])
vectorizer = TfidfVectorizer()
X_train = vectorizer.fit_transform(X_train_raw)
X_test = vectorizer.transform(X_test_raw)
classifier = LogisticRegression()
classifier.fit(X_train , y_train)
scores = cvs(classifier, X_train,y_train , cv=5 )
print(np.mean(scores) , scores)


