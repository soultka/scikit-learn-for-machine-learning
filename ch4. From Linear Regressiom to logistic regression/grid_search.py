import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model.logistic import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.metrics import precision_score, recall_score, accuracy_score

pipeline = Pipeline([
    ('vect' , TfidfVectorizer(stop_words='english'))
    ,('clf', LogisticRegression())
    ])
parameters = {
    'vect__max_df': (0.25, 0.5, 0.75),
    'vect__stop_words' : ('english', None),
    'vect__max_features' : (2500,5000,10000,None),
    'vect__ngram_range' : ((1,1) , (1,2)),
    'vect__use_idf' : (True,False),
    'vect__norm' : ('l1', 'l2'),
    'clf__penalty' : ('l1','l2'),
    'clf__C': (0.01,0.1,1,10)
    }

if __name__ == "__main__":
    grid_search = GridSearchCV(pipeline,parameters, n_jobs=-1, verbose=1,
                               scoring = 'accuracy' , cv=3)
    df = pd.read_csv('sms.csv')
    X, y = df['message'] , df['label']
    X_train , X_test , y_train ,y_test = train_test_split(X,y)
    grid_search.fit(X_train, y_train)
    print('Best score: %0.3f' % grid_search.best_score_)
    print('Best parameters set:')
    best_parameters = grid_search.best_estimator_.get_params()
    for param_name in sorted(parameters.keys()):
        print('\t%s: %r' % (param_name , best_parameters[param_name]))
    predictions = grid_search.predict(X_test)
    print('Accuracy:',accuracy_score(y_test,predictions))
    print('Precision:',precision_score(y_test,predictions))
    print('Recall:',recall_score(y_test,predictions))

    
    
