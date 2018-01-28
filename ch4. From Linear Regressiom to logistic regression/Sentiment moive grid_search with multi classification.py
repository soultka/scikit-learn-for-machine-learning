import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model.logistic import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.metrics import classification_report,accuracy_score,confusion_matrix

def main():
    pipeline = Pipeline([
        ('vect' , TfidfVectorizer(stop_words='english')),
        ('clf' , LogisticRegression())
    ])
    parameters = {
        'vect__max_df' : (0.25,0.5),
        'vect__ngram_range' : ((1,1), (1,2)),
        'vect__use_idf' : (True, False),
        'clf__C' : (0.1, 1, 10),
    }
    df = pd.read_csv('train.tsv' , delimiter = '\t', header=0)
    X,y = df['Phrase'] , df['Sentiment'].as_matrix()
    X_train , X_test, y_train, y_test = train_test_split(X,y,test_size=0.5)
    grid_search = GridSearchCV(pipeline,parameters,n_jobs=3, verbose=1, scoring='accuracy')
    grid_search.fit(X_train, y_train)
    print('Best score : %0.3f' % grid_search.best_score_)
    print('Best parameters set: ' )
    best_parameters = grid_search.best_estimator_.get_params()
    for parameter in sorted(best_parameters.keys()):
        print('\t%s: %r' % (parameter, best_parameters[parameter]))

    predictions = grid_search.predict(X_test)
    print('Accuracy :', accuracy_score(y_test, predictions))
    print('Confusion Matrix:' , confusion_matrix(y_test, predictions))
    print('Classification Report:' , classification_report(y_test,predictions))
    
    
if __name__ == '__main__':
    main()

    
