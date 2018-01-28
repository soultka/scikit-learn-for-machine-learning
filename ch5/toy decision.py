import pandas as pd
from sklearn.tree import DecisionTreeClassifier, export_graphviz

instances = [
    {'plays fetch': False, 'species': 'Dog'},
    {'plays fetch': False, 'species': 'Cat'},
    {'plays fetch': False, 'species': 'Cat'},
    {'plays fetch': True, 'species': 'Dog'}
]
df = pd.DataFrame(instances)
X_train = [[1] if a else [0] for a in df['plays fetch']]
y_train = [1 if d == 'Dog' else 0 for d in df['species']]
labels = ['plays fetch']
clf = DecisionTreeClassifier(max_depth=None, max_features=None, criterion='entropy',
                             min_samples_leaf=1, min_samples_split=2)
clf.fit(X_train, y_train)
f = 'level3-plays-fetch.dot'
export_graphviz(clf, out_file=f, feature_names=labels)
