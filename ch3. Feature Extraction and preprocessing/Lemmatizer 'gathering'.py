# needed download from nltk corpora
from nltk.stem.wordnet import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
print (lemmatizer.lemmatize('gathering', 'v'))
print (lemmatizer.lemmatize('gathering', 'n'))
