from sklearn.feature_extraction.text import TfidfVectorizer

corpus = [
    'The dog ate a sandwich and I ate a sandwich',
    'The wizard transfigured a sandwich'
    ]

vectorizer = TfidfVectorizer(stop_words='english')
print (vectorizer.fit_transform(corpus).todense())
print (vectorizer.vocabulary_)
