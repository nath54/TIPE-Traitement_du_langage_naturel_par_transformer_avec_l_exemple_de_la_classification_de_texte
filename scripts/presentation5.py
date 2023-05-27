import gensim.downloader as api
from gensim.models import Word2Vec

model = api.load('word2vec-google-news-300')

q = input(">>> ")
while q != "q":
    similar_words = model.most_similar(q, topn=50)
    print([sw[0] for sw in similar_words])
    q = input(">>> ")
