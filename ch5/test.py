import itertools
from lib2to3.pgen2 import token
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
from nltk.tokenize import word_tokenize
from collections import Counter
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from gensim.corpora.dictionary import Dictionary
from collections import defaultdict

articles = []
#Read TXT file
for i in range(10):
    f = open(f"D:/6206021622119/wiki/wiki_article_{i}.txt" , "r" ,encoding="utf8")
    article = f.read()

    tokens = word_tokenize(article)

    lower_tokens = [t.lower() for t in tokens]

    alpha_only = [t for t in lower_tokens if t.isalpha()]

    no_stops = [t for t in alpha_only if t not in stopwords.words('english')]

    wordnet_lemmatizer = WordNetLemmatizer()

    lemmatized = [wordnet_lemmatizer.lemmatize(t) for t in no_stops]

    articles.append(lemmatized)

dictionary = Dictionary(articles)
corpus = [dictionary.doc2bow(a) for a in articles]
total_word_count = defaultdict(int)

for word_id  , word_count in itertools.chain.from_iterable(corpus):
    total_word_count[word_id] += word_count

sorted_word_count = sorted(total_word_count.items(),key=lambda w:w[1],reverse=True)
for word_id , word_count in sorted_word_count[0:1]:
    print("คำที่มากที่สุด : " , dictionary.get(word_id) , word_count)

print("คำที่ไม่ซ้ำทั้งหมด : " ,len(sorted_word_count))