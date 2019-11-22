from bs4 import BeautifulSoup
from gensim.utils import simple_preprocess
from gensim.models import Word2Vec
from gensim.parsing.preprocessing import remove_stopwords
soup = BeautifulSoup(open("Question1.txt"), 'html.parser')
for junk in soup(["docno", "date", "author", "favorite"]):
    junk.decompose()
soup.prettify()
data = str(soup.get_text()).splitlines()
preprocessed_data = []
i = 0
for line in data:
    if line != '':
        sline = line.split("\t", 1)
        if len(sline) == 2:
            sline[1] = remove_stopwords(sline[1])
            preprocessed_data.append(simple_preprocess(sline[1]))
        else:
            sline[0] = remove_stopwords(sline[0])
            preprocessed_data.append(simple_preprocess(sline[0]))
model = Word2Vec(preprocessed_data, size=300, window=5, min_count=3, workers=8)
print(model.wv.most_similar("amazed"))
