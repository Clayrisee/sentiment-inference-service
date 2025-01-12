# File for preprocess
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import nltk
import pickle
from scipy.sparse import csr_matrix

nltk.download('stopwords')


class Preprocess:
    """
    Preprocessing Data
    """
    def __init__(self, vectorizer_path: str='assets/vectorizer.pickle') -> None:
        self.vectorizer = pickle.load(open(vectorizer_path, 'rb'))
        self.port_stem = PorterStemmer()

    def __run_stemming(self, content: str) -> str:
        stemmed_content = re.sub('[^a-zA-Z]', " ", content)   # the regular expression matches any pattern that is not a character
                                                            # (since negation ^ is used) and replaces those matched sequences 
                                                            # with empty space, thus all special characters and digits get 
                                                            # removed.
        stemmed_content = stemmed_content.lower()
        stemmed_content = stemmed_content.split()
        stemmed_content = [self.port_stem.stem(word) for word in stemmed_content if word not in stopwords.words('english')]   
                                                            # apply port_stem only on words not in the list of stop-words
        stemmed_content = " ".join(stemmed_content)
        # print(stemmed_content)
        return stemmed_content

    def run_preprocess(self, text: str) -> csr_matrix:
        stemmed_text = self.__run_stemming(content=text)
        return self.vectorizer.transform([stemmed_text])


if __name__ == '__main__':
    preprocess = Preprocess()
    print(preprocess.run_preprocess('that food is so good'))
    print(type(preprocess.run_preprocess('that food is so good')))