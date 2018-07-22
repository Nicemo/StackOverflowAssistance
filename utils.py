import nltk
import pickle
import re
import numpy as np
import gensim
nltk.download('stopwords')
from nltk.corpus import stopwords


RESOURCE_PATH = {
    'INTENT_RECOGNIZER': 'intent_recognizer.pkl',
    'TAG_CLASSIFIER': 'tag_classifier.pkl',
    'TFIDF_VECTORIZER': 'tfidf_vectorizer.pkl',
    'THREAD_EMBEDDINGS_FOLDER': 'thread_embeddings_by_tags',
    'WORD_EMBEDDINGS': 'word_embeddings.tsv',
}


def text_prepare(text):
        
    replace_by_space_re = re.compile('[/(){}\[\]\|@,;]')
    bad_symbols_re = re.compile('[^0-9a-z #+_]')
    stopwords_set = set(stopwords.words('english'))

    text = text.lower()
    text = replace_by_space_re.sub(' ', text)
    text = bad_symbols_re.sub('', text)
    text = ' '.join([x for x in text.split() if x and x not in stopwords_set])

    return text.strip()


def load_embeddings(embeddings_path):
    """载入词向量（tsv文件）

    Args:
      路径

    Returns:
      embeddings - 对应的词向量
      embeddings_dim - 向量维度
    """
       

    embeddings = KeyedVectors.load_word2vec_format(embeddings_path, binary=True)
    embeddings_dim = embeddings.vector_size
    return embeddings, embeddings_dim

    
     

def question_to_vec(question, embeddings, dim):
        
    result = [embeddings[w] for w in question.split() if w in embeddings]
    return np.mean(result, axis=0) if len(result) > 0 else np.zeros((dim, ))


def unpickle_file(filename):
    
    with open(filename, 'rb') as f:
        return pickle.load(f)
