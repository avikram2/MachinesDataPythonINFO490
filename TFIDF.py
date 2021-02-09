def get_corpus():

    c1 = "Do you like Green eggs and ham"
    c2 = "I do not like them Sam I am  I do not like Green eggs and ham"
    c3 = "Would you like them Here or there"
    c4 = "I would not like them Here or there I would not like them Anywhere"
    return [c1, c2, c3, c4]
    
def split_into_tokens(data, normalize=True, min_length=0, stopwords = []):

   # returns an array of words
   # each word is simply a string
  arr = data.split()
  if (normalize):
    arr = [x.lower() for x in arr]
    if len(stopwords):
      stopwords = [elem.lower() for elem in stopwords]
  if len(stopwords):
    arr = [elem for elem in arr if elem not in stopwords]
  arr = [x for x in arr if len(x) > min_length]
   # splits the incoming data (a string) based on white-space
   # if normalize is True, normalize the case of the words
   
   # only return those words/tokens longer than min_length
  return arr
  
def test_split():
  corpus   = get_corpus()
  doc1     = corpus[0]
  print(split_into_tokens(doc1))
import collections

def build_tf(corpus, min_length=0, stopwords = []):

   # corpus is a list of documents
   # a document is an unparsed string of words
  master_list = []
  tf = list()
  for document in corpus:
    doc = collections.Counter(split_into_tokens(document, min_length = min_length, stopwords = stopwords))
    for key in doc:
      doc[key] /= len(split_into_tokens(document, min_length = min_length, stopwords = stopwords))
    tf.append(doc)
    master_list += (split_into_tokens(document, min_length = min_length, stopwords = stopwords))
  vocab = collections.Counter(master_list)



  return  vocab, tf
   
def test_tf():
  corpus = get_corpus()
  vocab, tf = build_tf(corpus)
  print(tf[0]['eggs'])  # 0.143
  print(tf[3]['there']) # 0.0714


import math
def build_idf(vocabulary, corpus_tf):

    # return a collection.Counter object
    # such that counter[term] is the idf for that term
    
    term_idf = collections.Counter()
    for word in vocabulary:
      term_count = 0
      for doc in corpus_tf:
        if doc[word] > 0:
          term_count+=1
      term_idf[word] = math.log(len(corpus_tf)/term_count)

    return term_idf


def compute_TFIDF(doc_tf, idf):
  x = collections.Counter()
  for key in doc_tf:
    x[key] = idf[key] * doc_tf[key]
  return x
def build_tf_idf(tfs, idf):
  tfidf = [collections.Counter() for x in tfs]
  for idx, doc_tf in enumerate(tfs):
    tfidf[idx] = compute_TFIDF(doc_tf, idf)
  return tfidf
def test_tfidf():
  corpus = get_corpus()
  vocab, tf = build_tf(corpus)
  idf = build_idf(vocab, tf)
  tfidf = build_tf_idf(tf, idf)

