# steps for text ranking :
# combine text from different sources
# split into sentences
# convert sentences into vectors
# create similarity matrix of all vectors
# convert similarity matrix to a graph with sentence vectors as vertices and similarity score as edges
# calculate ranks of sentences


# import standard modules -----------------------------------------------------
from typing import List, Tuple

# import third party modules --------------------------------------------------
import numpy
import spacy
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
import nltk.data

# import custom modules -------------------------------------------------------
# from text_preprocessing import text_preprocessor


def create_summary(text: str, len_summary: int) -> str:
  r"""returns summary"""

  sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')

  list_text: List[str] = sent_detector.tokenize(text.strip())

  # en_core_web_lg is a pre-trained model like glove for word embeddings
  nlp = spacy.load('en_core_web_lg')

  # creating a corpus of all sentences from input
  doc = [nlp(y) for x in list_text for y in x]

  # creating vectors of sentences with help of en_core_web_lg
  vectors = [token.vector for x in doc for token in x]
  
  # creating empty matrix for storing similarity scores between sentence vectors
  sim_mat = numpy.zeros([len(list_text), len(list_text)])

  # calculating similarity scores using cosine similarity and putting them in the similarity matrix
  for i in range(len(list_text)):
    for j in range(len(list_text)):
      if i != j:
        sim_mat[i][j] = cosine_similarity(vectors[i].reshape(1,300), vectors[j].reshape(1,300))[0,0]

  # creating a network graph with sentences as vertices and similarity scores as weights from the similarity matrix
  nx_graph = nx.from_numpy_array(sim_mat)

  # applying pagerank algorithm on the graph to calculate relevancy scores for each sentence in input
  scores = nx.pagerank(nx_graph)

  # sorting sentences on the basis of scores
  ranked_sentences = sorted(((scores[i],s) for i,s in enumerate(list_text)), reverse=True)

  
  # creating summary text by joining top n needed sentences from ranked sentences
  summary=''
  for i in range(len_summary):
    f = ranked_sentences[i][1]
    summary = summary+f

  return summary
  
if __name__=="__main__":
  text = '''WordPress (WP, WordPress.org) is a free and open-source content management system (CMS) written in PHP[4] and paired with a MySQL or MariaDB database. Features include a plugin architecture and a template system, referred to within WordPress as Themes. WordPress was originally created as a blog-publishing system but has evolved to support other web content types including more traditional mailing lists and forums, media galleries, membership sites, learning management systems (LMS) and online stores. One of the most popular content management system solutions in use, WordPress is used by 42.8% of the top 10 million websites as of October 2021.[5][6]

WordPress was released on May 27, 2003, by its founders, American developer Matt Mullenweg[1] and English developer Mike Little,[7][8] as a fork of b2/cafelog. The software is released under the GPLv2 (or later) license.[9]

To function, WordPress has to be installed on a web server, either part of an Internet hosting service like WordPress.com or a computer running the software package WordPress.org in order to serve as a network host in its own right.[10] A local computer may be used for single-user testing and learning purposes.

WordPress Foundation owns WordPress, WordPress project and other related trademarks.[11]
'''
  # text = input("Enter text to summarize: ")
  lines_summary = int(input("Number of lines required: "))
  summary = create_summary(text,lines_summary)
  print(summary)
  
