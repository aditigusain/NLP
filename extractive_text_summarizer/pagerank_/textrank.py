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
import nltk
import nltk.data
nltk.download('punkt')

# import custom modules -------------------------------------------------------
# from text_preprocessing import text_preprocessor


def create_summary(text: str, len_summary: int) -> str:
  r"""returns summary"""

  sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')

  list_text: List[str] = sent_detector.tokenize(text.strip())

  # en_core_web_lg is a pre-trained model like glove for word embeddings
  nlp = spacy.load('en_core_web_sm')

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
        sim_mat[i][j] = cosine_similarity(vectors[i].reshape(1,96), vectors[j].reshape(1,96))[0,0]

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
  # text = input("Enter text to summarize: \n")
  text: str = '''In botany, a fruit is the seed-bearing structure in flowering plants that is formed from the ovary after flowering.

Fruits are the means by which flowering plants (also known as angiosperms) disseminate their seeds. Edible fruits in particular have long propagated using the movements of humans and animals in a symbiotic relationship that is the means for seed dispersal for the one group and nutrition for the other; in fact, humans and many animals have become dependent on fruits as a source of food.[1] Consequently, fruits account for a substantial fraction of the world's agricultural output, and some (such as the apple and the pomegranate) have acquired extensive cultural and symbolic meanings.

In common language usage, "fruit" normally means the seed-associated fleshy structures (or produce) of plants that typically are sweet or sour and edible in the raw state, such as apples, bananas, grapes, lemons, oranges, and strawberries. In botanical usage, the term "fruit" also includes many structures that are not commonly called "fruits" in everyday language, such as nuts, bean pods, corn kernels, tomatoes, and wheat grains.[2][3]'''
  # text = input("Enter text to summarize: ")
  lines_summary: int = int(input("Number of lines required: "))
  summary: str = create_summary(text,lines_summary)
  print(summary)
  
