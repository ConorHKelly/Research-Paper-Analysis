# Once converted to text - Topic Mod

# imports
from io import StringIO
import re
import os
import pandas as pd
import numpy as np
import tqdm

# Gensim
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel

# tokenizer imports
from nltk import word_tokenize
import nltk.corpus
from nltk.corpus import stopwords
import string
# from nltk.probability import FreqDist

# analyze LDA
import pyLDAvis
# import pyLDAvis.gensim
import pyLDAvis.gensim_models
from IPython.core.display import HTML

# Line chart imports
import matplotlib.pyplot as plt

# initialize list
doc_list = []

coherence_scores = []

counter = []

def main():

    # def tokenizer function
    def token_gen(text):
        # create tokens
        tokens = word_tokenize(text)

        # lower tokens
        tokens = [w.lower() for w in tokens]

        # table will remove punctuation
        table = str.maketrans("", "", string.punctuation)
        stripped = [w.translate(table) for w in tokens]
        words = [word for word in stripped if word.isalpha()]

        # remove stopwords & common characters
        stop_words = nltk.corpus.stopwords.words("english")
        new_stop_words = ["et", "al", "https", "e", "r", "n", "c", "l", "data", "model", "research", "fig"]
        stop_words.extend(new_stop_words)
        data_words = [w for w in words if not w in stop_words]

        # remove single letter characters
        for word in data_words:
            if len(word) == 1:
                data_words.remove(word)

        # return list
        return (data_words)

    # give path of directory
    path_of_the_directory = r"Converted to Text"

    # get iterator
    object = os.scandir(path_of_the_directory)
    print("Files and Directories in '% s':" % path_of_the_directory)

    # iterate through list
    for n in object:
        if n.is_dir() or n.is_file():

            with open(n, 'rb') as in_file:

                data = in_file.read().decode("utf-8")
                in_file.close()

                doc = token_gen(data)

                # print(doc[:150])

                doc_list.append(doc)

    count = 0

    for x in range(1,20):

        # Create Dictionary
        id2word = corpora.Dictionary(doc_list)

        # Create Corpus
        texts = doc_list

        # Term Document Frequency
        corpus = [id2word.doc2bow(text) for text in texts]

        # number of topics
        num_topics = x

        # build LDA Model
        lda_model = gensim.models.LdaMulticore(corpus=corpus, id2word=id2word, num_topics=num_topics)

        # Print the Keyword in the 10 topics
        #print(lda_model.print_topics())
        doc_lda = lda_model[corpus]

        # Compute Coherence Score
        coherence_model_lda = CoherenceModel(model=lda_model, texts=doc_list, dictionary=id2word, coherence='c_v')
        coherence_lda = coherence_model_lda.get_coherence()
        coherence_scores.append(coherence_lda)
        count = count + 1

        counter.append(count)


        # supporting function
        def compute_coherence_values(corpus, dictionary, k, a, b):
            lda_model = gensim.models.LdaMulticore(corpus=corpus,
                                                   id2word=dictionary,
                                                   num_topics=k,
                                                   random_state=100,
                                                   chunksize=100,
                                                   passes=10,
                                                   alpha=a,
                                                   eta=b)

            coherence_model_lda = CoherenceModel(model=lda_model, texts=doc_list, dictionary=id2word,
                                                 coherence='c_v')

            return coherence_model_lda.get_coherence()

    print(f"Coherence scores: {coherence_scores}")
    print(f"Count: {counter}")

    plt.plot(counter, coherence_scores)
    plt.title('Coherence Scores')
    plt.xlabel('Topic Number')
    plt.ylabel('Coherence Score')
    plt.show()

if __name__ == "__main__":
    main()