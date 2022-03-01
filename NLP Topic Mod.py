#Actual Text Mining
#initial pdfminer,regex, os, pandas
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

#pdfminer imports
from pdfminer.converter import TextConverter
from pdfminer.layout import LAParams
from pdfminer.pdfdocument import PDFDocument
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.pdfpage import PDFPage
from pdfminer.pdfparser import PDFParser

# tokenizer imports
from nltk import word_tokenize
from nltk.corpus import stopwords
import string
# from nltk.probability import FreqDist

# analyze LDA
import pyLDAvis
# import pyLDAvis.gensim
import pyLDAvis.gensim_models
from IPython.core.display import HTML

doc_list = []

# main this ig
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

        # remove stopwords
        stop_words = set(stopwords.words("english"))
        data_words = [w for w in words if not w in stop_words]

        #return list
        return(data_words)

    # give path of directory
    path_of_the_directory = r"PDFs"

    # get iterator
    object = os.scandir(path_of_the_directory)
    print("Files and Directories in '% s':" % path_of_the_directory)

    # iterate through list
    for n in object:
        if n.is_dir() or n.is_file():

            # pdfminer text harvesting
            output_string = StringIO()
            with open(n, 'rb') as in_file:
                parser = PDFParser(in_file)
                doc = PDFDocument(parser)
                rsrcmgr = PDFResourceManager()
                device = TextConverter(rsrcmgr, output_string, laparams=LAParams())
                interpreter = PDFPageInterpreter(rsrcmgr, device)
                for page in PDFPage.create_pages(doc):
                    interpreter.process_page(page)

                # add full text string to variable
                full_text = (output_string.getvalue())

                doc = token_gen(full_text)
                doc_list.append(doc)

    # Create Dictionary
    id2word = corpora.Dictionary(doc_list)

    # Create Corpus
    texts = doc_list

    # Term Document Frequency
    corpus = [id2word.doc2bow(text) for text in texts]

    # number of topics
    num_topics = 10

    # build LDA Model
    lda_model = gensim.models.LdaMulticore(corpus=corpus, id2word=id2word, num_topics=num_topics)

    # Print the Keyword in the 10 topics
    #print(lda_model.print_topics())
    doc_lda = lda_model[corpus]

    # Visualize the topics
    # visualisation = pyLDAvis.gensim_models.prepare(lda_model, corpus, id2word)
    # pyLDAvis.save_html(visualisation, 'LDA_Visualization.html')

    # Compute Coherence Score
    coherence_model_lda = CoherenceModel(model=lda_model, texts=doc_list, dictionary=id2word, coherence='c_v')
    coherence_lda = coherence_model_lda.get_coherence()
    print('\nCoherence Score: ', coherence_lda)

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

if __name__ == "__main__":
    main()

