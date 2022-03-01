#imports
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
pd.set_option("max_rows", 600)
from pathlib import Path
import glob

#define directory path
directory_path = "Practice Research Files"

#list of files in directory
text_files = glob.glob(f"{directory_path}/*.txt")
text_titles = [Path(text).stem for text in text_files]

#vectorize
tfidf_vectorizer = TfidfVectorizer(input='text_titles', stop_words='english')

tfidf_vector = tfidf_vectorizer.fit_transform(text_files)

tfidf_df = pd.DataFrame(tfidf_vector.toarray(), index=text_titles, columns=tfidf_vectorizer.get_feature_names_out())

tfidf_df.loc['00_Document Frequency'] = (tfidf_df > 0).sum()

#compare against this bag of words?
tfidf_slice = tfidf_df["poverty", "hunger", "health", "well-being", "education",
                 "equality", "water", "sanitation", "affordable", "clean", "energy", "work", "growth",
                 "industry", "innovation", "infrastructure", "inequality", "sustainable", "responsible",
                 "consumption", "production", "environment", "action", "life", "peace", "justice", "institutions",
                 "partnership",]

tfidf_slice.sort_index().round(decimals=2)
