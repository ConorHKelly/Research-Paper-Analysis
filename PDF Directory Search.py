# initial pdfminer,regex, os, pandas
from io import StringIO
import re
import os
import pandas as pd


# pdfminer imports
from pdfminer.converter import TextConverter
from pdfminer.layout import LAParams
from pdfminer.pdfdocument import PDFDocument
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.pdfpage import PDFPage
from pdfminer.pdfparser import PDFParser


# pdfminer text harvesting
output_string = StringIO()
with open('PDFs/Urban Forest Biodiversity and Cardiovascular Disease.pdf', 'rb') as in_file:
    parser = PDFParser(in_file)
    doc = PDFDocument(parser)
    rsrcmgr = PDFResourceManager()
    device = TextConverter(rsrcmgr, output_string, laparams=LAParams())
    interpreter = PDFPageInterpreter(rsrcmgr, device)
    for page in PDFPage.create_pages(doc):
        interpreter.process_page(page)

# add full text string to variable
full_text = (output_string.getvalue())

# format full text, lowercase and split on spaces
lower = full_text.lower()
splat = lower.split(" ")

# initiate lists, one to remove newline characters, one to split new word chunks, one to store file data
res = []
new_res = []
instances = []
full_search = {}

# substitute newline character for a space
for sub in splat:
    res.append(re.sub("\n","",sub))

# split newly subbed items into two list elements
for item in res:
    new_res.extend(item.split(" "))

# create list of search criteria/goals
criteria_list = ["poverty", "hunger", "health", "well-being", "education",
                 "equality", "water", "sanitation", "affordable", "clean", "energy", "work", "growth",
                 "industry", "innovation", "infrastructure", "inequality", "sustainable", "responsible",
                 "consumption", "production", "environment", "action", "life", "peace", "justice", "institutions",
                 "partnership", ]

# create function to search through text
def word_search(criteria):
    key_word_count = {}

    # iterate through word list
    for word in new_res:

        # iterate through criteria list
        for term in criteria:

            # if selected words meets search criteria add to dictionary
            if word == term:
                instances.append(word)

    # iterate through list to count instances
    for prompt in instances:
        # if that day has not been added yet, add it
        if prompt not in key_word_count:
            key_word_count[prompt] = 1
        # if it has been added, add 1
        else:
            key_word_count[prompt] = key_word_count[prompt] + 1
    return(key_word_count)

# give path of directory
path_of_the_directory = r"C:\Users\conor\PycharmProjects\ResearchPaperAnalysis\Practice Research Files"

# get iterator
object = os.scandir(path_of_the_directory)
print("Files and Directories in '% s':" % path_of_the_directory)

# iterate through list
for n in object:
    if n.is_dir() or n.is_file():

        # open file
        fhand = open(n,encoding = "utf8")

        full_search[n] = word_search(criteria_list)

# print dictionaries on each line
for key, value in full_search.items():
    print(key, " : ", value)

#display all columns
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

# feed full search diction of diction into pandas dataframe
df = pd.DataFrame.from_dict(full_search, orient="index")
print(df)
#df.to_csv("Directory.Search.1.csv")

