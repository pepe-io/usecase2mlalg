# Search Engine for Applications Using Machine Learning Algorithms

The aim of this project is to provide a search-engine for machine learning use cases.
Therefor we need at least two parts:

- data
- search engine

the purpose of the search engine is to implement and analyze different strategies to enable search based on relevance  
it evaluates different search methods like:

- keyword based search (TF\*IDF/BM25)
- keyword based search (boolean)
- semantic search (USE (DAN) and USE large (Transformer))
  and explore methods of boosting and penalization

the current stage is a prototype to compare the different methods listed above  
rather than a self-updating seacrh engine

## pipelines

this folder contains Jupyter Notebooks to acquire and preprocess datasets for different data pools  
pipelines exist for following datapools:

- GitHub (Git repository, project website, project API) [01-09]
- Kaggle (competions, datasets, notebooks) [20-29]
- The Clever Programmer [40]
- Youtube (Two Minute Papers) [60]

there are also scripts to evaluate the different summarizers and categorizers [90]  
to equalize the different data-sets [90]  
to visualize the final data [91]
to explore n-grams of the final data [92]  
evaluate Universal Sentence Encoder (USE) in our search engine scenario [93]

## data

### ./data/database

contains CSV representation for different stages of the preprocessing steps (up until [90])

### ./data/patterns

contains the patterns used to detect the usage of machine lerning due to the presence of:

- machine learning terms
- machine learning abbreviations
- libraries and packages used in machine learning

the filter is a dictionary to equalize the tags out of different data-pools in a uniform representation suitable for the needs of the search engine

### ./data/json

contain the final records (out of step [90]) before vectorization & ingest

## elasticsearch

contains the search engine based on Elasticsearch (ES)
in mainly consists of two scripts

### es.py

to ingest the data from ./data/json into the search index

### app.py

a Flask App as wrapper for an Elasticsearch Docker Instance  
it contains all business logic and provide a GUI as well as API support
