# imports
from flask import Flask, request
from datetime import datetime
import json
import sys
import csv
import pandas as pd
from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk
import tensorflow as tf
import tensorflow_hub as hub


# connect to ES on localhost on port 9200
print('##################################################')
print('Connecting to Elasticsearch...')
es = Elasticsearch([{'host': 'localhost', 'port': 9200}])
if es.ping():
    print('established')
    print('try: http://localhost:9200/usecase2mlalg')
else:
    print('FAILED!')
    sys.exit()
print('##################################################')


# load USE4 model

# embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")
embed = hub.load("./.USE4/")


def str_to_list(s):
    s = s.replace("'", "").replace(' ,', ',').replace(
        '[', '').replace(']', '').split(',')
    s = [i for i in s if i]
    return s


def mapper(row, style):
    '''
    mapper to adopt csv to db-schema

    title, title_vector, description, description_vector,
    link, category, category_score, subcategory, subcategory_score, 
    tags, kind, ml_libs, host, license, language, score,
    date_project, date_scraped
    '''

    # kaggle mapping
    if style == 'kaggle':
        return {
            'title': row['title'],
            'description': row['description'],
            'link': row['link'],
            # 'category': '',
            # 'category_score': 0,
            # 'subcategory': '',
            # 'subcategory_score': 0,
            'tags': list(set(str_to_list(row['tags']) + str_to_list(row['tags']))),
            'kind': 'project',
            'ml_libs': str_to_list(row['ml_libs']),
            'host': 'www.kaggle.com',
            'license': row['license'],
            'language': row['type'],
            'score': row['score_views'],
            'date_project': datetime.strptime(row['date'], "%Y-%m-%d %H:%M:%S"),
            'date_scraped': datetime.strptime(row['scraped_at'], "%Y-%m-%d %H:%M:%S"),
            # 'ml_terms': row['ml_terms'],
            # 'score_raw': json.dumps({'views': row['views'], 'votes': row['votes'], 'score_private': row['score_private'], 'score_public': row['score_public']}),
        }

    # github mapping
    if style == 'github':
        cat_score = 1 if row['industry'] != '' else 0
        subcat_score = 1 if row['type'] != '' else 0
        #tags = row['ml_tags'] if len(row['ml_tags']) > 0 else ''
        return {
            'title': row['name'],
            'description': row['description2'],
            'link': row['link'],
            'category': row['industry'],
            'category_score': cat_score,
            'subcategory': row['type'],
            'subcategory_score': subcat_score,
            'tags': str_to_list(row['ml_tags']),
            'kind': 'Project',
            'ml_libs': str_to_list(row['ml_libs']),
            'host': 'www.github.com',
            'license': row['license'],
            'language': row['language_primary'],
            'score': row['stars_score'],
            'date_project': datetime.strptime(row['pushed_at'], "%Y-%m-%d %H:%M:%S"),
            'date_scraped': datetime.strptime(row['scraped_at'], "%Y-%m-%d %H:%M:%S"),
            # 'ml_terms': row['keywords'],
            # 'score_raw': json.dumps({'stars': row['stars'], 'contributors': row['contributors']}),
        }

    # mlart mapping
    if style == 'mlart':
        title = row['Title'] if row['Title'] != '' else row['title']
        cat_score = 1 if row['Theme'] != '' else 0
        subcat_score = 1 if row['Medium'] != '' else 0
        return {
            'title': title,
            'description': row['subtitle'],
            'link': row['url'],
            'category': str_to_list(row['Theme']),
            'category_score': cat_score,
            'subcategory': str_to_list(row['Medium']),
            'subcategory_score': subcat_score,
            'tags': str_to_list(row['Technology']),
            'kind': 'Showcase',
            # 'ml_libs': [],
            'host': 'mlart.co',
            # 'license': '',
            # 'language': '',
            # 'score': 0,
            'date_project': datetime.strptime(row['Date'], "%Y-%m-%d"),
            'date_scraped': datetime.strptime(row['scraped_at'], "%Y-%m-%d %H:%M:%S"),
            # 'score_raw': json.dumps({'days_since_featured': row['Days Since Featured']}),
        }

    # thecleverprogrammer
    # date	link	ml_libs	ml_score	ml_slugs	ml_tags	ml_terms	text	title
    if style == 'tcp':
        return {
            'title': row['title'],
            'description': row['text'],
            'link': row['link'],
            # 'category': '',
            # 'category_score': 0,
            # 'subcategory': '',
            # 'subcategory_score': 0,
            'tags': str_to_list(row['ml_tags']),
            'kind': 'Project',
            'ml_libs': str_to_list(row['ml_libs']),
            'host': 'thecleverprogrammer.com',
            # 'license': '',
            'language': 'Python',
            # 'score': 0,
            'date_project': datetime.strptime(row['date'], "%Y-%m-%d %H:%M:%S"),
            'date_scraped': datetime.strptime('2020-12-20', "%Y-%m-%d"),
            # 'score_raw': json.dumps({'days_since_featured': row['Days Since Featured']}),
        }

    return None


# CONSTANTS
NUM_INDEXED = 100000

cnt = 0
i = 0

# kaggle competitions
csv_in = '../data/database/kaggle_competitions_01_original.csv'
csv_format = 'kaggle'

# mlart
csv_in = '../data/database/mlart_01_original.csv'
csv_format = 'mlart'

# github
csv_in = '../data/database/db_04_analyzed_v02.csv'
csv_format = 'github'

# thecleverprogrammer
csv_in = '../data/database/thecleverprogrammer_01_original.csv'
csv_format = 'tcp'

with open(csv_in, encoding='utf-8') as csvfile:
    # let's store converted csv to temp-folder for analysis
    csv_out = '../data/database/.temp/'
    df = pd.DataFrame()

    # readCSV = csv.reader(csvfile, delimiter=';')
    readCSV = csv.DictReader(csvfile, delimiter=';')
    # next(readCSV, None)  # skip the headers
    for row in readCSV:
        # print(row)
        row = mapper(row, csv_format)
        df = df.append(row, ignore_index=True)

        if row == None:
            print('mapping not found')
            sys.exit()
        # print(row)
        # print(row['title'])

        print(row['link'])
        i += 1

        vec_t = tf.make_ndarray(tf.make_tensor_proto(
            embed([row['title']]))).tolist()[0]
        vec_d = tf.make_ndarray(tf.make_tensor_proto(
            embed([row['description']]))).tolist()[0]

        #record = json.dumps(row)
        vectors = {
            "title_vector": vec_t,
            "description_vector": vec_d,
        }
        b = {**row, **vectors}
        # print(json.dumps(tmp,indent=4))

        res = es.index(index="usecase2mlalg", body=b)
        print(res)

        # keep count of # rows processed
        cnt += 1
        if cnt % 100 == 0:
            print(cnt)

        if cnt == NUM_INDEXED:
            break

    # store parsed csv
    fp = csv_in.split('/')[-1]
    df.to_csv(csv_out + fp, sep=';', index=False)


print('##################################################')
print('Done', i, 'items added')
