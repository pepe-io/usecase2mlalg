# imports
from flask import Flask, request
from datetime import datetime
import json
import os
import sys
# import csv
# import pandas as pd
import time
from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk
import tensorflow as tf
import tensorflow_hub as hub


# es instance names
es_indexes = {
    'use': 'usecase2mlalg',
    'use_large': 'usecase2mlalg_large'
}


# parse arguments
print(len(sys.argv), sys.argv)
if len(sys.argv) < 3:
    print('use "use" or "use_large" as argument to select an elasticsearch instance')
    print('provide a dataset or * as second argument')
    sys.exit()
else:
    model = sys.argv[1]
    index = es_indexes[model]
    dataset = sys.argv[2]
    print('index:', index)


# connect to ES on localhost on port 9200
print('##################################################')
print('Connecting to Elasticsearch...')
es = Elasticsearch([{'host': 'localhost', 'port': 9200}])
if es.ping():
    print('established')
    print('try: http://localhost:9200/'+index)
else:
    print('FAILED!')
    sys.exit()
print('##################################################')


# load USE4 model
if model == 'use':
    # embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")
    embed = hub.load("./.USE4/")
# load USE5_large model
elif model == 'use_large':
    embed = hub.load("./.USE5_large/")
# exit if model is not known
else:
    print('model not defined')
    sys.exit()


# load dataset
path = '../data/database/json/'
subfolder = os.listdir(path)


# CONSTANTS
FORCE_QUIT = 0          # 0 ... disable
STORE_CHUNKS = True     # store chunks of 1000 records
i = last = 0

if dataset == '*':
    folders = subfolder
else:
    if dataset in subfolder:
        folders = [dataset]
    else:
        print('dataset not found:', dataset)
        print('datasets available:', subfolder)
        sys.exit()

print('datasets:', folders)

start = time.time()
for folder in folders:
    bulk = []
    j = 0

    fp = os.path.join(path, folder)
    files = os.listdir(fp)
    print('items:', len(files))

    for file in files:
        fp = os.path.join(path, folder, file)

        with open(fp, encoding='utf-8') as f:

            id = file.replace('.json', '')
            # print(id)

            raw = f.read()
            raw = json.loads(raw)

            # print(raw)
            # sys.exit()

            # cleanup record to prevent flooding
            record = {}

            if 'title' in raw and raw['title'] != '':
                record['title'] = raw['title']
            else:
                record['title'] = ''

            if 'description' in raw and raw['description'] != '':
                record['description'] = raw['description']
            else:
                record['description'] = ''

            if 'words' in raw and raw['words'] != '':
                record['words'] = raw['words']

            if 'summarization' in raw and raw['summarization'] != '':
                record['summarization'] = raw['summarization']
            else:
                record['summarization'] = record['description']

            if 'sum_words' in raw and raw['sum_words'] != '':
                record['sum_words'] = raw['sum_words']

            if 'link' in raw and raw['link'] != '':
                record['link'] = raw['link']

            if 'category' in raw and raw['category'] != '':
                record['category'] = raw['category']

            if 'category_score' in raw and raw['category_score'] != '':
                record['category_score'] = raw['category_score']

            if 'subcategory' in raw and raw['subcategory'] != '':
                record['subcategory'] = raw['subcategory']

            if 'subcategory_score' in raw and raw['subcategory_score'] != '':
                record['subcategory_score'] = raw['subcategory_score']

            if 'tags' in raw and raw['tags'] != '':
                record['tags'] = raw['tags']

            if 'kind' in raw and raw['kind'] != '':
                record['kind'] = raw['kind']

            if 'ml_libs' in raw and raw['ml_libs'] != '':
                record['ml_libs'] = raw['ml_libs']

            if 'host' in raw and raw['host'] != '':
                record['host'] = raw['host']

            if 'license' in raw and raw['license'] != '':
                record['license'] = raw['license']

            if 'language' in raw and raw['language'] != '':
                record['language'] = raw['language']

            if 'score' in raw and raw['score'] != '':
                record['score'] = raw['score']

            if 'date_project' in raw and raw['date_project'] != '':
                record['date_project'] = raw['date_project']

            if 'date_scraped' in raw and raw['date_scraped'] != '':
                record['date_scraped'] = raw['date_scraped']

            # print(record)
            # sys.exit()

            #print(i, record['link'])

            # convert date-strings to datetime objects
            if 'date_project' in record:
                try:
                    record['date_project'] = datetime.strptime(
                        record['date_project'], "%Y-%m-%d %H:%M:%S")
                except:
                    record.pop('date_project')
            if 'date_scraped' in record:
                try:
                    record['date_scraped'] = datetime.strptime(
                        record['date_scraped'], "%Y-%m-%d %H:%M:%S")
                except:
                    record.pop('date_scraped')

            # append category and subcategory on summary
            # if 'subcategory' in record:
            #     if isinstance(record['subcategory'], list):
            #         s = record['subcategory'][0]
            #     else:
            #         s = record['subcategory']
            #     record['summarization'] = s + '. ' + record['summarization']

            # if 'category' in record:
            #     if isinstance(record['subcategory'], list):
            #         s = record['category'][0]
            #     else:
            #         s = record['category']
            #     record['summarization'] = s + '. ' + record['summarization']

            vec_t = tf.make_ndarray(tf.make_tensor_proto(
                embed([record['title']]))).tolist()[0]
            vec_d = tf.make_ndarray(tf.make_tensor_proto(
                embed([record['description']]))).tolist()[0]
            vec_s = tf.make_ndarray(tf.make_tensor_proto(
                embed([record['summarization']]))).tolist()[0]

            vectors = {
                "title_vector": vec_t,
                "description_vector": vec_d,
                "summarization_vector": vec_s,
            }
            record.update(vectors)

            # print(record)
            # sys.exit()

            #b = {**record, **vectors}
            # print(json.dumps(tmp,indent=4))

            # res = es.index(index=index, body=b)
            # print(res)
            bulk.append({
                "index": {
                    "_id": id
                }
            })
            bulk.append(record)

            # keep count of rows processed
            i += 1
            j += 1
            if i % 100 == 0:
                print('total:', i, '/ batch:', j, 'of',
                      len(files), '/ folder:', folder)

            if STORE_CHUNKS == True and i % 1000 == 0:
                print('')
                print(i, 'storing items')
                res = es.bulk(index=index, body=bulk)
                res_clean = dict(res)
                res_clean['item_count'] = len(res['items'])
                res_clean.pop('items')
                print('===== RECORDS STORED IN ELASTICSEARCH =====')
                print(res_clean)
                print('')
                print('')

                bulk = []

            end = time.time()
            dur = round(end-start, 0)
            if dur % 10 == 0 and dur != last:
                print('')
                print('----- RUNNING:', dur, 'seconds -----')
                print('')
                last = dur

            if FORCE_QUIT != 0 and i >= FORCE_QUIT:
                print('FORCED QUIT')
                break

    print('')
    print(i, 'storing items')
    res = es.bulk(index=index, body=bulk)
    res_clean = dict(res)
    res_clean['item_count'] = len(res['items'])
    res_clean.pop('items')
    print('===== RECORDS STORED IN ELASTICSEARCH =====')
    print(res_clean)
    print('')
    print('')

end = time.time()

print('##################################################')
print('DONE - added', i, 'items in', round(end-start, 3), 'seconds')
