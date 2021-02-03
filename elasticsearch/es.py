# imports
from datetime import datetime
import json
import os
import sys
import time
from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk
import tensorflow as tf
import tensorflow_hub as hub

# parse arguments
print(len(sys.argv), sys.argv)
# quit if arguments missing
if len(sys.argv) == 1 or not '-' in sys.argv[1]:
    print('use "-up" or "-down" to setup your elasticsearch instance')
    print('use "-index" to ingest data into the index')
    print('use "-update" to update data (except vectors)')
    print('provide a dataset or "*" as second argument')
    sys.exit()
else:
    mode = sys.argv[1]

if mode == '-index' or mode == '-update':
    if len(sys.argv) < 3:
        print('second argument missing')
        print('provide a dataset or "*" as second argument')
        sys.exit()
    else:
        dataset = sys.argv[2].split('+')
else:
    dataset = None

# es instance name
index = 'usecase2ml'

# set treshold for category and subcategory score
treshold = 0.25

# print some values
print('index:', index)
print('mode:', mode)
print('dataset:', dataset)

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


# Refer: https://www.elastic.co/guide/en/elasticsearch/reference/current/mapping.html
# Mapping: Structure of the index
# Property/Field: name and type

mapping = {
    "title": {
        "type": "text",
        "analyzer": "english"
    },
    "title_boolean": {
        "type": "text",
        "analyzer": "english",
        "similarity": "boolean"
    },
    "title_vector_use4": {
        "type": "dense_vector",
        "dims": 512
    },
    "title_vector_use5": {
        "type": "dense_vector",
        "dims": 512
    },
    "summarization": {
        "type": "text",
        "analyzer": "english"
    },
    "summarization_boolean": {
        "type": "text",
        "analyzer": "english",
        "similarity": "boolean"
    },
    # "summarization_lemmatized": {
    #     "type": "text",
    #     "analyzer": "english"
    # },
    "summarization_vector_use4": {
        "type": "dense_vector",
        "dims": 512
    },
    "summarization_vector_use5": {
        "type": "dense_vector",
        "dims": 512
    },
    "fulltext": {
        "type": "text",
        "analyzer": "english"
    },
    "fulltext_boolean": {
        "type": "text",
        "analyzer": "english",
        "similarity": "boolean"
    },
    "fulltext_vector_use4": {
        "type": "dense_vector",
        "dims": 512
    },
    "fulltext_vector_use5": {
        "type": "dense_vector",
        "dims": 512
    },
    "words": {
        "type": "float"
    },
    "sum_words": {
        "type": "float"
    },
    "link": {
        "type": "text"
    },
    "source": {
        "type": "text",
        "fields": {
            "keyword": {
                "type": "keyword"
            }
        }
    },
    "category": {
        "type": "text",
        "analyzer": "english",
        "fields": {
            "keyword": {
                "type": "keyword"
            }
        }
    },
    "category_score": {
        "type": "float"
    },
    "subcategory": {
        "type": "text",
        "analyzer": "english",
        "fields": {
            "keyword": {
                "type": "keyword"
            }
        }
    },
    "subcategory_score": {
        "type": "float"
    },
    "tags": {
        "type": "text",
        "fields": {
            "keyword": {
                "type": "keyword"
            }
        }
    },
    "tags_descriptive": {
        "type": "text",
        "fields": {
            "keyword": {
                "type": "keyword"
            }
        }
    },
    "kind": {
        "type": "text",
        "fields": {
            "keyword": {
                "type": "keyword"
            }
        }
    },
    "ml_libs": {
        "type": "text",
        "fields": {
            "keyword": {
                "type": "keyword"
            }
        }
    },
    "host": {
        "type": "text",
        "fields": {
            "keyword": {
                "type": "keyword"
            }
        }
    },
    "license": {
        "type": "text",
        "fields": {
            "keyword": {
                "type": "keyword"
            }
        }
    },
    "programming_language": {
        "type": "text",
        "fields": {
            "keyword": {
                "type": "keyword"
            }
        }
    },
    "ml_score": {
        "type": "float"
    },
    "learn_score": {
        "type": "float"
    },
    "explore_score": {
        "type": "float"
    },
    "compete_score": {
        "type": "float"
    },
    "engagement_score": {
        "type": "float"
    },
    "date_project": {
        "type": "date"
    },
    "date_scraped": {
        "type": "date"
    },
}

b = {
    "mappings": {
        "properties": mapping
    }
}


# up index
if mode == '-up':
    # 400 caused by IndexAlreadyExistsException,
    ret = es.indices.create(index=index, ignore=400, body=b)
    print(json.dumps(ret, indent=4))
    print('elasticsearch instance created')
    print('Please visit: http://localhost:9200/'+index)
    sys.exit()

# drop indix
if mode == '-down':
    ret = es.indices.delete(index=index)
    print('elasticsearch instance dropped')
    sys.exit()


# ingest
if mode == '-index' or mode == '-update':
    # config
    FORCE_QUIT = 0          # 0 ... disable
    STORE_CHUNKS = True     # store chunks of 1000 records
    CHUNK_SIZE = 500

    if mode == '-index':
        # load embedding
        embeddings = {}
        # load USE4 model
        print('load USE4 embedding')
        use4_start = time.time()
        embeddings['use4'] = hub.load("./.USE4/")
        use4_end = time.time()
        print('loaded ('+str(round(use4_end-use4_start, 3))+'sec)')
        print('##################################################')

        # load USE5_large model
        print('load USE5_large embedding')
        use5_start = time.time()
        embeddings['use5'] = hub.load("./.USE5_large/")
        use5_end = time.time()
        print('loaded ('+str(round(use5_end-use5_start, 3))+'sec)')
        print('##################################################')

    # load dataset
    path = '../data/database/json/'
    subfolder = os.listdir(path)

    if '*' in dataset:
        folders = subfolder
    else:
        folders = set(dataset) & set(subfolder)
        if len(folders) == 0:
            print('dataset not found:', dataset)
            print('datasets available:', subfolder)
            sys.exit()

    print('datasets:', folders)

    i = last = x = 0
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

                id_ = file.replace('.json', '')
                # print(id_)

                raw = f.read()
                raw = json.loads(raw)

                # print(raw)
                # sys.exit()

                # cleanup record to prevent flooding
                record = {}

                # map all fields from mapping
                # for key in mapping.keys():
                #     if key in raw and raw[key] != '':
                #         record[key] = raw[key]
                record = {k: v for k, v in raw.items() if k in mapping.keys()}

                # if 'description_lemmatized' in raw:
                #     record['description'] = raw['description_lemmatized']

                # if 'summarization_lemmatized' in raw:
                #     record['summarization'] = raw['summarization_lemmatized']

                # if 'tags_descriptive' in raw:
                #     record['tags'] = raw['tags_descriptive']

                skip = False

                print(i, record['link'])

                # skip record if title is missing
                if not 'title' in record or record['title'] == '':
                    print('skipped - missing title')
                    skip = True
                else:
                    record['title'] = record['title'].strip()

                # skip if language_code is not 'en'
                languages = ['en', 'af']
                if not 'language_code' in raw:
                    print('skipped - language code missing')
                    skip = True
                elif not raw['language_code'] in languages:
                    print('skipped - wrong language: "' +
                          str(raw['language_code'])+'"')
                    skip = True

                # skip record if description is missing
                # if not 'description' in raw or raw['description'] == '':
                #     print('skipped - missing description')
                #     skip = True

                if skip == True:
                    x += 1
                else:
                    # fill summary
                    if not 'summarization' in record:
                        record['summarization'] = raw['description'] if 'description' in raw else ''
                        record['summarization'] = raw['sum_nltk'] if 'sum_nltk' in raw and raw['sum_nltk'] != '' else record['summarization']

                    # if not 'summarization_lemmatized' in record:
                    #     record['summarization_lemmatized'] = raw['description_lemmatized'] if 'description_lemmatized' in raw else ''

                    # print(record)
                    # sys.exit()

                    #print(i, record['link'])

                    # clear category and subcategory if below treshold
                    if 'category' in record:
                        if record['category_score'] < treshold:
                            print('dropped - category')
                            record.pop('category')
                            record.pop('category_score')
                        else:
                            record['category_score'] = round(
                                record['category_score'], 3)

                    if 'subcategory' in record:
                        if record['subcategory_score'] < treshold:
                            print('dropped - subcategory')
                            record.pop('subcategory')
                            record.pop('subcategory_score')
                        else:
                            record['subcategory_score'] = round(
                                record['subcategory_score'], 3)

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

                    # create fulltext
                    ft = record['summarization']

                    # append title
                    ft = record['title'] + '. ' + ft

                    # append subcategory
                    if 'subcategory' in record:
                        if isinstance(record['subcategory'], list):
                            s = ', '.join(record['subcategory'])
                        else:
                            s = record['subcategory']
                        ft = s + '. ' + ft

                    # append category
                    if 'category' in record:
                        if isinstance(record['category'], list):
                            s = ', '.join(record['category'])
                        else:
                            s = record['category']
                        ft = s + '. ' + ft

                    # append tags
                    if 'tags_descriptive' in record:
                        if isinstance(record['tags_descriptive'], list):
                            s = ', '.join(record['tags_descriptive'])
                        else:
                            s = record['tags_descriptive']
                        ft = ft + s

                    # store fulltext
                    record['fulltext'] = ft

                    # duplicates fields for boolean search
                    record['title_boolean'] = record['title']
                    record['summarization_boolean'] = record['summarization']
                    record['fulltext_boolean'] = record['fulltext']

                    # convert scores to float
                    for r in record:
                        if 'score' in r and r != '':
                            #print(i, record[i], type(record[i]))
                            record[r] = float(record[r])

                    if mode == '-index':
                        # create vectors
                        vectorize = [
                            'title', 'summarization', 'fulltext']
                        for field in vectorize:
                            # print(field)
                            for embed in embeddings.keys():
                                # print(embed)
                                vec = tf.make_ndarray(tf.make_tensor_proto(
                                    embeddings[embed]([record[field]]))).tolist()[0]
                                name = field.replace(
                                    '_lemmatized', '')+'_vector_'+embed
                                record[name] = vec

                    # print(record)
                    # sys.exit()

                    #b = {**record, **vectors}
                    # print(json.dumps(tmp,indent=4))

                    # res = es.index(index=index, body=b)
                    # print(res)
                    if mode == '-index':
                        bulk.append({
                            "index": {
                                "_id": id_
                            }
                        })
                        bulk.append(record)
                    else:
                        bulk.append({
                            "update": {
                                "_id": id_
                            }
                        })
                        bulk.append({"doc": record})

                        # res = es.update(index=index, id=id_,
                        #                 body={"doc": record})

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
        if len(bulk) > 0:
            res = es.bulk(index=index, body=bulk)
            res_clean = dict(res)
            res_clean['item_count'] = len(res['items'])
            res_clean.pop('items')
            print('===== RECORDS STORED IN ELASTICSEARCH =====')
            print(res_clean)
        else:
            print('FAILED: no records')
        print('')
        print('')

    end = time.time()

    print('##################################################')
    print('DONE - added:', i, 'items | skipped:', x,
          'items | in', round(end-start, 3), 'seconds')
