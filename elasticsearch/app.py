# imports
from flask import Flask, render_template, request
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
import json
import sys
from elasticsearch import Elasticsearch
# from elasticsearch.helpers import bulk
import tensorflow as tf
import tensorflow_hub as hub

# define flask app
app = Flask(__name__)

# activate ENV
# activate env
# source env/bin/activate				# mac
# source \env\Scripts\activate.bat	    # win
# .\env\Scripts\activate				# win10

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

# embed = hub.load('https://tfhub.dev/google/universal-sentence-encoder/4')
embed = hub.load('./.USE4/')


def keySearch(es, queries):
    '''
    Search by Keyword, td-idf
    '''
    # Search by Keywords
    # b = {
    #     'size': 1,
    #     'query': {
    #         'match': {
    #             tag: q
    #         }
    #     }
    # }

    q = {}
    for tag in queries:
        q.update({"match": {tag: queries[tag]}})
    b = {
        "size": 10,
        "query": {
            "bool": {
                "must": [
                    q
                ]
            }
        }
    }
    res = es.search(index='usecase2mlalg', body=b)
    return res


def semSearch(es, sent, tag_, queries):
    '''
    Search by Vec Similarity
    '''
    query_vector = tf.make_ndarray(
        tf.make_tensor_proto(embed([sent]))).tolist()[0]
    q = {}
    for tag in queries:
        q.update({"match": {tag: queries[tag]}})

    # print(q)
    if len(q) == 0:
        b = {
            'size': 10,
            'query': {
                'script_score': {
                    'query': {
                        'match_all': {}
                    },
                    'script': {
                        'source': 'cosineSimilarity(params.query_vector, "'+tag_+'") + 1.0',
                        'params': {'query_vector': query_vector}
                    }
                }
            }
        }
    else:
        b = {
            'size': 10,
            'query': {
                'script_score': {
                    "query": {
                        "bool": {
                            "must": [
                                q
                            ]
                        }
                    },
                    'script': {
                        'source': 'cosineSimilarity(params.query_vector, "'+tag_+'") + 1.0',
                        'params': {'query_vector': query_vector}
                    }
                }
            }
        }

    # print(json.dumps(b,indent=4))
    res = es.search(index='usecase2mlalg', body=b)

    return res


def format_result(res, extra={}):
    '''
    parse formatting for layout
    '''
    ret = {}
    # if len(extra) > 0:
    ret['id'] = res['_id']
    ret['search_score'] = res['_score']
    ret.update(res['_source'])
    ret.pop('description')
    ret.pop('title_vector')
    ret.pop('description_vector')
    ret.update(extra)

    return ret


def bundle_results(items):
    '''
    bundle query-results
    similar result by engine get grouped together
    '''
    ret = []
    ids = []

    for item in items:
        # print(item['id'])
        if not item['id'] in ids:
            ids.append(item['id'])
            ret.append(item)
        else:
            i = next((i for i, key in enumerate(ret)
                      if key['id'] == item['id']), None)
            #print(i, item['id'], ret[i]['id'])
            ret[i]['search'] = '*' + ret[i]['search'] + '/' + item['search']

    return ret


@ app.route('/', methods=['GET', 'POST'])
@ app.route('/<query>', methods=['GET'])
# index / search route
def search(query=''):
    '''
    index / search route
    '''
    # get query from POST
    if request.method == 'POST':
        q = request.form['search']

    # get query from GET
    if request.method == 'GET':
        q = query.replace('+', ' ')

    # items for view
    items = []

    # filters are stored as queries
    queries = {}
    # query engines and their html-from-counterpart
    query_engines = {
        'kt': ' checked',
        'st': ' checked',
        'kd': ' checked',
        'sd': ' checked',
    }
    # css-attribute to hide filters (default value)
    show_filter = 'hidden'

    if q != '':
        i = 0

        # add filters to tags
        if request.form.get('tags'):
            queries['tags'] = request.form.get('tags')
            show_filter = ''
        if request.form.get('kind'):
            queries['kind'] = request.form.get('kind')
            show_filter = ''
        if request.form.get('libs'):
            queries['ml_libs'] = request.form.get('libs')
            show_filter = ''

        # print(queries)

        # perform keyword search for title
        if request.form.get('kt'):
            query_engines['kt'] = ' checked'
            # res_kw = keywordSearch(es, q, 'title')
            res_kw = keySearch(es, dict({'title': q}, **queries))
            for hit in res_kw['hits']['hits']:
                items.append(format_result(
                    hit, extra={'index': 'index'+str(i), 'search': 'KT'}))
                i += 1
        else:
            query_engines['kt'] = ''

        # perform keyword search for description
        if request.form.get('kd'):
            query_engines['kd'] = ' checked'
            # res_kw = keywordSearch(es, q, 'description')
            res_kw = keySearch(es, dict({'description': q}, **queries))
            for hit in res_kw['hits']['hits']:
                items.append(format_result(
                    hit, extra={'index': 'index'+str(i), 'search': 'KD'}))
                i += 1
        else:
            query_engines['kd'] = ''

        # perform semantic search for title
        if request.form.get('st'):
            query_engines['st'] = ' checked'
            # res_semantic = sentenceSimilaritybyNN(es, q, 'title_vector')
            res_semantic = semSearch(es, q, 'title_vector', queries)
            for hit in res_semantic['hits']['hits']:
                items.append(format_result(
                    hit, extra={'index': 'index'+str(i), 'search': 'ST'}))
                i += 1
        else:
            query_engines['st'] = ''

        # perform semantic search for description
        if request.form.get('sd'):
            query_engines['sd'] = ' checked'
            #res_semantic = sentenceSimilaritybyNN(es, q, 'description_vector')
            res_semantic = semSearch(es, q, 'description_vector', queries)
            for hit in res_semantic['hits']['hits']:
                items.append(format_result(
                    hit, extra={'index': 'index'+str(i), 'search': 'SD'}))
                i += 1
        else:
            query_engines['sd'] = ''

    # bundle items (similar matches by engine get grouped together)
    items = bundle_results(items)
    # print(items)

    # return view
    queries['query'] = q
    return render_template('index.html', query=queries, query_engines=query_engines, items=items, show_filter=show_filter)


# start app
if __name__ == '__main__':
    app.run(debug=True)
