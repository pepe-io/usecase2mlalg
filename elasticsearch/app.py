# imports
from flask import Flask, render_template, request, send_from_directory, Response
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
import json
import sys
import os
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
    print(b)
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


@app.route('/favicon.ico')
def favicon():
    return send_from_directory(os.path.join(app.root_path, 'static'), 'favicon.ico', mimetype='image/vnd.microsoft.icon')


@ app.route('/', methods=['GET', 'POST'], endpoint='gui')
@ app.route('/api', methods=['GET'], endpoint='api')
# @ app.route('/<query>', methods=['GET'], endpoint='api')
# index / search route
def search(query=''):
    '''
    index / search route
    '''

    # print entry point
    print(request.endpoint)

    # search query
    q = ''

    # items for view
    items = []

    # filters are stored as queries
    queries = {}

    # get query from POST
    if request.method == 'POST':
        q = request.form['search']

        # add filters to tags
        if request.form.get('tags'):
            queries['tags'] = request.form.get('tags')
            show_filter = ''
        if request.form.get('kind'):
            queries['kind'] = request.form.get('kind')
            show_filter = ''
        if request.form.get('ml_libs'):
            queries['ml_libs'] = request.form.get('ml_libs')
            show_filter = ''

    # get query from GET
    api_q_engine = []
    if request.method == 'GET':
        if 'q' in request.args:
            q = request.args.get('q')

        if 'engine' in request.args:
            api_q_engine = request.args.get('engine').split(' ')
        else:
            api_q_engine = ['kt']

        # add filters to tags
        if 'tags' in request.args:
            queries['tags'] = request.args.get('tags')

        if 'kind' in request.args:
            queries['kind'] = request.args.get('kind')

        if 'ml_libs' in request.args:
            queries['ml_libs'] = request.args.get('ml_libs')

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

        # print(queries)

        # perform keyword search for title
        if request.form.get('kt') or 'kt' in api_q_engine:
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
        if request.form.get('kd') or 'kd' in api_q_engine:
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
        if request.form.get('st') or 'st' in api_q_engine:
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
        if request.form.get('sd') or 'sd' in api_q_engine:
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
    if request.endpoint == 'gui':
        queries['query'] = q
        return render_template('index.html', query=queries, query_engines=query_engines, items=items, show_filter=show_filter)

    if request.endpoint == 'api':
        return Response(json.dumps(items), mimetype='application/json')


# start app
if __name__ == '__main__':
    app.run(debug=True)
