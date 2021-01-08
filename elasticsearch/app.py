# imports
from flask import Flask, render_template, request, send_from_directory, Response
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
import json
import sys
import os
import time
from elasticsearch import Elasticsearch
# from elasticsearch.helpers import bulk
import tensorflow as tf
import tensorflow_hub as hub

# start runtime measure
start = time.time()

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
    print('or:  http://localhost:9200/usecase2mlalg_large')
else:
    print('FAILED!')
    sys.exit()
print('##################################################')


# es instance names
es_indexes = {
    'use': 'usecase2mlalg',
    'use_large': 'usecase2mlalg_large'
}

# load embedding
# load USE4 model
print('load USE4 embedding')
use4_start = time.time()
embed_use = hub.load("./.USE4/")
use4_end = time.time()
print('loaded ('+str(round(use4_end-use4_start, 3))+'sec)')
print('##################################################')

# load USE5_large model
print('load USE5_large embedding')
use5_start = time.time()
embed_use_large = hub.load("./.USE5_large/")
use5_end = time.time()
print('loaded ('+str(round(use5_end-use5_start, 3))+'sec)')
print('##################################################')

# set default embedding
embed = embed_use

# print runtime measure
end = time.time()
print('READY (boot took: '+str(round(end-start, 3))+'sec)')
print('##################################################')


def parse_es_bool_query(must, must_not):
    '''
    parse Elasticsearch boolean query
    '''
    # bool query
    b = {}

    # match
    m = {}
    for tag in must:
        m.update({"match": {tag: must[tag]}})

        # # exists
        # m.update({
        #     "exists": {
        #         "field": "category"
        #     }
        # })

    if len(m) > 0:
        b.update({
            "must": [
                m
            ]
        })

    # match not
    mn = {}
    for tag in must_not:
        mn.update({"match": {tag: must_not[tag]}})

    if len(mn) > 0:
        b.update({
            "must_not": [
                mn
            ]
        })

    # # exists
    # b.update({
    #     "exists": {
    #         "field": "category"
    #     }
    # })

    return b


def keySearch(es, must, must_not={}, index='usecase2mlalg', size=20, boost=0):
    '''
    Search by Keyword, td-idf

    @args:
        es          Elasticsearch instance
        must        dict of must-match-rules of format {field: value}
        must_not    dict of must_not-match-rules of format {field: value}
        index       name of ES instance
        size        number of records returned
        boost       boost multiplier for score (aka social_score from database)

    @return:
        results
    '''

    # parse boolean query
    b = parse_es_bool_query(must, must_not)

    # ES body
    b = {
        "size": size,
        "query": {
            "function_score": {
                "query": {
                    "bool": b
                },
                "script_score": {
                    "script": {
                        "source": "doc['score'].size() > 0 ? _score + doc['score'].value*" + str(boost) + " : _score"
                    }
                }
            }
        }
    }
    # print(json.dumps(b, indent=2))
    res = es.search(index=index, body=b)
    return res


def semSearch(es, sent, tag_, embedding, must, must_not={}, index='usecase2mlalg', size=20, boost=0):
    '''
    Search by Vec Similarity

    @args:
        es          Elasticsearch instance
        sent        search query term
        tag_        tag for vector search ('title' or 'description')
        must        dict of must-match-rules of format {field: value}
        must_not    dict of must_not-match-rules of format {field: value}
        index       name of ES instance
        size        number of records returned
        boost       boost multiplier for score (aka social_score from database)

    @return:
        results
    '''

    # get query vector from embedding
    query_vector = tf.make_ndarray(
        tf.make_tensor_proto(embedding([sent]))).tolist()[0]

    # parse boolean query
    b = parse_es_bool_query(must, must_not)
    # print('boolean-query', b)

    if len(b) == 0:
        b = {
            "size": size,
            "query": {
                "script_score": {
                    "query": {
                        "match_all": {}
                    },
                    "script": {
                        # 'source': 'cosineSimilarity(params.query_vector, "'+tag_+'") + 1.0', # without boosting
                        "source": "(cosineSimilarity(params.query_vector, '"+tag_+"') + 1.0) / 2 + ( doc['score'].size() > 0 ? doc['score'].value*" + str(boost) + " : 0 )",
                        "params": {"query_vector": query_vector}
                    }
                }
            }
        }
    else:
        b = {
            "size": size,
            "query": {
                "script_score": {
                    "query": {
                        "bool": b
                    },
                    "script": {
                        "source": "(cosineSimilarity(params.query_vector, '"+tag_+"') + 1.0) / 2 + ( doc['score'].size() > 0 ? doc['score'].value*" + str(boost) + " : 0 )",
                        "params": {"query_vector": query_vector}
                    }
                }
            }
        }

    #print(json.dumps(b, indent=4))
    res = es.search(index=index, body=b)
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
    gui & api access
    '''

    # print entry point
    print(request.endpoint)

    # search query
    q = ''

    # number of records
    size = 20

    # items for view
    items = []

    # bundle items
    bundle = True

    # filters are stored as queries
    queries = {}
    match_not = {}

    # define default embedding
    model = 'use'
    instance = es_indexes[model]

    # define default boosting for social_score
    boosting = 0

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
        if 'not_tags' in request.args:
            match_not['tags'] = request.args.get('not_tags')

        if 'kind' in request.args:
            queries['kind'] = request.args.get('kind')
        if 'not_kind' in request.args:
            match_not['kind'] = request.args.get('not_kind')

        if 'libs' in request.args:
            queries['ml_libs'] = request.args.get('libs')
        if 'not_libs' in request.args:
            match_not['ml_libs'] = request.args.get('not_libs')

        # switch model
        if 'model' in request.args:
            model = request.args.get('model')

        # add boosting
        if 'boosting' in request.args:
            boosting = request.args.get('boosting')

        # apply size
        if 'size' in request.args:
            size = request.args.get('size')

        # apply bundling
        if 'bundle' in request.args:
            bundle = request.args.get('bundle')

    # assing embedding
    # load USE4 model
    if model == 'use':
        embedding = embed_use
        instance = es_indexes[model]
    # load USE5_large model
    elif model == 'use_large':
        embedding = embed_use_large
        instance = es_indexes[model]
    # exit if model is not known
    else:
        print('model not defined')
        sys.exit()

    print('model:', model, ', instance:', instance)

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
        print('q:', q)

        # print(queries)

        # perform keyword search for title
        if request.form.get('kt') or 'kt' in api_q_engine:
            query_engines['kt'] = ' checked'
            # res_kw = keywordSearch(es, q, 'title')
            res_kw = keySearch(
                es, dict({'title': q}, **queries), index=instance, size=size, must_not=match_not, boost=boosting)
            for hit in res_kw['hits']['hits']:
                items.append(format_result(
                    hit, extra={'index': 'index'+str(i), 'search': 'KT', 'instance': instance, 'embedding': model}))
                i += 1
        else:
            query_engines['kt'] = ''

        # perform keyword search for description
        if request.form.get('kd') or 'kd' in api_q_engine:
            query_engines['kd'] = ' checked'
            # res_kw = keywordSearch(es, q, 'description')
            res_kw = keySearch(es, dict(
                {'title': q}, **queries), index=instance, size=size, must_not=match_not, boost=boosting)
            for hit in res_kw['hits']['hits']:
                items.append(format_result(
                    hit, extra={'index': 'index'+str(i), 'search': 'KD', 'instance': instance, 'embedding': model}))
                i += 1
        else:
            query_engines['kd'] = ''

        # perform semantic search for title
        if request.form.get('st') or 'st' in api_q_engine:
            query_engines['st'] = ' checked'
            # res_semantic = sentenceSimilaritybyNN(es, q, 'title_vector')
            res_semantic = semSearch(
                es, q, 'title_vector', embedding, queries, index=instance, size=size, must_not=match_not, boost=boosting)
            for hit in res_semantic['hits']['hits']:
                items.append(format_result(
                    hit, extra={'index': 'index'+str(i), 'search': 'ST', 'instance': instance, 'embedding': model}))
                i += 1
        else:
            query_engines['st'] = ''

        # perform semantic search for description
        if request.form.get('sd') or 'sd' in api_q_engine:
            query_engines['sd'] = ' checked'
            #res_semantic = sentenceSimilaritybyNN(es, q, 'description_vector')
            res_semantic = semSearch(
                es, q, 'description_vector', embedding, queries, index=instance, size=size, must_not=match_not, boost=boosting)
            for hit in res_semantic['hits']['hits']:
                items.append(format_result(
                    hit, extra={'index': 'index'+str(i), 'search': 'SD', 'instance': instance, 'embedding': model}))
                i += 1
        else:
            query_engines['sd'] = ''

    # bundle items (similar matches by engine get grouped together)
    if bundle == True:
        items = bundle_results(items)

    # cut items
    if bundle == False:
        items = items[:int(size)]

    # return view to gui endpoint
    if request.endpoint == 'gui':
        queries['query'] = q
        return render_template('index.html', query=queries, query_engines=query_engines, items=items, show_filter=show_filter)

    # return json to api endpoint
    if request.endpoint == 'api':
        return Response(json.dumps(items), mimetype='application/json')


# start app
if __name__ == '__main__':
    app.run(debug=True)
