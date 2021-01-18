# imports
from flask import Flask, render_template, request, send_from_directory, Response
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
import json
import math
import os
import sys
import time
from elasticsearch import Elasticsearch
import tensorflow as tf
import tensorflow_hub as hub

# start runtime measure
start = time.time()

# define flask app
app = Flask(__name__)

# es instanc name
es_index = 'usecase2mlalg'

# connect to ES on localhost on port 9200
print('##################################################')
print('Connecting to Elasticsearch...')
es = Elasticsearch([{'host': 'localhost', 'port': 9200}])
if es.ping():
    print('established')
    print('try: http://localhost:9200/'+es_index)
else:
    print('FAILED!')
    sys.exit()
print('##################################################')


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

# define default embedding
model = 'use'
embed = embed_use

# load gui guide
html_guide = './templates/guide.json'
if os.path.isfile(html_guide):
    with open(html_guide, 'r', encoding='utf-8', errors="ignore") as fp:
        html_guide = fp.read()
        html_guide = json.loads(html_guide)
else:
    print('guide not found')
    sys.exit()

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
    m = []
    for tag in must:
        m.append({"match": {tag: must[tag]}})

    if len(m) > 0:
        b.update({
            "must": m
        })

    # match not
    mn = []
    for tag in must_not:
        mn.append({"match": {tag: must_not[tag]}})

    if len(mn) > 0:
        b.update({
            "must_not": mn
        })

    # print('b', b)
    return b


def keySearch(es, must, must_not={}, index=es_index, size=20, boost=0):
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

    print('keybased search')

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
                        "source": "(doc['engagement_score'].size() > 1 ? _score*0.01 * doc['engagement_score'].value*" + str(boost) + " : _score*0.01) / " + str(boost)
                    }
                }
            }
        }
    }
    # print(json.dumps(b, indent=2))
    res = es.search(index=index, body=b)
    return res


def semSearch(es, query, tags, embedding, must, must_not={}, index=es_index, size=20, boost=10, boost_ml=0, boost_eng=0):
    '''
    Search by Vec Similarity

    @args:
        es          Elasticsearch instance
        query        search query term
        tags        tag for vector search ('title' or 'summarization')
        must        dict of must-match-rules of format {field: value}
        must_not    dict of must_not-match-rules of format {field: value}
        index       name of ES instance
        size        number of records returned
        boost       boost multiplier for score (aka social_score from database)

    @return:
        results
    '''

    print('semantic search')

    # get query vector from embedding
    query_vector = tf.make_ndarray(
        tf.make_tensor_proto(embedding([query]))).tolist()[0]

    # parse boolean query
    b = parse_es_bool_query(must, must_not)
    # print('boolean-query', b)

    # parse scoring script
    # without boosting
    # s = "cosineSimilarity(params.query_vector, '"+tag_+"') + 1.0 / 2"
    s = []
    for t in tags:
        s.append("(cosineSimilarity(params.query_vector, '"+t+"') + 1.0 / 2)")
    s = ' * '.join(s)

    # with boosting
    # s = "(cosineSimilarity(params.query_vector, '"+tag_+"') + 1.0) / 2 + ( doc['engagement_score'].size() > 0 ? doc['engagement_score'].value*" + str(boost) + " : 0 )",
    s = s + \
        " * ( doc['engagement_score'].size() > 0 ? doc['engagement_score'].value*" + \
        str(boost) + " : 1 ) / " + str(boost)

    # print(s)

    if len(b) == 0:
        b = {
            "size": size,
            "query": {
                "script_score": {
                    "query": {
                        "match_all": {}
                    },
                    "script": {
                        # without boosting
                        # "source": "cosineSimilarity(params.query_vector, '"+tag_+"') + 1.0",
                        # "source": "cosineSimilarity(params.query_vector, 'title_vector_use4') * cosineSimilarity(params.query_vector, 'summarization_vector_use4') * cosineSimilarity(params.query_vector, 'fulltext_vector_use4') + 1.0 / 2",
                        # with boosting
                        # "source": "(cosineSimilarity(params.query_vector, '"+tag_+"') + 1.0) / 2 + ( doc['engagement_score'].size() > 0 ? doc['engagement_score'].value*" + str(boost) + " : 0 )",

                        "source": s,
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
                        "source": s,
                        "params": {"query_vector": query_vector}
                    }
                }
            }
        }

    # print(json.dumps(b, indent=4))
    res = es.search(index=index, body=b)
    return res


def sigmoid(z):
    # return 1/(1 + np.exp(-z))
    return 1 / (1 + math.exp(-x))


def format_result(res, extra={}, res_filter=[]):
    '''
    parse formatting for response
    '''
    ret = {}
    # rename some elasticsearch variables
    ret['id'] = res['_id']
    ret['search_score'] = round(res['_score'], 3)
    ret.update(res['_source'])
    # provide a title if missing
    if ret['title'] == '':
        ret['title'] = 'None'
    # delete vectors from response
    pop = ['title_vector_use4', 'title_vector_use5', 'summarization_vector_use4',
           'summarization_vector_use5', 'fulltext_vector_use4', 'fulltext_vector_use5']
    for p in pop:
        if p in ret:
            ret.pop(p)
    # add extras
    ret.update(extra)

    return ret


def bundle_results(items):
    '''
    bundle query-results

    combine identical result (by id) together in a group
    because different engines can get the same result
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
            # print(i, item['id'], ret[i]['id'])
            ret[i]['search'] = '*' + ret[i]['search'] + '/' + item['search']
            ret[i]['search_score'] += item['search_score']
            ret[i]['search_score'] = round(ret[i]['search_score'], 3)

    return ret


def filter_response(items, res_filter):
    '''
    filter response
    return only params requested in 'filter'
    '''
    ret = []
    if len(res_filter) > 0:
        for item in items:
            ret.append({key: item[key] for key in res_filter if key in item})
    print(ret)
    return ret


@app.route('/favicon.ico')
def favicon():
    '''
    serve favicon
    '''
    return send_from_directory(os.path.join(app.root_path, 'static'), 'favicon.ico', mimetype='image/vnd.microsoft.icon')


# index / search route
@ app.route('/', methods=['GET', 'POST'], endpoint='gui')
@ app.route('/api', methods=['GET'], endpoint='api')
def search(query=''):
    '''
    index / search route
    gui & api access
    '''

    # runtime
    start = time.time()

    # print entry point
    print('route:', request.endpoint)

    # search query
    q = ''

    # number of records
    size = 20

    # items for view
    items = []

    # bundle items
    bundle = True

    # filters & queries
    match = {}
    match_not = {}

    # define default boosting for engagement_score
    boosting = 1

    # filter response
    res_filter = []

    # html-form elements
    html_q_engines = {
        'k': 'checked',
        's': 'checked',
    }
    html_q_secondary = {
        'f': 'checked',
        't': '',
        's': '',
    }
    html_embeddings = {
        'use': 'checked',
        'use_large': '',
    }
    html_boosting = {
        'false': 'checked',
        'true': '',
    }

    # css-attribute to hide filters (default value)
    show_filter = 'hidden'

    # set default values
    q = ''
    q_engines = []
    q_secondary = 'f'

    ### POST / GUI ###
    if request.method == 'POST':
        r = request.form

    ### GET / API ###
    if request.method == 'GET':
        r = request.args

    print(dict(r))

    # PARSE REQUEST
    # get search query
    q = r.get('search') if r.get('search') else ''

    # get query engines
    # primary search (title)
    if r.get('k'):
        q_engines.append('k')
        html_q_engines['k'] = 'checked'
    else:
        html_q_engines['k'] = ''

    if r.get('s'):
        q_engines.append('s')
        html_q_engines['s'] = 'checked'
    else:
        html_q_engines['s'] = ''

    # field (title / summarization / fulltext)
    if r.get('field'):
        q_field = r.get('field')
        html_q_secondary['t'] = ''
        html_q_secondary['s'] = ''
        html_q_secondary['f'] = ''
        html_q_secondary[r.get('field')[0]] = 'checked'

    # word based search
    # if r.get('solo'):
    #     solo = True
    # else:
    #     solo = False
    solo = False

    if solo == True:
        q = q.split(' ')
    else:
        q = [q]

    # get search engines (api)
    if 'engine' in r:
        q_engines = r.get('engine').split(' ')

    # get filters
    if r.get('category'):
        match['category'] = r.get('category')
        show_filter = ''
    if r.get('category_not'):
        match['category'] = r.get('category_not')
        show_filter = ''

    if r.get('subcategory'):
        match_not['subcategory'] = r.get('subcategory')
        show_filter = ''
    if r.get('subcategory_not'):
        match_not['subcategory'] = r.get('subcategory_not')
        show_filter = ''

    if r.get('tags'):
        match['tags'] = r.get('tags')
        show_filter = ''
    if r.get('tags_not'):
        match_not['tags'] = r.get('tags_not')
        show_filter = ''

    if r.get('kind'):
        match['kind'] = r.get('kind')
        show_filter = ''
    if r.get('kind_not'):
        match_not['kind'] = r.get('kind_not')
        show_filter = ''

    if r.get('ml_libs'):
        match['ml_libs'] = r.get('ml_libs')
        show_filter = ''
    if r.get('ml_libs_not'):
        match_not['ml_libs'] = r.get('ml_libs_not')
        show_filter = ''

    if r.get('host'):
        match['host'] = r.get('host')
        show_filter = ''
    if r.get('host_not'):
        match_not['host'] = r.get('host_not')
        show_filter = ''

    # switch embedding
    if r.get('model'):
        model = r.get('model')

    # add boosting
    if r.get('boosting'):
        # gui
        if r.get('boosting') == 'true':
            boosting = 10
            html_boosting = {
                'false': '',
                'true': 'checked',
            }
        # api
        elif isinstance(r.get('boosting'), int):
            boosting = r.get('boosting')

    # apply size
    if 'size' in r:
        size = r.get('size')

    # apply bundling
    if 'bundle' in r:
        bundle = r.get('bundle')

    # response filter
    if 'filter' in r:
        res_filter = r.get('filter').split(' ')

    # assing embedding
    vec = 'use4'
    # load USE4 model
    if model == 'use':
        embedding = embed_use
        vec = 'use4'
        html_embeddings['use'] = 'checked'
        html_embeddings['use_large'] = ''
    # load USE5_large model
    elif model == 'use_large':
        embedding = embed_use_large
        vec = 'use5'
        html_embeddings['use'] = ''
        html_embeddings['use_large'] = 'checked'
    # exit if model is not known
    else:
        print('model not defined')
        sys.exit()

    ### PERFORM QUERY ##
    if q != '':
        i = 0
        print('###')
        print('search:', q)
        print('engines:', q_engines)
        print('field:', q_field)
        print('model:', model,)
        print('instance:', es_index)
        print('boosting:', boosting)
        print('---')

        # perform keyword based search
        if 'k' in q_engines:
            query = ' '.join(q)
            tags = {q_field: query}
            if q_field == 'fulltext':
                tags = {'title': query,
                        'summarization': query, 'fulltext': query}

            tags.update(match)
            res = keySearch(
                es, tags, size=size, must_not=match_not, boost=boosting)
            for hit in res['hits']['hits']:
                items.append(format_result(
                    hit, extra={'index': 'index'+str(i), 'search': 'k'+q_field[0], 'instance': es_index, 'embedding': model}))
                i += 1

        # perform semantic search
        if 's' in q_engines:
            tags = [q_field+'_vector_'+vec]
            if q_field == 'fulltext':
                tags = ['title_vector_'+vec, 'summarization_vector_' +
                        vec, 'fulltext_vector_'+vec]
            for query in q:
                res = semSearch(
                    es, query, tags, embedding, match, size=size, must_not=match_not, boost=boosting)
                for hit in res['hits']['hits']:
                    items.append(format_result(
                        hit, extra={'index': 'index'+str(i), 'search': 's'+q_field[0], 'instance': es_index, 'embedding': model}))
                    i += 1

    # bundle items (similar matches by engine get grouped together)
    if bundle == True:
        items = bundle_results(items)

    # cut items
    if bundle == False:
        items = items[:int(size)]

    # filter response
    if len(res_filter) > 0:
        items = filter_response(items, res_filter)

    # runtime
    end = time.time()
    dur = round(end-start, 3)
    print('runtime:', dur, 'sec')

    # if len(items) > 0:
    #     print(items[0])
    # else:
    #     print('nothing found')

    # return view to gui endpoint
    if request.endpoint == 'gui':
        match['query'] = ' '.join(q)
        return render_template('index.html', query=match, query_not=match_not, query_engines=html_q_engines, query_secondary=html_q_secondary, boosting=html_boosting, embeddings=html_embeddings, items=items, show_filter=show_filter, runtime=dur, js=html_guide)

    # return json to api endpoint
    if request.endpoint == 'api':
        return Response(json.dumps(items), mimetype='application/json')


# start app
if __name__ == '__main__':
    app.run(debug=True)
