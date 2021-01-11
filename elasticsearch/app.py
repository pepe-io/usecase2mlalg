# imports
from flask import Flask, render_template, request, send_from_directory, Response
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
import json
import sys
import os
import time
from elasticsearch import Elasticsearch
import tensorflow as tf
import tensorflow_hub as hub

# start runtime measure
start = time.time()

# define flask app
app = Flask(__name__)

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

# load gui guide
html_guide = './templates/guide.json'
if os.path.isfile(html_guide):
    with open(html_guide, 'r', encoding='utf-8', errors="ignore") as fp:
        html_guide = fp.read()
        html_guide = json.loads(html_guide)
        # for key in html_guide:
        #     html_guide[key].sort()
        # html_guide = {
        #     'tags': json.dumps(html_guide['tags']),
        #     'kind': json.dumps(html_guide['kind']),
        #     'libs': json.dumps(html_guide['ml_libs']),
        #     'host': json.dumps(html_guide['host']),
        # }
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
        tag_        tag for vector search ('title' or 'summarization')
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
                        # without boosting
                        # "source": "cosineSimilarity(params.query_vector, '"+tag_+"') + 1.0",
                        # with boosting
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
                        # without boosting
                        # "source": "cosineSimilarity(params.query_vector, '"+tag_+"') + 1.0",
                        # with boosting
                        "source": "(cosineSimilarity(params.query_vector, '"+tag_+"') + 1.0) / 2 + ( doc['score'].size() > 0 ? doc['score'].value*" + str(boost) + " : 0 )",
                        "params": {"query_vector": query_vector}
                    }
                }
            }
        }

    # print(json.dumps(b, indent=4))
    res = es.search(index=index, body=b)
    return res


def format_result(res, extra={}):
    '''
    parse formatting for response
    '''
    ret = {}
    ret['id'] = res['_id']
    ret['search_score'] = res['_score']
    ret.update(res['_source'])
    if ret['title'] == '':
        ret['title'] = 'None'
    ret.pop('title_vector')
    ret.pop('description_vector')
    ret.pop('summarization_vector')
    ret.update(extra)

    return ret


def bundle_results(items):
    '''
    bundle query-results

    stick identical result (by id) together in a group
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

    # define default embedding
    model = 'use'
    instance = es_indexes[model]

    # define default boosting for social_score
    boosting = 0

    # html-form elements
    html_q_engines = {
        'kt': 'checked',
        'st': 'checked',
        'ks': 'checked',
        'ss': 'checked',
    }
    html_q_secondary = {
        'sum': 'checked',
        'des': '',
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
    q_secondary = 'summarization'

    ### POST / GUI ###
    if request.method == 'POST':
        # get search query
        q = request.form['search']

        # get query engines
        # primary search (title)
        if request.form.get('kt'):
            q_engines.append('kt')

        if request.form.get('st'):
            q_engines.append('st')

        # secondary search (summarization / description)
        if request.form.get('secondary'):
            q_secondary = request.form.get('secondary')
            if q_secondary == 'summarization':
                # switch secondary to summarization
                if request.form.get('ks'):
                    q_engines.append('ks')

                if request.form.get('ss'):
                    q_engines.append('ss')

                html_q_secondary['sum'] = 'checked'
                html_q_secondary['des'] = ''
            else:
                # switch secondary to description
                if request.form.get('ks'):
                    q_engines.append('kd')

                if request.form.get('ss'):
                    q_engines.append('sd')

                # html form
                html_q_secondary['sum'] = ''
                html_q_secondary['des'] = 'checked'

        # get filters
        if request.form.get('category'):
            match['category'] = request.form.get('category')
            show_filter = ''
        if request.form.get('category_not'):
            match['category'] = request.form.get('category_not')
            show_filter = ''

        if request.form.get('subcategory'):
            match_not['subcategory'] = request.form.get('subcategory')
            show_filter = ''
        if request.form.get('subcategory_not'):
            match_not['subcategory'] = request.form.get('subcategory_not')
            show_filter = ''

        if request.form.get('tags'):
            match['tags'] = request.form.get('tags')
            show_filter = ''
        if request.form.get('tags_not'):
            match_not['tags'] = request.form.get('tags_not')
            show_filter = ''

        if request.form.get('kind'):
            match['kind'] = request.form.get('kind')
            show_filter = ''
        if request.form.get('kind_not'):
            match_not['kind'] = request.form.get('kind_not')
            show_filter = ''

        if request.form.get('ml_libs'):
            match['ml_libs'] = request.form.get('ml_libs')
            show_filter = ''
        if request.form.get('ml_libs_not'):
            match_not['ml_libs'] = request.form.get('ml_libs_not')
            show_filter = ''

        if request.form.get('host'):
            match['host'] = request.form.get('host')
            show_filter = ''
        if request.form.get('host_not'):
            match_not['host'] = request.form.get('host_not')
            show_filter = ''

        # switch embedding
        if request.form.get('model'):
            model = request.form.get('model')

        # add boosting
        if request.form.get('boosting') and request.form.get('boosting') == 'true':
            boosting = 10
            html_boosting = {
                'false': '',
                'true': 'checked',
            }

    ### GET / API ###
    if request.method == 'GET':
        # get search query
        if 'q' in request.args:
            q = request.args.get('q')

        # get search engines
        if 'engine' in request.args:
            q_engines = request.args.get('engine').split(' ')
        else:
            q_engines = ['kt']

        # get filters
        if 'category' in request.args:
            match['category'] = request.args.get('category')
        if 'not_category' in request.args:
            match_not['category'] = request.args.get('not_category')

        if 'subcategory' in request.args:
            match['subcategory'] = request.args.get('subcategory')
        if 'not_subcategory' in request.args:
            match_not['subcategory'] = request.args.get('not_subcategory')

        if 'tags' in request.args:
            match['tags'] = request.args.get('tags')
        if 'not_tags' in request.args:
            match_not['tags'] = request.args.get('not_tags')

        if 'kind' in request.args:
            match['kind'] = request.args.get('kind')
        if 'not_kind' in request.args:
            match_not['kind'] = request.args.get('not_kind')

        if 'libs' in request.args:
            match['ml_libs'] = request.args.get('libs')
        if 'not_libs' in request.args:
            match_not['ml_libs'] = request.args.get('not_libs')

        if 'host' in request.args:
            match['host'] = request.args.get('host')
        if 'not_host' in request.args:
            match_not['host'] = request.args.get('not_host')

        # switch embedding
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
        html_embeddings['use'] = 'checked'
        html_embeddings['use_large'] = ''
    # load USE5_large model
    elif model == 'use_large':
        embedding = embed_use_large
        instance = es_indexes[model]
        html_embeddings['use'] = ''
        html_embeddings['use_large'] = 'checked'
    # exit if model is not known
    else:
        print('model not defined')
        sys.exit()

    ### PERFORM QUERY ##
    if q != '':
        i = 0
        print('')
        print('q:', q)
        print('engines:', q_engines)
        print('model:', model,)
        print('instance:', instance)
        print('boosting:', boosting)
        print('')

        # perform keyword primary search (title)
        if 'kt' in q_engines:
            res = keySearch(
                es, dict({'title': q}, **match), index=instance, size=size, must_not=match_not, boost=boosting)
            for hit in res['hits']['hits']:
                items.append(format_result(
                    hit, extra={'index': 'index'+str(i), 'search': 'KT', 'instance': instance, 'embedding': model}))
                i += 1

            html_q_engines['kt'] = 'checked'
        else:
            html_q_engines['kt'] = ''

        # perform keyword secondary search (summarization / description)
        if 'ks' in q_engines or 'kd' in q_engines:
            res = keySearch(es, dict(
                {q_secondary: q}, **match), index=instance, size=size, must_not=match_not, boost=boosting)
            for hit in res['hits']['hits']:
                items.append(format_result(
                    hit, extra={'index': 'index'+str(i), 'search': 'K'+q_secondary[0].upper(), 'instance': instance, 'embedding': model}))
                i += 1

            html_q_engines['ks'] = 'checked'
        else:
            html_q_engines['ks'] = ''

        # perform semantic primary search (title)
        if 'st' in q_engines:
            res = semSearch(
                es, q, 'title_vector', embedding, match, index=instance, size=size, must_not=match_not, boost=boosting)
            for hit in res['hits']['hits']:
                items.append(format_result(
                    hit, extra={'index': 'index'+str(i), 'search': 'ST', 'instance': instance, 'embedding': model}))
                i += 1

            html_q_engines['st'] = 'checked'
        else:
            html_q_engines['st'] = ''

        # perform semantic secondary search (summarization / description)
        if 'sd' in q_engines or 'ss' in q_engines:
            res = semSearch(
                es, q, q_secondary+'_vector', embedding, match, index=instance, size=size, must_not=match_not, boost=boosting)
            for hit in res['hits']['hits']:
                items.append(format_result(
                    hit, extra={'index': 'index'+str(i), 'search': 'S'+q_secondary[0].upper(), 'instance': instance, 'embedding': model}))
                i += 1

            html_q_engines['ss'] = 'checked'
        else:
            html_q_engines['ss'] = ''

    # bundle items (similar matches by engine get grouped together)
    if bundle == True:
        items = bundle_results(items)

    # cut items
    if bundle == False:
        items = items[:int(size)]

    # runtime
    end = time.time()
    dur = round(end-start, 3)
    print('runtime:', dur, 'sec')

    # return view to gui endpoint
    if request.endpoint == 'gui':
        match['query'] = q
        return render_template('index.html', query=match, query_not=match_not, query_engines=html_q_engines, query_secondary=html_q_secondary, boosting=html_boosting, embeddings=html_embeddings, items=items, show_filter=show_filter, runtime=dur, js=html_guide)

    # return json to api endpoint
    if request.endpoint == 'api':
        return Response(json.dumps(items), mimetype='application/json')


# start app
if __name__ == '__main__':
    app.run(debug=True)
