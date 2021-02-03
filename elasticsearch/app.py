# imports
from flask import Flask, render_template, request, send_from_directory, Response
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
from collections import Counter
import json
import math
import os
import sys
import time
from elasticsearch import Elasticsearch
import tensorflow as tf
import tensorflow_hub as hub
import nltk
import re
import pandas as pd
import unicodedata
from nltk.corpus import stopwords

# start runtime measure
start = time.time()

# define flask app
app = Flask(__name__)

# es instanc name
es_index = 'usecase2ml'

# options & debug
options = {
    'use4': True,
    'use5': True,
    'default_embedding': 'use5',
    'search_engines': {
        'keyword_default': True,
        'keyword_boolean': True,
        'semantic': True,
    },
    'max_results': 20,
    'score_treshold': 0.1,
    'bundle_items': True,
    'cutoff_items': False,
    'aggregations': True,
    'ngrams': True,
    'scale_search_score': True,
    'max_items_aggregations': 10,
}

debug = {
    'print_keysearch_query': False,
    'print_keysearch_result': False,
    'print_keysearch_aggregation': False,
    'print_semantic_query': False,
    'print_semantic_result': False,
    'print_semantic_aggregation': False,
    'print_request_args': True,
    'print_first_record': False,
    'print_aggregations': False,
    'print_filter_response': False,
}

debug_view = {
    'print_fulltext': False,
}

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
if options['use4']:
    print('load USE4 embedding')
    use4_start = time.time()
    embed_use = hub.load("./.USE4/")
    use4_end = time.time()
    print('loaded ('+str(round(use4_end-use4_start, 3))+'sec)')
    print('##################################################')

# load USE5_large model
if options['use5']:
    print('load USE5_large embedding')
    use5_start = time.time()
    embed_use_large = hub.load("./.USE5_large/")
    use5_end = time.time()
    print('loaded ('+str(round(use5_end-use5_start, 3))+'sec)')
    print('##################################################')

# define default embedding
embed = embed_use

# load gui guide
guide = './templates/guide.json'
if os.path.isfile(guide):
    with open(guide, 'r', encoding='utf-8', errors="ignore") as fp:
        guide = fp.read()
        guide = json.loads(guide)
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


def parse_es_search_score(s_func, boost={}, scale=1):
    '''
    parse Elasticsearch boolean query
    '''

    ### the formula to calculate the search score ###
    # calculate the category score
    # sum up category scores to apply filtering by category
    # scale resulting TD*IDF score to fit in range 0...1
    # s...score, b...boost
    'c = [(s * s_cat1 * b_cat1) + (s * s_cat2 * b_cat2) + ... + (s * s_cat_n * b_cat_n)] / n / scale'

    # calculate boosting scores
    'b = [(s_b1 * b1) + (s_b2 * b2) + ... + (s_b_n * b_n)] / n'

    # the final weighted score
    's_final = s_cat * 0.75 + s_boost * 0.25'

    # let's parse it into a string
    c = []
    b = []
    for k, v in boost.items():
        # category score
        if not 'boost_' in k:
            c.append("{score_search} * doc['{field}'].value * {boost}".format(
                score_search=s_func, field=k, boost=v))
        # boosting score
        else:
            k = k.replace('boost_', '')
            fallback = 0
            b.append("doc['{field}'].size() > 1 ? doc['{field}'].value * {boost} : {false}".format(
                field=k, boost=v, false=fallback))

    # finalize category score
    score_c = ' + '.join(c)
    # catch no category filtering applied
    if len(c) > 0:
        score_c = "({c}) / {n} / {scale}".format(c=score_c,
                                                 n=len(c), scale=scale)
    # return default score
    else:
        score_c = "{score_search} / {scale}".format(
            score_search=s_func, scale=scale)

    # finalize boosting score
    score_b = ' + '.join(b)
    # catch no boosting applied
    if len(b) > 0:
        score_b = "({b}) / {n}".format(b=score_b, n=len(b))
    # return default boosting
    else:
        score_b = None

    # calculate final score
    if score_b != None:
        score = "{score_c} * 0.75 + {score_b} * 0.25".format(
            score_c=score_c, score_b=score_b)
    else:
        score = score_c

    # print(score)
    return score


def get_es_aggregations():
    '''
    return aggregations dict for query
    '''
    return {
        "categories": {
            "terms": {"field": "category.keyword"}
        },
        "subcategories": {
            "terms": {"field": "subcategory.keyword"}
        },
        "tags": {
            "terms": {"field": "tags.keyword"}
        },
        "libs": {
            "terms": {"field": "ml_libs.keyword"}
        },
        "sources": {
            "terms": {"field": "host.keyword"}
        },
        "kinds": {
            "terms": {"field": "kind.keyword"}
        },
        "licenses": {
            "terms": {"field": "license.keyword"}
        },
        "languages": {
            "terms": {"field": "programming_language.keyword"}
        }
    }


def parse_es_aggregations(aggs_raw):
    '''
    parse aggregations from elasticsearch
    return a cleaned up dict
    '''
    aggs = {}
    for k, v in aggs_raw.items():
        agg = {
            k: {b['key']: b['doc_count'] for b in v['buckets']}
        }
        aggs.update(agg)
    return aggs


def basic_clean(text):
    """
    A simple function to clean up the data. All the words that
    are not designated as a stop word is then lemmatized after
    encoding and basic regex parsing are performed.
    """
    wnl = nltk.stem.WordNetLemmatizer()
    stopwords = nltk.corpus.stopwords.words('english')
    text = (unicodedata.normalize('NFKD', text)
            .encode('ascii', 'ignore')
            .decode('utf-8', 'ignore')
            .lower())
    words = re.sub(r'[^\w\s]', '', text).split()
    return [wnl.lemmatize(word) for word in words if word not in stopwords]


def parse_aggregations(items):
    if options['aggregations']:
        aggs = {
            # 'agg_name': 'field_name',
            'categories': 'category',
            'subcategories': 'subcategory',
            'tags': 'tags',
            'libs': 'ml_libs',
            'sources': 'host',
            'kinds': 'kind',
            'licenses': 'license',
            'languages': 'programming_language',
        }
        a = {
            'categories': [],
            'subcategories': [],
            'tags': [],
            'libs': [],
            'sources': [],
            'kinds': [],
            'licenses': [],
            'languages': [],
        }
    else:
        a = {}

    words = ''
    for i in items:
        if options['ngrams']:
            words += ' ' + i['_source']['summarization']
        if options['aggregations']:
            for k, v in aggs.items():
                if v in i['_source']:
                    j = i['_source'][v]
                    if isinstance(j, list):
                        a[k].extend(j)
                    elif isinstance(j, str):
                        a[k].append(j)

    if options['aggregations']:
        for k, v in a.items():
            # sort
            a[k] = dict(sorted(dict(Counter(v)).items(),
                               key=lambda item: item[1], reverse=True))
            # restrict count
            a[k] = dict(list(a[k].items())[:options['max_items_aggregations']])

    # n-grams
    if options['ngrams']:
        words = basic_clean(''.join(str(words)))
        names = ['unigram', 'bigram', 'trigram']
        for i in range(1, 4):
            n = nltk.ngrams(words, i)
            # sort
            n = dict(sorted(dict(Counter(n)).items(),
                            key=lambda item: item[1], reverse=True))
            # restrict count
            n = dict(list(n.items())[:options['max_items_aggregations']])
            # join
            n = {' '.join(x): y for x, y in n.items()}
            # store
            a[names[i-1]] = n

    return a


def bundle_aggregations(a, b):
    '''
    bundle aggregation from keysearch and semantic search
    '''

    # if a empty return b
    if len(a) == 0:
        return b

    c = dict(a)
    # combine aggregation
    for k, v in b.items():
        for kk, vv in v.items():
            # store maximum
            if kk in c[k]:
                c[k][kk] = max(vv, c[k][kk])
            # add item
            else:
                c[k][kk] = vv

    # sort aggregations
    for k, v in c.items():
        # sort
        c[k] = dict(sorted(v.items(), key=lambda item: item[1], reverse=True))
        # restrict count
        c[k] = dict(list(c[k].items())[:options['max_items_aggregations']])

    return c


def keySearch(es, must, must_not={}, index=es_index, size=options['max_results'], boost={}, scale=False):
    '''
    Search by Keyword, td-idf

    @args:
        es          Elasticsearch instance
        must        dict of must-match-rules of format {field: value}
        must_not    dict of must_not-match-rules of format {field: value}
        index       name of ES instance
        size        number of records returned
        boost       boost multipliers for score

    @return:
        results
    '''

    print('keybased search')

    # parse boolean query
    b = parse_es_bool_query(must, must_not)

    # parse scoring script
    # use a placeholder, which will be replaced later, to not rebuild the whole query
    placeholder = '#SCALE#'
    s = parse_es_search_score("_score*0.01", boost=boost, scale=placeholder)

    # get aggregations
    # a = get_es_aggregations()

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
                        "source": s
                    }
                }
            }
        }
        # ,
        # "aggs": a
    }

    # print query sructure
    if debug['print_keysearch_query']:
        print('keysearch query:', json.dumps(b, indent=2))

    # replace the placeholder with 1
    c = json.dumps(b)
    c = c.replace(placeholder, str(1))
    c = json.loads(c)

    # execute query
    res = es.search(index=index, body=c)

    # scale TD*IDF score: maximum to 1
    # we have to do the search a second time
    # to apply scaling before boosting
    # only do this, if we have results
    if len(res['hits']['hits']) > 0 and scale == True:
        scale = res['hits']['hits'][0]['_score']

        # print result (zeroize vectors)
        if debug['print_keysearch_result']:
            s = res['hits']['hits'][0]
            s['_source'] = {
                k: v if not 'vector' in k else [] for k, v in s['_source'].items()}
            print('keysearch result', json.dumps(s, indent=2))

        # replace the placeholder with scaling factor
        c = json.dumps(b)
        c = c.replace(placeholder, str(scale))
        c = json.loads(c)

        # execute query (apply rescaling)
        res = es.search(index=index, body=c)

    res['aggregations'] = parse_aggregations(res['hits']['hits'])
    if debug['print_keysearch_aggregation']:
        print('keysearch aggregation:', json.dumps(
            res['aggregations'], indent=2))

    return res


def semSearch(es, query, tags, embedding, must, must_not={}, index=es_index, size=options['max_results'], boost={}):
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

    # print result (zeroize vectors)
    if debug['print_semantic_query']:
        s = str(b)
        s['_source'] = {
            k: v if not 'vector' in k else [] for k, v in s['_source'].items()}
        print(json.dumps(s, indent=2))

    if debug['print_semantic_query']:
        print('semantic query:', json.dumps(b, indent=2))

    # parse scoring script
    s = []
    for t in tags:
        s.append(
            "(cosineSimilarity(params.query_vector, '{tag}') + 1.0) / 2".format(tag=t))
    s_func = ' + '.join(s)
    s_func = "({s_func}) / {n}".format(s_func=s_func, n=len(s))

    s = parse_es_search_score(s_func, boost=boost)

    # get aggregations
    # a = get_es_aggregations()

    if len(b) == 0:
        b = {
            "size": size,
            "query": {
                "script_score": {
                    "query": {
                        "match_all": {}
                    },
                    "script": {
                        "source": s,
                        "params": {"query_vector": query_vector}
                    }
                }
            }
            # ,
            # "aggs": a
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
            # ,
            # "aggs": a
        }

    # print query
    if debug['print_semantic_query']:
        print(json.dumps(b, indent=4))

    # execute query
    res = es.search(index=index, body=b)

    res['aggregations'] = parse_aggregations(res['hits']['hits'])
    # print aggregations
    if debug['print_semantic_aggregation']:
        print('semantic aggregation:', json.dumps(
            res['aggregations'], indent=2))

    return res


def format_result(res, query='', field='', scale=1, extra={}, res_filter=[], treshold=None, mode='gui'):
    '''
    parse formatting for response
    '''
    ret = {}

    # rename some elasticsearch variables
    ret['id'] = res['_id']
    ret['search_score'] = res['_score']
    ret.update(res['_source'])

    # provide a title if missing
    ret['title'] = 'None' if ret['title'] == '' else ret['title']

    # delete vectors from response
    ret = {k: v if not 'vector' in k else None for k, v in ret.items()}

    # bundle scores
    scores = ['learn_score', 'explore_score',
              'compete_score', 'ml_score', 'engagement_score']
    ret['scores'] = {x: ret[x] for x in scores if x in ret and ret[x] > 0}
    # format score for gui
    if mode == 'gui':
        ret['scores'] = ' | '.join(['{k}: <b>{v}</b>'.format(k=k.replace('_', ' '), v=v)
                                    for k, v in ret['scores'].items()])

    # get type
    types = ['learn_score', 'explore_score', 'compete_score']
    ret['type'] = [x.replace('_score', '').title()
                   for x in types if ret[x] > 0]

    # detect missing query items & recalculate
    ret['missing'] = []
    q_terms = query.split(' ')
    for q in q_terms:
        if ret[field].lower().find(q.lower()) == -1:
            ret['missing'].append(q)
    ret['search_score'] = round((len(q_terms) - len(ret['missing'])
                                 ) / len(q_terms) * ret['search_score'] / scale, 3)

    # add extras
    ret.update(extra)

    # cutoff results by treshold
    if treshold == None or ret['search_score'] >= treshold:
        return ret
    else:
        return None


def bundle_results(items):
    '''
    bundle query-results

    combine identical results in a group, by id
    because different engines can get the same records
    leading to duplications
    '''
    ret = []
    ids = []

    for item in items:
        if not item['id'] in ids:
            ids.append(item['id'])
            ret.append(item)
        else:
            i = next((i for i, key in enumerate(ret)
                      if key['id'] == item['id']), None)
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
    if debug['print_filter_response']:
        print(ret)
    return ret


@ app.route('/favicon.ico')
def favicon():
    '''
    serve favicon
    '''
    return send_from_directory(os.path.join(app.root_path, 'static'), 'favicon.ico', mimetype='image/vnd.microsoft.icon')


# index / search route
@ app.route('/', methods=['GET', 'POST'], endpoint='gui')
@ app.route('/api', methods=['GET'], endpoint='api')
def search(query='', options=options, guide=guide):
    '''
    index / search route
    gui & api access
    '''

    # runtime
    start = time.time()

    # print entry point
    print('route:', request.endpoint)

    # number of records
    size = options['max_results']

    # cut off item under score n
    treshold = options['score_treshold']

    # items for view
    items = []

    # bundle items
    bundle = options['bundle_items']

    # filters & queries
    match = {}
    match_not = {}

    # define default score boostings
    boosting = {
        # 'learn_score': 1,
        # 'explore_score': 1,
        # 'compete_score': 1,
        # 'boost_engagement_score': 1,
    }
    boost_engagement_score_factor = 3

    # filter response
    res_filter = []

    # html-form elements
    html_mode = {
        'all': 'checked',
        'learn': '',
        'explore': '',
        'compete': '',
    }
    html_q_secondary = {
        'f': 'checked',
        't': '',
        's': '',
    }
    html_embeddings = {
        'use4': '',
        'use5': '',
    }
    html_embeddings[options['default_embedding']] = 'checked'

    # set default values
    q = ''
    q_field = 'fulltext'
    q_engines = ['k', 'b', 's']
    q_secondary = q_field[0]
    aggregations = {}
    aggregations_raw = []
    aggs_checked = ['categories', 'tags', 'libs']

    ### POST / GUI ###
    if request.endpoint == 'gui':
        r = request.form

        # get query engines
        if 'engine_k' in r or 'engine_b' in r or 'engine_s' in r:
            q_engines = []
        if 'engine_k' in r:
            q_engines.append('k')
        if 'engine_b' in r:
            q_engines.append('b')
        if 'engine_s' in r:
            q_engines.append('s')

    ### GET / API ###
    if request.endpoint == 'api':
        r = dict(request.args)
        # set some default values if missing
        if not 'mode' in r:
            r['mode'] = 'all'
        if not 'model' in r:
            r['model'] = 'use5'
        if not 'engine' in r:
            r['engine'] = 'k b s'

        # get query engines
        q_engines = r.get('engine').split(' ')

    # print request args
    if debug['print_request_args']:
        print(dict(r))

    # PARSE REQUEST
    if len(r) > 0:
        # get search query
        q = r.get('search').strip() if r.get('search') else ''

        # field (title / summarization / fulltext)
        if r.get('field'):
            q_field = r.get('field')
            html_q_secondary['t'] = ''
            html_q_secondary['s'] = ''
            html_q_secondary['f'] = ''
            html_q_secondary[r.get('field')[0]] = 'checked'

        # get charts
        if 'charts' in r:
            aggs_checked = r['charts'].split(',')

        # get search engines (api)
        if 'engine' in r and request.method == 'GET':
            q_engines = r.get('engine').split(' ')

        # get filters
        if r.get('category'):
            match['category'] = r.get('category')
            show_filter = ''
        if r.get('category_not'):
            match_not['category'] = r.get('category_not')
            show_filter = ''

        if r.get('subcategory'):
            match['subcategory'] = r.get('subcategory')
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
            # api
            if isinstance(r.get('boosting'), int):
                boost_engagement_score_factor = r.get('boosting')

        # add mode
        if r['mode'] == 'learn':
            boosting = {
                'learn_score': 1,
                'explore_score': 0,
                'compete_score': 0,
                'boost_engagement_score': boost_engagement_score_factor,
            }
            html_mode = {
                'all': '',
                'learn': 'checked',
                'explore': '',
                'compete': '',
            }

        if r['mode'] == 'explore':
            boosting = {
                'learn_score': 0,
                'explore_score': 1,
                'compete_score': 0,
            }
            html_mode = {
                'all': '',
                'learn': '',
                'explore': 'checked',
                'compete': '',
            }

        if r['mode'] == 'compete':
            boosting = {
                'learn_score': 0,
                'explore_score': 0,
                'compete_score': 1,
            }
            html_mode = {
                'all': '',
                'learn': '',
                'explore': '',
                'compete': 'checked',
            }

        # apply cutoff by size
        if 'size' in r:
            size = r.get('size')

        # apply bundling
        if 'bundle' in r:
            bundle = r.get('bundle')

        # response filter
        if 'filter' in r:
            res_filter = r.get('filter').split(' ')

        # assing embedding
        vec = options['default_embedding']
        # load USE4 model
        if model == 'use4':
            embedding = embed_use
            vec = 'use4'
            html_embeddings['use4'] = 'checked'
            html_embeddings['use5'] = ''
        # load USE5_large model
        elif model == 'use5':
            embedding = embed_use_large
            vec = 'use5'
            html_embeddings['use4'] = ''
            html_embeddings['use5'] = 'checked'
        # exit if model is not known
        else:
            print('model not defined')
            sys.exit()

    # print some statistics
    print('###')
    print('search:', q)
    print('engines:', q_engines)
    print('field:', q_field)
    # print('model:', model)
    print('instance:', es_index)
    print('boosting:', boosting)
    print('---')

    ### PERFORM QUERY ##
    if q != '' and q != ['']:
        i = 0

        # perform keyword based search
        if 'k' in q_engines or 'b' in q_engines:
            # set tags to query
            tags = {q_field: q}

            # append match filters
            tags.update(match)

            # perform search (default scoring function: BM25 or TD*IDF)
            if 'k' in q_engines and options['search_engines']['keyword_default']:
                res = keySearch(
                    es, tags, size=size, must_not=match_not, boost=boosting, scale=True)

                # store aggregations
                aggregations_raw.append(res['aggregations'])

                # get scale
                if len(res['hits']['hits']) > 0:
                    scale = res['hits']['hits'][0]['_score']
                # parse results
                for hit in res['hits']['hits']:
                    # add some extra fields to the record
                    extra = {'index': str(i), 'search': 'k' +
                             q_field[0], 'embedding': model}
                    # format record
                    item = format_result(
                        hit, query=q, field=q_field, scale=scale, extra=extra, treshold=treshold, mode=request.endpoint)
                    # append record
                    if item != None:
                        items.append(item)
                        i += 1

            # perform search (scoring function: boolean)
            if 'b' in q_engines and options['search_engines']['keyword_boolean']:
                tags = {k+'_boolean': v for k, v in tags.items()}
                res = keySearch(
                    es, tags, size=size, must_not=match_not, boost=boosting, scale=True)

                # store aggregations
                aggregations_raw.append(res['aggregations'])

                # get scale
                if len(res['hits']['hits']) > 0:
                    scale = res['hits']['hits'][0]['_score']
                # parse results
                for hit in res['hits']['hits']:
                    # add some extra fields to the record
                    extra = {'index': str(i), 'search': 'b' +
                             q_field[0], 'embedding': model}
                    # format record
                    item = format_result(
                        hit, query=q, field=q_field, scale=scale, extra=extra, treshold=treshold, mode=request.endpoint)
                    # append record
                    if item != None:
                        items.append(item)
                        i += 1

        # perform semantic search
        if 's' in q_engines and options['search_engines']['semantic']:
            # set tags to query
            tags = [q_field+'_vector_'+model]

            # perform search
            res = semSearch(
                es, q, tags, embedding, match, size=size, must_not=match_not, boost=boosting)

            # store aggregations
            aggregations_raw.append(res['aggregations'])

            # parse results
            for hit in res['hits']['hits']:
                # add some extra fields to the record
                extra = {'index': str(i), 'search': 's' +
                         q_field[0], 'embedding': model}
                # format record
                item = format_result(
                    hit, query=q, field=q_field, extra=extra, treshold=treshold, mode=request.endpoint)
                # append record
                if item != None:
                    items.append(item)
                    i += 1

        # bundle items (similar matches by engine get grouped together)
        if bundle == True:
            items = bundle_results(items)

        # cut items
        if options['cutoff_items']:
            items = items[:int(size)]

        # filter response
        if len(res_filter) > 0:
            items = filter_response(items, res_filter)

    # runtime
    end = time.time()
    dur = round(end-start, 3)
    print('runtime:', dur, 'sec')

    # print first record
    if debug['print_first_record']:
        if len(items) > 0:
            print('first record:', items[0])
        else:
            print('no records found')

    # bundle aggregations
    aggregations = {}
    for a in aggregations_raw:
        aggregations = bundle_aggregations(aggregations, a)

    # print aggregations
    if debug['print_aggregations']:
        print('aggregations:', json.dumps(b, indent=2))

    # refresh guide
    if len(aggregations) > 0:
        guide = {}
        for k, v in aggregations.items():
            guide[k] = v.keys()

    # scale search scores
    scale = len(q_engines)
    if options['scale_search_score'] and scale > 1:
        for i in items:
            i.update((k, round(v/scale, 3))
                     for k, v in i.items() if k == 'search_score')

    # return view to gui endpoint
    if request.endpoint == 'gui':
        # sort items
        items = sorted(items, key=lambda k: k['search_score'], reverse=True)

        match['query'] = q
        return render_template(
            'index.html',
            match=match,
            match_not=match_not,
            query_engines=q_engines,
            query_secondary=html_q_secondary,
            query_mode=html_mode,
            embeddings=html_embeddings,
            items=items,
            runtime=dur,
            guides=guide,
            aggregations=aggregations,
            aggs_checked=aggs_checked,
            debug=debug_view,
        )

    # return json to api endpoint
    if request.endpoint == 'api':
        # sort items
        sort_key = 'search_score'
        sort_reverse = True
        if 'sort' in r:
            sort = r['sort'].split(' ')
            # get field to be sorted
            if len(items) > 0 and sort[0] in items[0]:
                sort_key = sort[0]
            # detect reverse sort order
            if len(sort) > 1 and sort[1] == 'reverse':
                sort_reverse = True
            else:
                sort_reverse = False
        # perform sorting
        items = sorted(items, key=lambda k: k[sort_key], reverse=sort_reverse)

        # return csv
        if 'format' in r and r['format'] == 'csv':
            df = pd.DataFrame(items)
            csv = df.to_csv(index=False, sep=';')
            return Response(csv, mimetype='text/csv')

        # return json
        else:
            return Response(json.dumps(items), mimetype='application/json')


# start app
if __name__ == '__main__':
    app.run(debug=True)
