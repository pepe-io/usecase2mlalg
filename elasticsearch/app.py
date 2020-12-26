# imports
from flask import Flask, render_template, request
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
import json
# import time
import sys
# import csv
from elasticsearch import Elasticsearch
# from elasticsearch.helpers import bulk
import tensorflow as tf
import tensorflow_hub as hub

'''
# configure db
app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///usecase2ml.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# configure table


class Record(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(256))
    link = db.Column(db.String(256), nullable=False)
    category = db.Column(db.String(256))
    subcategory = db.Column(db.String(256))
    description = db.Column(db.Text())
    tags = db.Column(db.String(256))
    date_project = db.Column(db.DateTime)
    date_scraped = db.Column(db.DateTime)
    kind = db.Column(db.String(64))
    libraries = db.Column(db.String(256))
    score = db.Column(db.Float())
    score_raw = db.Column(db.String(256))

    def __repr__(self):
        return '<Record %r>' % self.id


# create db with table if not present
db.create_all()
'''

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


def keywordSearch(es, q, tag):
    # Search by Keywords
    # b = {
    #     'size': 1,
    #     'query': {
    #         'match': {
    #             tag: q
    #         }
    #     }
    # }

    b = {
        "size": 10,
        "query": {
            "bool": {
                "must": [
                    {"match": {tag: q}}
                ]
            }
        }
    }

    res = es.search(index='usecase2mlalg', body=b)

    return res


def keySearch(es, queries):
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
    # print(b)

    res = es.search(index='usecase2mlalg', body=b)
    return res


def sentenceSimilaritybyNN(es, sent, tag):
    # Search by Vec Similarity
    query_vector = tf.make_ndarray(
        tf.make_tensor_proto(embed([sent]))).tolist()[0]
    b = {
        'size': 10,
        'query': {
            'script_score': {
                'query': {
                    'match_all': {}
                },
                'script': {
                    'source': 'cosineSimilarity(params.query_vector, "'+tag+'") + 1.0',
                    'params': {'query_vector': query_vector}
                }
            }
        }
    }

    # print(json.dumps(b,indent=4))
    res = es.search(index='usecase2mlalg', body=b)

    return res


def semSearch(es, sent, tag_, queries):
    # Search by Vec Similarity
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


@ app.route('/test', methods=['POST', 'GET'])
# test route
def test():
    print(request.method)
    print(request.form['search'])
    return request.method


@ app.route('/', methods=['GET', 'POST'])
@ app.route('/<query>', methods=['GET'])
# index / search route
def search(query=''):
    if request.method == 'POST':
        q = request.form['search']

    if request.method == 'GET':
        q = query.replace('+', ' ')

    # print(request.form.get('kt'))
    # print(request.form.get('kd'))

    items = []

    queries = {}
    query_engines = {
        'kt': ' checked',
        'st': ' checked',
        'kd': ' checked',
        'sd': ' checked',
    }
    show_filter = 'hidden'

    if q != '':
        i = 0

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

    # print(items)
    items = bundle_results(items)
    # print(items)

    queries['query'] = q
    return render_template('index.html', query=queries, query_engines=query_engines, items=items, show_filter=show_filter)


# start app
if __name__ == '__main__':
    app.run(debug=True)
