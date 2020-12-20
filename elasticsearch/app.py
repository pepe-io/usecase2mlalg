# imports
from flask import Flask, request
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

#embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")
embed = hub.load("./.USE4/")


def keywordSearch(es, q):
    # Search by Keywords
    b = {
        'query': {
            'match': {
                "title": q
            }
        }
    }

    res = es.search(index='usecase2mlalg', body=b)

    return res


def sentenceSimilaritybyNN(es, sent):
    # Search by Vec Similarity
    query_vector = tf.make_ndarray(
        tf.make_tensor_proto(embed([sent]))).tolist()[0]
    b = {"query": {
        "script_score": {
            "query": {
                "match_all": {}
            },
            "script": {
                "source": "cosineSimilarity(params.query_vector, 'title_vector') + 1.0",
                "params": {"query_vector": query_vector}
            }
        }
    }
    }

    # print(json.dumps(b,indent=4))
    res = es.search(index='usecase2mlalg', body=b)

    return res


@app.route('/', methods=['POST', 'GET'])
# index route
def index():
    return 'moin'


@app.route('/search/<query>')
# search route
def search(query):
    q = query.replace("+", " ")
    res_kw = keywordSearch(es, q)
    res_semantic = sentenceSimilaritybyNN(es, q)

    ret = ""
    for hit in res_kw['hits']['hits']:
        ret += (" KW: " + str(hit['_score']) +
                "\t" + hit['_source']['title'] + "\n")

    for hit in res_semantic['hits']['hits']:
        ret += (" Semantic: " + str(hit['_score']) +
                "\t" + hit['_source']['title'] + "\n")
    return ret


# start app
if __name__ == "__main__":
    app.run(debug=True)
