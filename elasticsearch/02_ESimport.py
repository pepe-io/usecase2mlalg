# imports
from flask import Flask, request
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
import json
# import time
import sys
import csv
from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk
import tensorflow as tf
import tensorflow_hub as hub


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


# configure db
app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///usecase2ml.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)


class Record(db.Model):
    # configure table
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
    ml_libraries = db.Column(db.String(256))
    ml_terms = db.Column(db.String(256))
    score = db.Column(db.Float())
    score_raw = db.Column(db.String(256))
    lic = db.Column(db.String(256))
    language = db.Column(db.String(256))

    def __repr__(self):
        return '<Record %r>' % self.id


# create db with table if not present
db.create_all()


# load USE4 model

# embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")
embed = hub.load("./.USE4/")


def mapper(row, style):
    '''
    mapper to adopt csv to db-schema
    '''

    # kaggle mapping
    if style == 'kaggle':
        return {
            'title': row['title'],
            'link': row['link'],
            'category': '',
            'subcategory': '',
            'description': row['description'],
            'tags': row['tags'],
            'date_project': datetime.strptime(row['date'], "%Y-%m-%d %H:%M:%S"),
            'date_scraped': datetime.strptime(row['scraped_at'], "%Y-%m-%d %H:%M:%S"),
            'kind': 'project',
            'ml_libraries': row['ml_libs'],
            'ml_terms': row['ml_terms'],
            'score': row['score_views'],
            'score_raw': json.dumps({'views': row['views'], 'votes': row['votes'], 'score_private': row['score_private'], 'score_public': row['score_public']}),
            'lic': row['license'],
            'language': row['type'],
        }

    # github mapping
    if style == 'github':
        return {
            'title': row['name'],
            'link': row['link'],
            'category': row['industry'],
            'subcategory': row['type'],
            'description': row['description2'],
            'tags': '',
            'date_project': datetime.strptime(row['pushed_at'], "%Y-%m-%d %H:%M:%S"),
            'date_scraped': datetime.strptime(row['scraped_at'], "%Y-%m-%d %H:%M:%S"),
            'kind': 'project',
            'ml_libraries': row['ml_libs'],
            'ml_terms': row['keywords'],
            'score': row['stars_score'],
            'score_raw': json.dumps({'stars': row['stars'], 'contributors': row['contributors']}),
            'lic': row['license'],
            'language': row['language_primary'],
        }

    # mlart mapping
    if style == 'mlart':
        title = row['Title'] if row['Title'] != '' else row['title']
        return {
            'title': title,
            'link': row['url'],
            'category': 'Art',
            'subcategory': row['Theme'],
            'description': row['subtitle'],
            'tags': row['Technology'],
            'date_project': datetime.strptime(row['Date'], "%Y-%m-%d"),
            'date_scraped': datetime.strptime(row['scraped_at'], "%Y-%m-%d %H:%M:%S"),
            'kind': row['Medium'],
            'ml_libraries': '',
            'ml_terms': '',
            'score': 0,
            'score_raw': json.dumps({'days_since_featured': row['Days Since Featured']}),
            'lic': '',
            'language': '',
        }

    return None


# CONSTANTS
NUM_INDEXED = 100000

cnt = 0
i = 0

# kaggle competitions
# csvfile = '../data/database/kaggle_competitions_01_original.csv'
# csvformat = 'kaggle'

# mlart
# csvfile = '../data/database/mlart_01_original.csv'
# csvformat = 'mlart'

# github
csvfile = '../data/database/db_04_analyzed.csv'
csvformat = 'github'

with open(csvfile, encoding='utf-8') as csvfile:
    # readCSV = csv.reader(csvfile, delimiter=';')
    readCSV = csv.DictReader(csvfile, delimiter=';')
    # next(readCSV, None)  # skip the headers
    for row in readCSV:
        # print(row)
        row = mapper(row, csvformat)

        if row == None:
            print('mapping not found')
            sys.exit()
        # print(row)
        # print(row['title'])

        print(row['link'])
        record = Record.query.filter_by(link=row['link']).first()
        if record == None:
            i += 1
            record = Record(title=row['title'],
                            link=row['link'],
                            category=row['category'],
                            subcategory=row['subcategory'],
                            description=row['description'],
                            tags=row['tags'],
                            date_project=row['date_project'],
                            date_scraped=row['date_scraped'],
                            kind=row['kind'],
                            ml_libraries=row['ml_libraries'],
                            ml_terms=row['ml_terms'],
                            score=row['score'],
                            score_raw=row['score_raw'],
                            lic=row['lic'],
                            language=row['language'])

            try:
                db.session.add(record)
                db.session.commit()
            except Exception as e:
                print("Oops!", e.__class__, "occurred.")
                print(e)

            #print('id', record.id)

            vec_t = tf.make_ndarray(tf.make_tensor_proto(
                embed([row['title']]))).tolist()[0]
            vec_d = tf.make_ndarray(tf.make_tensor_proto(
                embed([row['description']]))).tolist()[0]

            b = {"title": row['title'],
                 "title_vector": vec_t,
                 "description": row['description'],
                 "description_vector": vec_d,
                 }
            # print(json.dumps(tmp,indent=4))

            res = es.index(index="usecase2mlalg", id=record.id, body=b)
            print(res)

        # keep count of # rows processed
        cnt += 1
        if cnt % 100 == 0:
            print(cnt)

        if cnt == NUM_INDEXED:
            break


print('##################################################')
print('Done', i, 'items added')
