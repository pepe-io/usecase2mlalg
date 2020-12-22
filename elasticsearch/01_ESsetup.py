# imports
import json
import sys
from elasticsearch import Elasticsearch

# parse arguments
print(len(sys.argv), sys.argv)
if len(sys.argv) == 1:
    print('use -up or -down to setup your elasticsearch instance')
    sys.exit()

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

# Refer: https://www.elastic.co/guide/en/elasticsearch/reference/current/mapping.html
# Mapping: Structure of the index
# Property/Field: name and type
b = {"mappings": {
    "properties": {
        "title": {
            "type": "text",
                    "analyzer": "english"
        },
        "title_vector": {
            "type": "dense_vector",
                    "dims": 512
        },
        "description": {
            "type": "text",
                    "analyzer": "english"
        },
        "description_vector": {
            "type": "dense_vector",
                    "dims": 512
        },
        "link": {
            "type": "text"
        },
        "category": {
            "type": "text",
                    "analyzer": "english"
        },
        "category_score": {
            "type": "float"
        },
        "subcategory": {
            "type": "text",
                    "analyzer": "english"
        },
        "subcategory_score": {
            "type": "float"
        },
        "tags": {
            "type": "text"
        },
        "kind": {
            "type": "text"
        },
        "ml_libs": {
            "type": "text"
        },
        "host": {
            "type": "text"
        },
        "license": {
            "type": "text"
        },
        "language": {
            "type": "text"
        },
        "score": {
            "type": "float"
        },
        "date_project": {
            "type": "date"
        },
        "date_scraped": {
            "type": "date"
        },
    }
}
}

# dump indices
if sys.argv[1] == '-down':
    ret = es.indices.delete(index='usecase2mlalg')
    print('elasticsearch instance dropped')

# 400 caused by IndexAlreadyExistsException,
if sys.argv[1] == '-up':
    ret = es.indices.create(index='usecase2mlalg', ignore=400, body=b)
    print(json.dumps(ret, indent=4))
    print('elasticsearch instance created')
    print('Please visit: http://localhost:9200/usecase2mlalg')
