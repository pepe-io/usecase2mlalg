# imports
import json
import sys
from elasticsearch import Elasticsearch


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
        }
    }
}
}

# dump indices
#ret = es.indices.delete(index='usecase2mlalg')

# 400 caused by IndexAlreadyExistsException,
ret = es.indices.create(index='usecase2mlalg', ignore=400, body=b)
print(json.dumps(ret, indent=4))
print('Please try: http://localhost:9200/usecase2mlalg')
