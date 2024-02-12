import os
from os.path import dirname, join
from nltk.tokenize import word_tokenize
from nltk import PorterStemmer
import re
import string
from elasticsearch import Elasticsearch

doc_folder = './IR_data/AP_DATA/ap89_collection'
sw_path = './IR_data/AP_DATA/stoplist.txt'

textMap = {}
ps = PorterStemmer()
sw_list = []

pattern = re.compile(r"<DOC>(.*?)</DOC>", re.DOTALL)

es = Elasticsearch("http://localhost:9200")

print(es.ping())

def stem_text(text):
    stemmed = []
    for w in text.split():
        if w not in sw_list:
            stemmed.append(ps.stem(w))
    
    stemmed_text = " ".join(stemmed)
    return stemmed_text


def parse_file(file_path):
    with open(file_path, 'r', encoding = 'ISO-8859-1') as file:
        docs = pattern.findall(file.read())
        for content in docs:
            docNo = re.search(r'<DOCNO>(.*?)</DOCNO>', content, re.DOTALL).group(1).strip()
            text_blocks = re.findall(r'<TEXT>(.*?)</TEXT>', content, re.DOTALL)

            text = " ".join(text_blocks).strip()
            stemmed_text = stem_text(text)
            textMap[docNo] = stemmed_text


with open(sw_path, 'r') as file: 
    for w in file.readlines():
        sw_list.append(w.strip())


for file_name in os.listdir(doc_folder):
    if file_name != 'readme':
        file_path = os.path.join(doc_folder, file_name)
        parse_file(file_path)
        print ("Parsed file")

print('Parsing Completed')


index_name = "ap89_data0"
configurations = {
    "settings" : {
        "number_of_shards": 1,
        "number_of_replicas": 1,
        "analysis": {
            "filter": {
                "english_stop": {
                    "type": "stop",
                    "stopwords_path": "my_stoplist.txt"
                }
            },
            "analyzer": {
                "stopped": {
                    "type": "custom",
                    "tokenizer": "standard",
                    "filter": [
                        "lowercase",
                        "english_stop"
                    ]
                }
            }
      }
    },
    "mappings": {
        "properties": {
            "content": {
                "type": "text",
                "fielddata": True,
                "analyzer": "stopped",
                "index_options": "positions"
            }
        }
    }
}

es.indices.create(index = index_name, body = configurations)

def add_data(_id, text):
    es.index(
        index = index_name,
        body = {
            'content': text
        },
        id = _id)

for key in textMap:
    add_data(key, textMap[key])

print("All documents have been added to the index!")
print(len(textMap))