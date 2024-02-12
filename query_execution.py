from elasticsearch import Elasticsearch, helpers
from elasticsearch.client import IndicesClient
import os
import json
import requests
import re
import math
import numpy as np
from operator import getitem

es = Elasticsearch(timeout = 300)
ic = IndicesClient(es)

# Required Data Structures
sw_path = './IR_data/AP_DATA/stoplist.txt'
sw_list = []
query_path = './IR_data/AP_DATA/query_desc.51-100.short.txt'
query_list = []
doc_vecs = {}
doc_length = {}
term_doc_freq = {}
term_freq = {}
unique_words = set()

def get_doc_ids():
    with open('./IR_data/AP_DATA/doclist_new_0609.txt') as file:
        doc_ids = []
        for line in file:
            line = (line.strip().split())[1]
            doc_ids.append(line)
    return doc_ids

def query_analyzer(query):
    body = {
        "tokenizer": "standard",
        "filter": ["porter_stem", "lowercase", "english_stop"],
        "text": query
    }
    res = ic.analyze(body = body, index = "ap89_data0")
    return [list["token"] for list in res["tokens"] if list["token"] not in sw_list]


def query_search(query):
    res = es.search(
        index = "ap89_data0",
        body = {
            "size": 1000,
            "query": {
                "match": {
                    "content": " ".join(query)
                }
            }
        }, 
        scroll = "3m"
    )
    return res


def get_term_vectors(doc_ids):
    body = {
        "ids": doc_ids,
        "parameters": {
            "fields": ["content"],
            "term_statistics": True,
            "offsets": True,
            "payloads": True,
            "positions": True,
            "field_statistics": True
        }
    }
    '''
    r = requests.post(
        os.path.join('http://localhost:9200', 'ap89_data0', '_mtermvectors'),
        headers = {'Content-Type': 'application/json'},
        data = json.dumps(data)
    )
    '''
    return es.mtermvectors(index = 'ap89_data0', body = body)


def retrieve_term_doc_info(batch_size = 1000):
    for i in range(0, len(doc_ids), batch_size):
        print('retrieved term vecs for', i)
        id_batch = doc_ids[i : i + batch_size]
        vectors = get_term_vectors(id_batch)

        for term_vector in vectors['docs']:
            doc_id = term_vector['_id'] 

            if 'content' not in term_vector['term_vectors']:
                doc_vecs[doc_id] = {}
                doc_length[doc_id] = 0
            else:
                terms = term_vector['term_vectors']['content']['terms']

                for t in terms:
                    term_doc_freq[t] = terms[t]['doc_freq']
                    term_freq[t] = terms[t]['ttf']

                doc_vecs[doc_id] = terms
                doc_length[doc_id] = sum([terms[x]['term_freq'] for x in terms])
                for t in terms:
                    unique_words.add(t)

doc_ids = get_doc_ids()
num_docs = len(doc_ids)

print(num_docs)

retrieve_term_doc_info()

avg_doc_length = sum([doc_length[d] for d in doc_ids]) / num_docs
total_doc_length = sum([doc_length[d] for d in doc_length])
vocab_size = len(unique_words)

def df(w):
    if w in term_doc_freq:
        return term_doc_freq[w]
    return 0


def tf(w, d):
    if w in doc_vecs[d]:
        return doc_vecs[d][w]['term_freq']
    return 0


def tf_q(w, q):
    return q.count(w)


def scroll_body(scroll_id):
    body = {
        "scroll_id": scroll_id,
        "scroll": "3m"
    }
    return body


with open(sw_path, 'r') as file: 
    for w in file.readlines():
        sw_list.append(w.strip())

print("opened stopwords file")

with open(query_path, 'r') as file: 
    for q in file.readlines():
        query_list.append(q.strip())

print("opened query file")


# Elastic Search Retrieval Method
def ES_Search(query_list):
    scores = {}
    for query in range(len(query_list)):
        processed_query = query_analyzer(query_list[query])
        search_results = query_search(processed_query)
        scroll_id = search_results.get("_scroll_id")

        docs_collected = 0
        while scroll_id and docs_collected < 1000:
            hits = search_results.get("hits", {}).get("hits", [])
            if len(hits) == 0:
                print(processed_query)
                break
        
            for hit in hits:
                doc_id = hit["_id"]
                score = hit["_score"]

                if query not in scores:
                    scores[query] = []

                scores[query].append((doc_id, (docs_collected + 1), score))
                docs_collected += 1

            search_results = es.scroll(body=scroll_body(scroll_id))
            scroll_id = search_results.get("_scroll_id")
        
        docs_collected = 0
        
    return scores

result_scores_es = ES_Search(query_list)
print("found top 1000 docs for all queries using Elastic Search")

with open("output_es_built_in.txt", "w") as output_file:
    for query, results in result_scores_es.items():
        for rank, (doc_id, rank, score) in enumerate(results, start=1):
            output_file.write(f"{(query_list[query]).split('.')[0]} Q0 {doc_id} {rank} {score} Exp\n")


# Okapi TF Retrieval Model
def okapi_TF(query_list):
    scores = {}
    for query in range(len(query_list)):
        processed_query = query_analyzer(query_list[query])

        if query not in scores:
            scores[query] = []

        for doc in doc_ids:
            score = sum([(tf(w, doc) / (tf(w, doc) + 0.5 + (1.5 * (doc_length[doc] / avg_doc_length)))) for w in processed_query])
            scores[query].append((doc, score))
        
        scores[query] = sorted(scores[query], key = lambda x: x[1], reverse = True)[:1000]
    
    return scores

result_scores_okapi_tf = okapi_TF(query_list)
print("Found top 1000 docs for all queries using Okapi TF")

with open("output_okapi_tf.txt", "w") as output_file:
    for query in range(len(result_scores_okapi_tf)):
        for doc in range(len(result_scores_okapi_tf[query])):
            doc_score = (result_scores_okapi_tf[query])[doc]
            output_file.write(f"{(query_list[query]).split('.')[0]} Q0 {doc_score[0]} {doc + 1} {doc_score[1]} Exp\n")


# TF-IDF Retrieval Model
def tf_idf(query_list):
    scores = {}
    for query in range(len(query_list)):
        processed_query = query_analyzer(query_list[query])

        if query not in scores:
            scores[query] = []

        for doc in doc_ids:
            score = sum([((tf(w, doc) / (tf(w, doc) + 0.5 + (1.5 * (doc_length[doc] / avg_doc_length)))) * (np.log10(num_docs / df(w)))) 
                         for w in processed_query if df(w) > 0])
            scores[query].append((doc, score))
        
        scores[query] = sorted(scores[query], key = lambda x: x[1], reverse = True)[:1000]
    
    return scores

result_scores_tf_idf = tf_idf(query_list)
print("Found top 1000 docs for all queries using TF-IDF")

with open("output_tf_idf.txt", "w") as output_file:
    for query in range(len(result_scores_tf_idf)):
        for doc in range(len(result_scores_tf_idf[query])):
            doc_score = (result_scores_tf_idf[query])[doc]
            output_file.write(f"{(query_list[query]).split('.')[0]} Q0 {doc_score[0]} {doc + 1} {doc_score[1]} Exp\n")


# Okapi BM-25 Retrieval Model
def bm_25(query_list):
    scores = {}
    for query in range(len(query_list)):
        processed_query = query_analyzer(query_list[query])

        if query not in scores:
            scores[query] = []

        for doc in doc_ids:
            score = bm_25_score(doc, processed_query)
            scores[query].append((doc, score))
        
        scores[query] = sorted(scores[query], key = lambda x: x[1], reverse = True)[:1000]
    
    return scores


def bm_25_score(d, q):
    k1 = 1.2
    k2 = 500
    b = 0.75
    score = 0
    
    for w in q:
        t1 = np.log10((num_docs + 0.5) / (0.5 + df(w)))
        
        t2 = (tf(w, d) +  k1 * tf(w,d)) / (tf(w,d) + k1 * ((1 - b) + b * (doc_length[d] / avg_doc_length)))
        
        t3 = (tf_q(w,q) + k2 * tf_q(w,q)) / (tf_q(w,q) + k2)
        
        score += (t1 * t2 * t3)
    return score

result_scores_bm_25 = bm_25(query_list)
print("Found top 1000 docs for all queries using Okapi BM-25")

with open("output_bm_25.txt", "w") as output_file:
    for query in range(len(result_scores_bm_25)):
        for doc in range(len(result_scores_bm_25[query])):
            doc_score = (result_scores_bm_25[query])[doc]
            output_file.write(f"{(query_list[query]).split('.')[0]} Q0 {doc_score[0]} {doc + 1} {doc_score[1]} Exp\n")

print("done!")