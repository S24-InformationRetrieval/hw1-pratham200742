from elasticsearch import Elasticsearch, helpers
from elasticsearch.client import IndicesClient
import json
import re
import math
from operator import getitem

es = Elasticsearch()
ic = IndicesClient(es)

# Required Data Structures
sw_path = './IR_data/AP_DATA/stoplist.txt'
sw_list = []
query_path = './IR_data/AP_DATA/query_desc.51-100.short.txt'
query_list = []
term_vector_dict = {}
doc_length_dict = {}

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


def scroll_body(scroll_id):
    body = {
        "scroll_id": scroll_id,
        "scroll": "3m"
    }
    return body


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
        
        print("found 1000 docs for this query")
        docs_collected = 0
        
    return scores

with open(sw_path, 'r') as file: 
    for w in file.readlines():
        sw_list.append(w.strip())

print("opened stopwords file")

with open(query_path, 'r') as file: 
    for q in file.readlines():
        query_list.append(q.strip())

print("opened query file")

result_scores = ES_Search(query_list)

print("found top 1000 docs for each query")
print(len(result_scores))
print(len(result_scores[0]))

with open("output_es_built_in.txt", "w") as output_file:
    for query, results in result_scores.items():
        for rank, (doc_id, rank, score) in enumerate(results, start=1):
            output_file.write(f"{(query_list[query])[:2]} Q0 {doc_id} {rank} {score} Exp\n")