from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from elasticsearch import Elasticsearch, helpers, exceptions
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('all-mpnet-base-v2')


es_container_name = "elasticsearch"
client = Elasticsearch(
    # use es container name as host and port 9200
    "http://" + es_container_name + ":9200"  
)

app = FastAPI()

# Allow all origins during development
origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


#http://localhost:8000/storage_info

def pretty_response(response):
    temp_dict = {}
    order_key = 0
    for hit in response:
        id = hit['_id']
        score = hit['_score']
        pmid = hit['_source']['pmid']
        chunk_id = hit['_source']['chunk_id']  
        chunk = hit['_source']['chunk']      
        pretty_output = (f"\nID: {id}\nPMID: {pmid}\nChunk ID: {chunk_id}\nText: {chunk}")
        temp_dict[order_key] = pretty_output
        order_key += 1
    return temp_dict

@app.get("/read_root")
def read_root(message: str):
    response_message = f"FastAPI detected, that you said: {message}"
    return {"message": response_message}

"""
# Send a knn query to Elasticsearch
response = client.search(
  index = "pubmed_index",
  knn={
      "field": "embedding",
      "query_vector":  query_emb.tolist(),
      "k": 10,
      "num_candidates": 100
    }
)

"""


@app.get("/retrieve_documents_dense")
def retrieve_documents(query_str: str):
    print("Query str: ", query_str)
    query_vector = model.encode(query_str).tolist()


    knn_dict={
      "field": "embedding",
      "query_vector":  query_vector,
      "k": 10,
      "num_candidates": 100
    }
    
    response_message = client.search(index="pubmed_index", knn=knn_dict)
    responses_dict = pretty_response(response_message['hits']['hits'])

    response_str = ""
    for key,val in responses_dict.items():
        response_str += str(key) + ": " + str(val) + "\n"
    return {"message": response_str}


@app.get("/retrieve_documents_sparse")
def retrieve_documents_sparse(message: str):
    response_message = f"FastAPI detected, that you said: {message}"
    return {"message": response_message}