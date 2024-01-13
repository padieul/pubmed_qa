from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from elasticsearch import Elasticsearch, helpers, exceptions
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('all-mpnet-base-v2')


client = Elasticsearch(
    "http://localhost:9200"    
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




@app.get("/retrieve_documents_dense")
def retrieve_documents(query_str: str):
    response_message = client.search(
        index = "pubmed_index",
        knn={
            "field": "embedding",
            "query_vector":  model.encode(query_str).tolist(),
            "k": 10,
            "num_candidates": 100
            }
    )
    responses_dict = pretty_response(response_message['hits']['hits'])
    return {"message": responses_dict}


@app.get("/retrieve_documents_sparse")
def retrieve_documents_sparse(message: str):
    response_message = f"FastAPI detected, that you said: {message}"
    return {"message": response_message}