from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from utils import pretty_response, os_client


from angle_emb import AnglE, Prompts

# Initialize AnglE embedding model
angle = AnglE.from_pretrained("WhereIsAI/UAE-Large-V1", pooling_strategy="cls").cuda()

# Enable Prompt.C for retrieval optimized embeddings
angle.set_prompt(prompt=Prompts.C)

# Initialize OpenSearch instance
client = os_client()

# Initialize FastAPI instance
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


# http://localhost:8000/storage_info


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
    query_vector = angle.encode({"text": query_str}).tolist()[0]

    # Defining the knn query parameters
    search_query_desne = {    
        "query": {
            "knn": {
                "embedding": {
                    "vector": query_vector,
                    "k": 10
                }
            }
        }
    }

    response_message = client.search(index="pubmed_500_200", body=search_query_desne)
    
    responses_dict = pretty_response(response_message["hits"]["hits"])

    response_str = ""
    for key, val in responses_dict.items():
        response_str += str(key) + ": " + str(val) + "\n"
    return {"message": response_str}


@app.get("/retrieve_documents_sparse")
def retrieve_documents_sparse(message: str):
    response_message = f"FastAPI detected, that you said: {message}"
    return {"message": response_message}
