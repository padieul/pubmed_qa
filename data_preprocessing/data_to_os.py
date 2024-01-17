from opensearchpy import OpenSearch, helpers, exceptions
import pandas as pd
from tqdm import tqdm
import json


# This script will upload the data chunks with their embeddings to OpenSearch instance

# OpenSearch instance parameters
host = 'localhost'
port = 9200
auth = ('admin', 'admin')

# Create the client with SSL/TLS enabled and disable warnings
client = OpenSearch(
    hosts = [{'host': host, 'port': port}],    
    http_auth = auth,
    use_ssl = True,
    verify_certs = False,
    ssl_show_warn = False,
)


# Index parameters and mapping
index_name = "pubmed_index"

index_mapping = {
    "settings": {
        "index": {
        "knn": True,
        "knn.algo_param.ef_search": 100
        }
    },
    "mappings": {
        "properties": {
            "pmid": {
                "type": "integer"
            },
            "chunk_id": {
                "type": "integer"
            },
            "chunk": {
                "type": "text"
            },
            "year": {
                "type": "integer"
            },
            "month": {
                "type": "integer"
            },
            "embedding": {
                "type": "knn_vector",
                "dimension": 768,
                "method": {
                    "name": "hnsw",
                    "engine": "lucene"
                }                
            }
        }
    }
}


# Create the index only if it is not created already
try:
    client.indices.get(index=index_name)
except exceptions.NotFoundError:
    client.indices.create(index_name, body=index_mapping)


# Help function to store the data in OpenSearch
def dataframe_to_bulk_actions(df):
    for index, row in df.iterrows():
        yield {
            "_index": 'pubmed_index',            
            "_source": {
                'pmid' : row["pmid"],
                'chunk_id' : row["chunk_id"],
                'chunk' : row["chunk"],
                'year' : row["year"],
                'month' : row["month"],
                'embedding' : json.loads(row["embedding"])
            }
        }


# Read the data and the embeddings from CSV file
df_pubmed = pd.read_csv('C:\\Code\\Project\\data\\data_embeddings.csv')

# Upload data in batches to OpenSearch
start = 0
end = len(df_pubmed)
batch_size = 100
for batch_start in tqdm(range(start, end, batch_size), desc ="Uploading data to OpenSearch"):
    batch_end = min(batch_start + batch_size, end)
    batch_dataframe = df_pubmed.iloc[batch_start:batch_end]
    actions = dataframe_to_bulk_actions(batch_dataframe)
    helpers.bulk(client, actions)
