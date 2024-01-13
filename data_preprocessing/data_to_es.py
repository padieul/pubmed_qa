from elasticsearch import Elasticsearch, helpers, exceptions
import pandas as pd
from tqdm import tqdm
import json

# This script will upload the data chunks with their embeddings to Elasticsearch instance

# Create the client instance to connect to Elasticsearch
client = Elasticsearch(
    "http://localhost:9200"    
)



# Define the index mapping for data
index_mapping= {
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
          "type": "dense_vector",
          "dims": 768,
          "index": "true",
          "similarity": "cosine"
      }
    }
}


# Create the index only if it is not created already
try:
    client.indices.get(index="pubmed_index")
except exceptions.NotFoundError:
    client.indices.create(index="pubmed_index", mappings=index_mapping)



# Help function o store the date in Elasticsearch
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
df_pubmed = pd.read_csv('C:\\Users\\prdie\\OneDrive\\Sources\\pubmed_qa\\pubmed_qa\\data_preprocessing\\data\\data_embeddings.csv')


# Load data in batches to Elasticsearch
start = 0
end = len(df_pubmed)
batch_size = 100
for batch_start in tqdm(range(start, end, batch_size), desc ="Uploading data to Elasticsearch"):
    batch_end = min(batch_start + batch_size, end)
    batch_dataframe = df_pubmed.iloc[batch_start:batch_end]
    actions = dataframe_to_bulk_actions(batch_dataframe)
    helpers.bulk(client, actions)


