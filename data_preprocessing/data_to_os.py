from opensearchpy import OpenSearch, helpers, exceptions
import pandas as pd
from tqdm import tqdm
import json
import os, sys


# This script will upload the data chunks with their embeddings to OpenSearch instance

# Helper function to store the data in OpenSearch
def dataframe_to_bulk_actions(df, index_name):
    for index, row in df.iterrows():
        yield {
            "_index": index_name,            
            "_source": {
                'pmid' : row["pmid"],
                'title' : row["title"],
                'chunk_id' : row["chunk_id"],
                'chunk' : row["chunk"],
                'year' : row["year"],
                'month' : row["month"],
                'embedding' : json.loads(row["embedding"])
            }
        }



def upload_data(source_file: str=None, index_name: str=None) -> None:
    '''
    A helper function to upload data to OpenSearch

    Parameters:
        source_file (str): the full path to the file containing the chunked data and the embeddings
        index_name (str): the name of the index in OpenSearch
    '''
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
                "title": {
                    "type": "text"
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
                    "dimension": 1024,
                    "method": {
                        "name": "hnsw",
                        "engine": "lucene"
                    }                
                },
                "vector_field": {
                    "type": "alias",
                    "path" : "embedding"
                }
            }
        }
    }


    # Create the index only if it is not created already
    try:
        client.indices.get(index=index_name)
    except exceptions.NotFoundError:
        client.indices.create(index_name, body=index_mapping)


    # Read the data and the embeddings from CSV file
    df_pubmed = pd.read_csv(source_file)

    # Upload data in batches to OpenSearch
    start = 0
    end = len(df_pubmed)
    batch_size = 100
    for batch_start in tqdm(range(start, end, batch_size), desc ="Uploading data to OpenSearch"):
        batch_end = min(batch_start + batch_size, end)
        batch_dataframe = df_pubmed.iloc[batch_start:batch_end]
        actions = dataframe_to_bulk_actions(batch_dataframe, index_name)
        helpers.bulk(client, actions)


if __name__ == "__main__":

    source_file = os.path.join(sys.path[0], "data\\data_embeddings_500_250.csv") if os.name == "nt" else os.path.join(sys.path[0], "data/data_embeddings_500_250.csv")
    
    index_name = "pubmed_500_200"

    upload_data(source_file=source_file, index_name=index_name)
