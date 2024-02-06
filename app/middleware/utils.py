from opensearchpy import OpenSearch
from angle_emb import AnglE, Prompts
from langchain_community.vectorstores import OpenSearchVectorSearch
from typing import List


class AnglEModel:
    """
    A class to wrap AnglE to be used with LangChain

    Attributes:
        angel (model): Angle embedding model
    """

    def __init__(self, text_type: str = "query") -> None:
        """
        Initialize the Angle model with text type

        Parameters:
            text_type (str): the type of the text to be embedded, passage or query
        """
        self.angle = AnglE.from_pretrained(
            "WhereIsAI/UAE-Large-V1", pooling_strategy="cls"
        ).cuda()

        # Enable Prompt.C for retrieval optimized embeddings
        if text_type == "query":
            self.angle.set_prompt(prompt=Prompts.C)

    def embed_query(self, query: str = None) -> List[float]:
        """
        Generate and embedding for the query

        Parameters:
            query (str): the query to be embedded as a string

        Returns:
            List[float]: a list of floats with a length of 1024
        """
        self.angle

        embedding = self.angle.encode({"text": query}, to_numpy=True)

        return embedding.tolist()[0]


def pretty_response(response):
    """
    Generated dictionary of a properly formatted OpenSearch search results

    Parameters:
        response (json): the response receive from OpenSearch

    Returns:
        dict: a dictionary of all received results
    """
    temp_dict = {}
    order_key = 0
    for hit in response:
        id = hit["_id"]
        score = hit["_score"]
        pmid = hit["_source"]["pmid"]
        chunk_id = hit["_source"]["chunk_id"]
        chunk = hit["_source"]["chunk"]
        pretty_output = f"\nID: {id}\nPMID: {pmid}\nChunk ID: {chunk_id}\nText: {chunk}"
        temp_dict[order_key] = pretty_output
        order_key += 1
    return temp_dict


def os_client():
    """
    Create a new OpenSearch client object

    Returns:
        OpenSearch: an OpenSearch object that represent the running instance using default settings
    """
    # OpenSearch instance parameters
    host = "opensearch"
    port = 9200
    auth = ("admin", "admin")

    # Create the client with SSL/TLS enabled and disable warnings
    return OpenSearch(
        hosts=[{"host": host, "port": port}],
        http_auth=auth,
        use_ssl=True,
        verify_certs=False,
        ssl_show_warn=False,
    )


def opensearch_vector_store(index_name: str = None):
    """
    Create an OpenSearch vector store for RAG pipeline

    Parameters:
        index_name (str): the name of OpenSearch index where documents are stored

    Returns:
        OpenSearchVectorSearch: LangChain OpenSearch vector store object
    """
    if index_name is not None:

        os_store = OpenSearchVectorSearch(
            embedding_function=AnglEModel(),
            index_name=index_name,
            opensearch_url="https://localhost:9200",
            http_auth=("admin", "admin"),
            use_ssl=False,
            verify_certs=False,
            ssl_assert_hostname=False,
            ssl_show_warn=False,
        )

    return os_store
