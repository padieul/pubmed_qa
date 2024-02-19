from opensearchpy import OpenSearch
from models import AnglEModel
from langchain_community.vectorstores import OpenSearchVectorSearch
from langchain_community.llms import Replicate
from langchain import HuggingFaceHub


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
            opensearch_url="https://opensearch:9200",
            http_auth=("admin", "admin"),
            use_ssl=False,
            verify_certs=False,
            ssl_assert_hostname=False,
            ssl_show_warn=False,
        )

    return os_store


def llm_model(name: str = "falcon-7b-instruct"):
    """
    Create a new LLM model for langChain pipeline
    """
    if name == "replicate":
        llm = Replicate(
            model="meta/llama-2-70b-chat:02e509c789964a7ea8736978a43525956ef40397be9033abf9fd2badfe68c9e3",
            model_kwargs={"temperature": 0.01, "max_length": 500, "top_p": 1},
        )
    elif name == "falcon-7b-instruct":
        repo_id = "tiiuae/falcon-7b-instruct" 
        llm = HuggingFaceHub(
            repo_id=repo_id, model_kwargs={"temperature": 0.01, "max_new_tokens": 500}
        )        
    return llm


def build_references(sources):
    '''
    Format a list of URLs from the source documents returned from the vector database
    '''
    references = "\n\nReferences:"
    for source in sources:
        url = f"\nhttps://pubmed.ncbi.nlm.nih.gov/{str(source.metadata['pmid'])}/"
        references += url
    return references
