from opensearchpy import OpenSearch
from models import AnglEModel
from langchain_community.vectorstores import OpenSearchVectorSearch
from langchain_community.llms import Replicate
from langchain import HuggingFaceHub

from langchain_core.vectorstores import VectorStoreRetriever
from langchain_core.pydantic_v1 import Field
from langchain_core.documents.base import Document
from typing import List


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



class VariableRetriever(VectorStoreRetriever):
    vectorstore: VectorStoreRetriever
    search_type: str = "similarity"
    search_kwargs: dict = Field(default_factory=dict)
    #filter_title: str  
    #filter_month: str
    filter_year: str
    #filter_prefix: str

    """
    def retrieve(self):
        return self.retriever.retrieve()
    """
    
    def get_relevant_documents(self, query: str) -> List[Document]:
        results =  self.vectorstore.get_relevant_documents(query=query)
        print(f"Lenght of results: {len(results)}")
        filtered_results = [doc for doc in results if str(doc.metadata['year']).startswith(str(self.filter_year))]
        print(f"Lenght of filtered results: {len(filtered_results)}")
        if len(filtered_results) > 3:
            return filtered_results[:3]
        else:
            return filtered_results

"""
'source_documents': [Document(page_content='and severe motor disorders with and without truncal tone impairments 
treated in two specialized hospitals (60 inpatients and 42 outpatients; 60 males, mean age 16.5 +/- 1.2 years, 
range 12 to 18 yrs). Clinical and functional data were collected between 2006 and 2021. TT-PredictMed, a multiple 
logistic regression prediction model, was developed to identify factors associated with hypotonic or spastic TT following
 the guidelines of "Transparent Reporting of a multivariable prediction model for', 
 metadata={'pmid': 37371021, 'title': 'Identifying Postural Instability in Children with Cerebral 
 Palsy Using a Predictive Model: A Longitudinal Multicenter Study.', 'chunk_id': 1, 'chunk': 'and severe motor 
 disorders with and without truncal tone impairments treated in two specialized hospitals 
 (60 inpatients and 42 outpatients; 60 males, mean age 16.5 +/- 1.2 years, range 12 to 18 yrs). 
 Clinical and functional data were collected between 2006 and 2021. TT-PredictMed, a multiple 
 logistic regression prediction model, was developed to identify factors associated with hypotonic 
 or spastic TT following the guidelines of "Transparent Reporting of a multivariable prediction model 
 

"""


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

    Parameters:
        name (str): The name for LLM to be used

    Returns:
        llm (llm object): LangChain initialized LLM object
    
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

    Parameters:
        sources (List[Document]): A list for source documents retrieved from OpenSearch 

    Returns:
        references (str): A string of URLs for the abstracts in the retrieved documents

    '''

    pmids = []

    references = ""
    for source in sources:
        pmid = str(source.metadata['pmid'])
        # Create URLs for abstracts once and without duplicates
        if pmid not in pmids:
            url = f"|https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
            references += url
            pmids.append(pmid)
    return references

def processed_output(output: str):
    '''
    Process the string generated by LangChain before sending it to the web interface

    Parameters:
        output (str): The output received from the LLM 

    Returns:
        formatted_answer (str): A properly formatted answer to display in the web front-end
    '''
    formatted_answer = output[output.find('Answer:') + 8:]

    return formatted_answer
