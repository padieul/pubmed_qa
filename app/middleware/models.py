from angle_emb import AnglE, Prompts

from langchain_core.vectorstores import VectorStoreRetriever
from langchain_core.pydantic_v1 import Field
from langchain_core.documents.base import Document
from typing import List


class AnglEModel:
    """
    A class to wrap AnglE embedding models to be used with LangChain

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


class RetrievalFilter:

    """
    Helper Class to encapsulate the filtering logic for the retrieved documents.

    Filtering options:

    - title - is particular title included or not?
    -> empty string - no title 
    -> if one string - check if included or not
    -> field: metadata.title
    - year - particular year range included or not? 
    -> start and end
    -> start and end - if the same, then only that year 
    -> empty string - no year
    -> field: metadata.year
    - keywords - included or not 
    -> empty list - no keywords 
    -> otherwise - loop through list and check if included or not
    -> field: page_content

    """

    def __init__(self, filter_dict: dict):

        self._target_title = filter_dict["title"]
        self._target_years = filter_dict["years"]
        self._target_keywords = filter_dict["keywords"]

        print(f"Title: {self._target_title}")
        print(f"Years: {self._target_years}")
        print(f"Keywords: {self._target_keywords}")

        if self._target_title == "" and \
           self._target_years == [] and \
           self._target_keywords == []:
            self._filter_type = "no_filter"
        else:
            self._filter_type = "filter"

    def get_filter_type(self) -> str:
        return self._filter_type

    def _apply_title_filter(self, doc_list: List[Document]) -> List[Document]:
        
        return [doc for doc in doc_list if self._target_title.lower() in doc.metadata["title"].lower()]
    
    def _apply_years_filter(self, doc_list: List[Document]) -> List[Document]:
        return [doc for doc in doc_list if str(doc.metadata["year"]) in self._target_years]
    
    def _apply_keywords_filter(self, doc_list: List[Document]) -> List[Document]:
        return [doc for doc in doc_list if all(keyword.lower() in doc.page_content.lower() for keyword in self._target_keywords)]

    def apply(self, doc_list: List[Document]) -> List[Document]:

        if self._filter_type == "no_filter":
            return doc_list
        
        print(f"Applying filter: {self._filter_type}")
        filters = [self._apply_title_filter, self._apply_years_filter, self._apply_keywords_filter]

        for filter in filters:
            print(f"Applying filter: {filter.__name__}")
            print(f"Length of doc_list: {len(doc_list)}")
            doc_list = filter(doc_list)
            print(f"Length of doc_list after filtering: {len(doc_list)}")
        return doc_list
        

class VariableRetriever(VectorStoreRetriever):
    """
    A class to wrap around the Langchain VectorStoreRetriever class to enable 
    metadata filtering after retrieval as part of the RAG pipeline
    """
    vectorstore: VectorStoreRetriever
    search_type: str = "similarity"
    search_kwargs: dict = Field(default_factory=dict)
    retrieval_filter: RetrievalFilter

    
    def get_relevant_documents(self, query: str) -> List[Document]:
        results =  self.vectorstore.get_relevant_documents(query=query)
        print(f"Length of results: {len(results)}")
        filtered_results = self.retrieval_filter.apply(results) 
        print(f"Length of filtered results: {len(filtered_results)}")

        if len(filtered_results) > 3:
            return filtered_results[:3]
        else:
            return filtered_results