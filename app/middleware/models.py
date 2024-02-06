from angle_emb import AnglE, Prompts
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
