import os

"""  Set the API tokens for all services """

api_tokens = {"HUGGINGFACEHUB_API_TOKEN": "...."}


def set_api_keys():
    for key, value in api_tokens.items():
        os.environ[key] = value
