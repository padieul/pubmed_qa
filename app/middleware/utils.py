from opensearchpy import OpenSearch


def pretty_response(response):
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
