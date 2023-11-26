import requests
import xml.etree.ElementTree as ET

from tqdm import tqdm

# Replace with your own API key if registered
api_key = "4aac4074dc4662953a02c33abf21d1232908"

# Define the query parameters
query = "intelligence"
start_date = "2013/01/01"
end_date = "2023/12/31"

# Construct the PubMed E-Utilities API URL for the ESearch request to get the list of PMIDs
esearch_url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?db=pubmed&term={query}&mindate={start_date}&maxdate={end_date}&api_key={api_key}"

"""

# Make the ESearch API request to get the list of PMIDs
response = requests.get(esearch_url)

if response.status_code == 200:
    # Parse the XML response to extract PMIDs
    root = ET.fromstring(response.content)
    pmids = [element.text for element in root.findall(".//Id")]

    # Construct the URL for the ESummary request to get metadata and abstracts
    pmids_str = ",".join(pmids)
    esummary_url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi?db=pubmed&id={pmids_str}&api_key={api_key}"

    # Make the ESummary API request to get metadata and abstracts
    response = requests.get(esummary_url)

    if response.status_code == 200:
        # Parse the XML response to extract metadata and abstracts
        root = ET.fromstring(response.content)
        for doc in root.findall(".//DocSum"):
            abstract = None
            try:
                pmid = doc.find(".//Id").text
                title = doc.find(".//Item[@Name='Title']").text
                authors = doc.find(".//Item[@Name='AuthorList']").text
                pub_date = doc.find(".//Item[@Name='PubDate']").text
                abstract = doc.find(".//Item[@Name='Abstract']").text
            except:
                print("NO DATA!")

            # Print or store the metadata and abstract as needed
            print(f"PMID: {pmid}")
            print(f"Title: {title}")
            print(f"Authors: {authors}")
            print(f"Publication Date: {pub_date}")
            if not abstract == None:
                print(f"Abstract: {abstract}\n")
    else:
        print(f"ESummary request failed with status code: {response.status_code}")
else:
    print(f"ESearch request failed with status code: {response.status_code}")
"""

from metapub import PubMedFetcher
import pandas as pd

# Initialize the PubMedFetcher
fetch = PubMedFetcher()

# Define your query and the number of articles you want to retrieve
query = "intelligence"
num_of_articles = 5 #177505  # You can adjust this number as needed

# Get the PMIDs for the articles matching the query and date range
pmids = fetch.pmids_for_query(query, retmax=num_of_articles, datetype='pdat', mindate='2013', maxdate='2023')

# Create dictionaries to store metadata
titles = {}
abstracts = {}
authors = {}
years = {}
journals = {}

# Loop through PMIDs and fetch metadata for each article
for pmid in tqdm(pmids):
    article = fetch.article_by_pmid(pmid)
    titles[pmid] = article.title
    abstracts[pmid] = article.abstract
    authors[pmid] = article.authors
    years[pmid] = article.year
    journals[pmid] = article.journal
    # more metadaa can be added as needed

# Create DataFrames from the dictionaries
Title = pd.DataFrame(list(titles.items()), columns=['pmid', 'Title'])
Abstract = pd.DataFrame(list(abstracts.items()), columns=['pmid', 'Abstract'])
Author = pd.DataFrame(list(authors.items()), columns=['pmid', 'Author'])
Year = pd.DataFrame(list(years.items()), columns=['pmid', 'Year'])
Journal = pd.DataFrame(list(journals.items()), columns=['pmid', 'Journal'])

# Merge DataFrames to create a single DataFrame with metadata
data_frames = [Title, Abstract, Author, Year, Journal]
df_merged = pd.concat(data_frames, axis=1)

# Save the DataFrame to a CSV file
df_merged.to_csv('pubmed_intelligence_articles.csv', index=False)

# Display the DataFrame
print(df_merged)