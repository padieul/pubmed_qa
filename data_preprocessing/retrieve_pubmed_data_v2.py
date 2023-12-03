from Bio import Entrez
from Bio import Medline
import pandas as pd
import csv
import re


### The IDs of the articles are stored in the articles_ids.csv using Linux command:
### esearch -db pubmed -query "intelligence [title/abstract] hasabstract" | efetch -format uid >articles_ids.csv


def scraping_pubmed(email, ids_file):
    
    # Provide email address for PubMed to contact you in case of problems
    Entrez.email = email
    

    # Import the IDs of the articles exported using EDirect utilities  
    with open(ids_file, newline='') as f:
        reader = csv.reader(f)
        idlist = list(reader)
        
      
    # Fetch the articles in batches of 10000 (the maximum allowed by PubMed at onces)
    records = []
    for i in range(0, len(idlist), 10000):
        j = i + 10000
        if j >= len(idlist):
            j = len(idlist)
            
        handle=Entrez.efetch(db="pubmed", id=idlist[i:j],
                             rettype='medline', retmode='text')
        record=Medline.parse(handle)
 
        
        for r in record:
            records.append(r)    

    # Limit the scope of the retrieval to the period between 2013 and 2023 
    list_of_years = [str(year) for year in range(2013, 2023 + 1)]

    # Choosing the attributes to be extracted from PubMed       
    header = ['PMID', 'Title', 'Abstract', 'Key_words', 'Authors', 'Journal', 'Year', 'Month', 'Source','Country']
    
    with open('exported_data.csv', 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        
        # write the header
        writer.writerow(header)
        
        # write the data
        for paper in records:
            
            try:
                PMID = paper['PMID']
            except:
                PMID = None
                
            try:
                Title = paper['TI']
            except:
                try:
                    Title = paper['TT'] 
                except:
                    try:
                        Title = paper['BTI']
                    except:
                        Title = None
            
            try:
                Abstract = paper['AB']
            except:
                Abstract = None
            
            try:
                Key_words = paper['MH']
            except:
                try:
                    Key_words = paper['OT']
                except:
                    Key_words = None
                    
            try:
                Authors = paper['FAU']
            except:
                try:
                    Authors = paper['FED']
                except:
                    Authors = None
                
            try:
                Journal = paper['TA']
            except:
                Journal = None
        
            try:
                Year = paper['EDAT'].split('/')[0]   
                if Year not in list_of_years:
                    continue             
            except:
                Year = None
            
            try:
                Month = paper['EDAT'].split('/')[1]
            except:
                Month = None
            
            try:
                Source = paper['SO']
            except:
                Source = None
            
            try:
                Country = paper['AD'][0].split(',')[-1][:-1].lstrip()
                regex = re.compile('[\w\.-]+@[\w\.-]+(\.[\w]+)+')
                if re.search(regex, Country):
                    Country = Country.split(".")[0]
                else:
                    Country = Country              
                
            except:
                Country = None
            
            data = [PMID, Title, Abstract, Key_words, Authors, Journal, Year, Month, Source, Country]
            writer.writerow(data)
            
    df = pd.read_csv('exported_data.csv')
    print('A total of {} articles were retrieved from PubMed".'.format(df.shape[0]))

if __name__ == "__main__":
    scraping_pubmed('a-almasri@outlook.com', 'articles_ids.csv')