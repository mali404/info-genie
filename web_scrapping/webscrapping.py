import os
import glob
import pandas as pd
from tqdm import tqdm

import re
import requests
from bs4 import BeautifulSoup
from urllib3.util.retry import Retry
from requests.adapters import HTTPAdapter
from urllib.parse import urlparse


from unstructured.partition.auto import partition
from unstructured.chunking.title import chunk_by_title
from unstructured.cleaners.core import clean
from unstructured.cleaners.core import clean_ordered_bullets, clean_bullets, clean_non_ascii_chars  #tbd


class WebScraper:
    def __init__(self, file=None):
        self.file = file
     
    def scrape_select(self, url=None, url_hash=None):
        scraper_url = ScrapeAll(url=url, url_hash=url_hash)
        scraper_url.scrape_url()        

    def scrape(self):
        current_directory = os.getcwd()
        #here cwd is already chatbot
        scrape_dir = os.path.join(current_directory, 'scrapped_data') 

        if not os.path.exists(scrape_dir):
            os.mkdir(scrape_dir)            

        if self.file:
            df = pd.read_csv(self.file)
            urls_with_hashes = []
            os.chdir(scrape_dir)

            for index, row in tqdm(df.iterrows()):
                url = row['urls']
                url_hash = hash(url)
    
                # Extract data from the current URL and save it to a text file
                self.scrape_select(url, url_hash)
                urls_with_hashes.append([url, url_hash])
            df = pd.DataFrame(urls_with_hashes, columns=['urls', 'hashes'])
            df.to_csv(f"{scrape_dir}urls_with_hashes.csv", index=False) 

        path = os.path.join(new_directory, 'uploaded_docs')
        
        if not os.path.exists(path):
            os.mkdir(path) 

        os.chdir(path)

        # find all PDF files in the current directory
        file_extensions = ['*.pdf', '*.docx', '*.doc', "*.pptx", "*.csv", "*.epub", "*.md", "*.xlsx", "*.tsv", "*.rst", "*.rtf", "*.ppt", "*.odt", "*.json", "*.msg"]
        non_url_files = [file for extension in file_extensions for file in glob.glob(extension)]

        if non_url_files:
            scraper_no_url = ScrapeAll(file_list = non_url_files)
            scraper_no_url.scrape_non_url(non_url_files)            

        #now combining all the text files into one file
        output_file = "combined_all.txt"
    
        # Create a list to store the content of each file
        file_contents = []

        # Loop through all .txt files in the directory
        for filename in os.listdir(scrape_dir):
            if filename.endswith(".txt"):
                file_path = os.path.join(scrape_dir, filename)
                
                # Open and read the content of each .txt file
                with open(file_path, 'r', encoding='utf-8') as file:
                    file_contents.append(file.read())

        # Combine the contents of all files into one string
        combined_text = '\n'.join(file_contents)

        # Write the combined text to the output file
        output_file_path = os.path.join(scrape_dir, output_file)
        
        with open(output_file_path, 'w', encoding='utf-8') as output_file:
            output_file.write(combined_text)

        print(f"Combined text file for {scrape_dir} urls saved to {output_file}")

        #convert urls_with_hashes to dataframe
  
    def is_html(self, url):
    #Returns True if the response content is HTML, False otherwise.
        retry_strategy = Retry(
        total=8,  # Number of maximum retries
        backoff_factor=1,  # Exponential backoff factor
        status_forcelist=[500, 502, 503, 504],  # HTTP status codes to retry on
        )

        # Create an HTTP session with retry settings
        http = requests.Session()
        adapter = HTTPAdapter(max_retries=retry_strategy)
        http.mount("http://", adapter)
        http.mount("https://", adapter)

        try:
        # Send a GET request using the HTTP session
        #here 5 sec is connection_timeout where 27 sec is read time_out once the connection is established
            response = requests.get(url, timeout=(5, 27))
            content_type = response.headers['Content-Type']    
            pattern = "html"

            # Use re.search() to check if the pattern is in the string
            if re.search(pattern, content_type):
                return True
            else:
                return False
        except requests.exceptions.RequestException as e:
            print(f"Error while fetching URL: {url}. Exception: {e}")

    
    def optional_rm_headers_footers(self, url, new_list):
        if self.is_html(url):
            self.new_list = new_list[1:-1]
        else:
            pass
class ScrapeAll:
    def __init__(self, url_hash=None, url=None, file_list=None):
        self.url = url
        self.url_hash = url_hash
        self.file_list = file_list

    def scrape_url(self):
        retry_strategy = Retry(
            total=8,  # Number of maximum retries
            backoff_factor=1,  # Exponential backoff factor
            status_forcelist=[500, 502, 503, 504],  # HTTP status codes to retry on
        )

        # Create an HTTP session with retry settings
        http = requests.Session()
        adapter = HTTPAdapter(max_retries=retry_strategy)
        http.mount("http://", adapter)
        http.mount("https://", adapter)
        try:
            elements = partition(url=self.url, strategy='auto', new_after_n_chars=1600, request_timeout=(5, 27))
            self.chunks = chunk_by_title(elements)
            self.chunks_cleaned=[]
            for i in self.chunks:
                #have to check whether cleaning extra_whitespaces can disturb RecursiveCharacterTextSplitter
                # check @ https://unstructured-io.github.io/unstructured/core/cleaning.html
                #is_html() is designed to be used in ScrapeIIT class, have to change
                #to-do: change extra_whitespace to False
                j=clean(str(i), bullets=True, extra_whitespace=True, dashes=True)  
                k = clean_bullets(j)
                # Check if k is empty because clean_ordered_bullets() will throw an error if k is empty
                if not k:  
                    l=k
                else:
                    l = clean_ordered_bullets(k)
                m = clean_non_ascii_chars(l)
                self.chunks_cleaned.append(m)
                self.chunks_cleaned = list(dict.fromkeys(self.chunks_cleaned)) #remove duplicates

            self.new_list = []
            for i in range(len(self.chunks_cleaned)):
                if self.chunks_cleaned[i] and self.chunks_cleaned[i][0].isupper():
                    self.new_list.append(self.chunks_cleaned[i])
                elif self.new_list:  # Check if new_list is not empty
                    self.new_list[-1] += "" + self.chunks_cleaned[i]

            self.standard_list = []
            for i in self.new_list:
                j = f"sos: {i} \nInformation Source: {self.url}"
                self.standard_list.append(j)

            output_file_path = f'{self.url_hash}.txt'
            with open(output_file_path, 'w', encoding='utf-8') as output_file:
                for item in self.standard_list:
                    output_file.write(f"{item}\n\n")                

        except requests.exceptions.RequestException as e:
            print(f"Error while fetching URL: {self.url}. Exception: {e}")

        except ValueError as e:
            print(f"Error while fetching URL: {self.url}. Exception: {e}")

    def scrape_non_url(self, non_url_files):

        for non_url in non_url_files:
            with open(non_url, 'rb') as f:
                elements = partition(file=f, strategy='auto', new_after_n_chars=1600, request_timeout=(5, 27))
            self.chunks = chunk_by_title(elements)
            self.chunks_cleaned=[]
            for i in self.chunks:
                j=clean(str(i), bullets=True, extra_whitespace=True, dashes=True)  
                k = clean_bullets(j)
                if not k:  
                    l=k
                else:
                    l = clean_ordered_bullets(k)
                m = clean_non_ascii_chars(l)
                self.chunks_cleaned.append(m)
                self.chunks_cleaned = list(dict.fromkeys(self.chunks_cleaned)) #remove duplicates

            self.new_list = []
            for i in range(len(self.chunks_cleaned)):
                if self.chunks_cleaned[i] and self.chunks_cleaned[i][0].isupper():
                    self.new_list.append(self.chunks_cleaned[i])
                elif self.new_list:  # Check if new_list is not empty
                    self.new_list[-1] += "" + self.chunks_cleaned[i]

            self.standard_list = []
            for i in self.new_list:
                j = f"sos: {i}"
                self.standard_list.append(j)
            
            chatbot_dir = os.path.dirname(os.getcwd())
            # Create the path to the new directory
            scrape_dir = os.path.join(chatbot_dir, 'scrapped_data')
            #get the name of the file without the extension
            base_name = os.path.splitext(os.path.basename(non_url))[0]            
            output_file_path = f'{scrape_dir}/{base_name}.txt'

            with open(output_file_path, 'w', encoding='utf-8') as output_file:
                for item in self.standard_list:
                    output_file.write(f"{item}\n\n")

if __name__ == '__main__':
    current_directory = os.getcwd()
    # Create the path to the new directory
    new_directory = os.path.join(current_directory, 'chatbot')
    os.chdir(new_directory)
    scraper = WebScraper(file='urls_combined.csv')
    scraper.scrape()